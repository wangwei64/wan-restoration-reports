"""One reversible Wan adapter: prompt observation, selected queries, restored K/V.

No inheritance from historical periodic/coarse/grouped controllers. The Wan
model itself remains external and its original methods are restored by close().
"""

import types
import torch
from flash_attn import flash_attn_func
from wan.modules.model import rope_apply
from wan.modules.attention import flash_attention
from ..kernels.arithmetic import normalize, normalized_rotary, residual
from ..kernels.selected import selected_norm, selected_residual
from ..kernels.context import materialize_restoration_context, query_inverse
from ..subject import word_groups, automatic_saliency


def fixed_attention(q, k, v, window_size=(-1, -1)):
    return flash_attn_func(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), window_size=window_size
    )


class SparseWanController:
    def __init__(self, model):
        self.model = model
        self.original = []
        self.original_unpatchify = model.unpatchify
        self.closed = False
        self.reset()
        for i, b in enumerate(model.blocks):
            bf, sa, ca = b.forward, b.self_attn.forward, b.cross_attn.forward
            self.original.append((bf, sa, ca))
            b.forward = types.MethodType(self.block(i, bf), b)
            b.self_attn.forward = types.MethodType(self.attention(i, sa), b.self_attn)
            b.cross_attn.forward = types.MethodType(self.cross(i, ca), b.cross_attn)

        def unpatchify(m, x, grid_sizes):
            if self.mode == "sparse":
                full = x.new_zeros((x.shape[0], self.full_tokens, x.shape[-1]))
                full.index_copy_(1, self.indices, x)
                x = full
            return self.original_unpatchify(x, grid_sizes)

        model.unpatchify = types.MethodType(unpatchify, model)

    def reset(self):
        self.mode = "native"
        self.branch = "cond"
        self.slot = "left"
        self.sigma = 1.0
        self.stamps = {}
        self.anchors = {}
        self.indices = None
        self.full_tokens = None
        self.observe = False
        self.batched_cfg = False
        self.packed_k = None
        self.packed_v = None
        self.context_anchors = {}
        self.refresh_residual = {}
        self.text_kv = {}
        self.prompt_groups = []
        self.prompt_word_maps = []
        self.automatic_subject_report = None

    def close(self):
        if self.closed:
            return
        for b, (bf, sa, ca) in zip(self.model.blocks, self.original):
            b.forward = bf
            b.self_attn.forward = sa
            b.cross_attn.forward = ca
        self.model.unpatchify = self.original_unpatchify
        self.reset()
        self.closed = True

    def configure_prompt(self, runner, prompt):
        tokenizer = runner.official_text_encoder.tokenizer
        ids = tokenizer([prompt], return_mask=False)[0]
        self.prompt_length = len(runner.native_context[0])
        tokens = tokenizer.tokenizer.convert_ids_to_tokens(ids.tolist())[
            : self.prompt_length
        ]
        self.prompt_groups = word_groups(tokens)

    def subject(self, probability):
        saliency, details = automatic_saliency(self.prompt_word_maps, probability)
        words = [
            {**g, "weight": w, "foreground_agreement": a}
            for g, w, a in zip(
                self.prompt_groups, details["weights"], details["agreement"]
            )
        ]
        self.automatic_subject_report = {
            "mode": "automatic_prompt_latent_agreement",
            "manual_subject_supplied": False,
            "extra_model": False,
            "new_training": False,
            "fallback": details["fallback"],
            "words": words,
            "weight_rule": "squared positive foreground/background relevance contrast, normalized across prompt word groups",
            "observation_layers_zero_based": [5, 10, 15, 20],
        }
        return saliency

    def partition(self, token_mask, semantic):
        active = token_mask.flatten().bool()
        subject = semantic[..., ::2, ::2].flatten().bool()
        self.subject_idx = torch.nonzero(active & subject, as_tuple=False).flatten()
        self.low_idx = torch.nonzero(active & ~subject, as_tuple=False).flatten()
        self.restore_idx = torch.nonzero(~active, as_tuple=False).flatten()
        self.all_idx = torch.cat([self.subject_idx, self.low_idx])
        self.indices = self.all_idx
        self.packed_indices = torch.cat([self.all_idx, self.restore_idx])
        self.total_count = active.numel()

    def enable_batched_cfg(self):
        combined = {}
        for i in range(len(self.model.blocks)):
            c, u = self.anchors[(i, "cond")], self.anchors[(i, "uncond")]
            combined[(i, "joint")] = {
                s: tuple(torch.cat([a, b], 0) for a, b in zip(c[s], u[s])) for s in c
            }
        self.anchors = combined
        self.batched_cfg = True

    def select(self, due):
        positions = torch.nonzero(
            due.index_select(0, self.packed_indices), as_tuple=False
        ).flatten()
        self.indices = self.packed_indices[positions]
        self.context_query_map = query_inverse(positions, self.total_count)

    def block(self, i, original):
        def forward(block, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            if self.mode != "sparse":
                return original(
                    x, e, seq_lens, grid_sizes, freqs, context, context_lens
                )
            if i == 0:
                self.full_tokens = x.shape[1]
            es = (block.modulation + e).chunk(6, dim=1)
            xn = (
                selected_norm(x, self.indices, block.norm1, es[1], es[0])
                if i == 0
                else normalize(x, block.norm1, es[1], es[0])
            )
            y = block.self_attn(xn, seq_lens, grid_sizes, freqs)
            x = (
                selected_residual(x, self.indices, y, es[2])
                if i == 0
                else residual(x, y, es[2])
            )
            xn = normalize(x, block.norm3) if hasattr(block.norm3, "eps") else x
            x = residual(x, block.cross_attn(xn, context, context_lens))
            y = block.ffn(normalize(x, block.norm2, es[4], es[3]))
            return residual(x, y, es[5])

        return forward

    def attention(self, i, original):
        def forward(attn, x, seq_lens, grid_sizes, freqs):
            if self.mode == "native":
                return original(x, seq_lens, grid_sizes, freqs)
            b, n = x.shape[:2]
            key = (i, self.branch)
            if self.mode == "capture":
                q = attn.norm_q(attn.q(x)).view(b, n, attn.num_heads, attn.head_dim)
                k = attn.norm_k(attn.k(x)).view(b, n, attn.num_heads, attn.head_dim)
                v = attn.v(x).view(b, n, attn.num_heads, attn.head_dim)
                q = rope_apply(q, grid_sizes, freqs)
                k = rope_apply(k, grid_sizes, freqs)
                self.anchors.setdefault(key, {})[self.slot] = (
                    k.to(torch.bfloat16),
                    v.to(torch.bfloat16),
                )
                self.stamps[self.slot] = self.sigma
                return attn.o(
                    flash_attention(
                        q=q, k=k, v=v, k_lens=seq_lens, window_size=attn.window_size
                    ).flatten(2)
                )
            q = normalized_rotary(
                attn.q(x), attn.norm_q, attn.head_dim, grid_sizes, freqs, self.indices
            )
            k = normalized_rotary(
                attn.k(x), attn.norm_k, attn.head_dim, grid_sizes, freqs, self.indices
            )
            v = attn.v(x).view(*x.shape[:2], attn.num_heads, attn.head_dim)
            if self.packed_k is None:
                self.packed_k = torch.empty(
                    (b, self.total_count, attn.num_heads, attn.head_dim),
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                self.packed_v = torch.empty_like(self.packed_k)
            if key not in self.context_anchors:
                states = self.anchors.pop(key)
                self.context_anchors[key] = {
                    s: (
                        ak.index_select(1, self.packed_indices),
                        av.index_select(1, self.packed_indices),
                    )
                    for s, (ak, av) in states.items()
                }
                kl, vl = self.context_anchors[key]["left"]
                a = len(self.all_idx)
                self.refresh_residual[key] = (
                    torch.zeros_like(kl[:, :a]),
                    torch.zeros_like(vl[:, :a]),
                )
            kl, vl = self.context_anchors[key]["left"]
            kr, vr = self.context_anchors[key]["right"]
            dk, dv = self.refresh_residual[key]
            fraction = max(
                0.0,
                min(
                    1.0,
                    (self.sigma - self.stamps["right"])
                    / (self.stamps["left"] - self.stamps["right"]),
                ),
            )
            materialize_restoration_context(
                kl,
                vl,
                kr,
                vr,
                dk,
                dv,
                k,
                v,
                self.context_query_map,
                self.packed_k,
                self.packed_v,
                fraction,
                0,
            )
            return attn.o(
                fixed_attention(
                    q, self.packed_k, self.packed_v, attn.window_size
                ).flatten(2)
            )

        return forward

    def cross(self, i, original):
        def forward(attn, x, context, context_lens):
            if self.observe and self.branch == "cond" and i in (5, 10, 15, 20):
                b, n, d = x.shape[0], attn.num_heads, attn.head_dim
                q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
                k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
                v = attn.v(context).view(b, -1, n, d)
                if self.prompt_groups:
                    with torch.autocast("cuda", enabled=False):
                        centered = torch.stack(
                            [
                                k[0, g["indices"]].float().mean(0)
                                for g in self.prompt_groups
                            ]
                        ) - k[0, : self.prompt_length].float().mean(0)
                        score = torch.bmm(
                            q[0].float().permute(1, 0, 2), centered.permute(1, 2, 0)
                        ).mean(0)
                        lo, hi = torch.quantile(
                            score,
                            torch.tensor([0.05, 0.95], device=score.device),
                            dim=0,
                        )
                        self.prompt_word_maps.append(
                            ((score - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
                        )
                return attn.o(flash_attention(q, k, v, k_lens=context_lens).flatten(2))
            if self.mode != "sparse":
                return original(x, context, context_lens)
            assert context_lens is None
            b = x.shape[0]
            heads = attn.num_heads
            dim = attn.head_dim
            q = normalize(attn.q(x), attn.norm_q, rms=True).view(b, -1, heads, dim)
            key = (i, self.branch)
            if key not in self.text_kv:
                self.text_kv[key] = (
                    attn.norm_k(attn.k(context)).view(b, -1, heads, dim).bfloat16(),
                    attn.v(context).view(b, -1, heads, dim).bfloat16(),
                )
            k, v = self.text_kv[key]
            return attn.o(fixed_attention(q, k, v).flatten(2))

        return forward
