"""Fixed trained restoration architecture, with the unused legacy quality head removed.

All parameters are loaded from one flat safetensors file. No historical checkpoint,
training repository, image encoder or detector is constructed at inference time.
"""

import json
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from .blocks import (
    LayerNorm3d,
    TemporalNAFBlock,
    TemporalResidualFusion,
    SpatialPixelShuffle,
)


class RestorationBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(**config)
        c = self.config
        self.intro = nn.Conv3d(16, c.width, (1, 3, 3), padding=(0, 1, 1))
        self.ending = nn.Conv3d(c.width, 16, (1, 3, 3), padding=(0, 1, 1))
        self.output_norm = LayerNorm3d(c.width)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.heat_encoders = nn.ModuleList()
        self.heat_decoders = nn.ModuleList()
        ch = c.width
        for n in c.encoder_blocks:
            self.heat_encoders.append(nn.Conv3d(1, ch, 1))
            self.encoders.append(
                nn.Sequential(*[TemporalNAFBlock(ch) for _ in range(n)])
            )
            self.downs.append(nn.Conv3d(ch, 2 * ch, (1, 2, 2), stride=(1, 2, 2)))
            ch *= 2
        self.middle_blks = nn.Sequential(
            *[TemporalNAFBlock(ch) for _ in range(c.middle_blocks)]
        )
        for n in c.decoder_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv3d(ch, ch * 2, 1, bias=False), SpatialPixelShuffle(2)
                )
            )
            ch //= 2
            self.heat_decoders.append(nn.Conv3d(1, ch, 1))
            self.decoders.append(
                nn.Sequential(*[TemporalNAFBlock(ch) for _ in range(n)])
            )
        self.padder_size = 2 ** len(c.encoder_blocks)
        dw = c.detail_width
        self.detail_base = nn.Conv3d(c.width, dw, 1)
        self.detail_latent = nn.Conv3d(112, dw, 1)
        self.detail_heat = nn.Conv3d(1, dw, 1)
        self.detail_blks = nn.Sequential(
            *[TemporalNAFBlock(dw) for _ in range(c.detail_blocks)]
        )
        self.detail_temporal_blks = nn.Sequential(
            *[
                TemporalResidualFusion(dw, dilation=1 if i % 2 == 0 else 2)
                for i in range(c.detail_temporal_fusion_blocks)
            ]
        )
        self.detail_norm = LayerNorm3d(dw)
        self.detail_ending = nn.Conv3d(dw, 16, (1, 3, 3), padding=(0, 1, 1))

    def forward(self, z10, heat):
        c = self.config
        height, width = z10.shape[-2:]
        ph = (-height) % self.padder_size
        pw = (-width) % self.padder_size
        z = F.pad(z10, (0, pw, 0, ph, 0, 0), mode="reflect") if ph or pw else z10
        heat = F.pad(heat, (0, pw, 0, ph, 0, 0), mode="replicate") if ph or pw else heat
        heat = heat.to(device=z10.device, dtype=z10.dtype)
        x = self.intro(z)
        skips = []
        for enc, down, hproj in zip(self.encoders, self.downs, self.heat_encoders):
            x = x + hproj(F.adaptive_avg_pool3d(heat, x.shape[-3:]))
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip, hproj in zip(
            self.decoders, self.ups, reversed(skips), self.heat_decoders
        ):
            x = up(x) + skip
            x = x + hproj(F.adaptive_avg_pool3d(heat, x.shape[-3:]))
            x = dec(x)
        base = self.ending(self.output_norm(x)) * c.residual_scale
        high = lambda k: z - F.avg_pool3d(
            z, (1, k, k), stride=1, padding=(0, k // 2, k // 2), count_include_pad=False
        )
        # Keep the former evaluation order and seven channels-of-bands exactly.
        h5 = high(5)
        h3 = high(3)
        gx = F.pad(z[..., 1:] - z[..., :-1], (0, 1, 0, 0, 0, 0), mode="replicate")
        gy = F.pad(z[..., 1:, :] - z[..., :-1, :], (0, 0, 0, 1, 0, 0), mode="replicate")
        h9 = high(9)
        prev = torch.cat((z[:, :, :1], z[:, :, :-1]), dim=2)
        nxt = torch.cat((z[:, :, 1:], z[:, :, -1:]), dim=2)
        bands = torch.cat(
            (h3, h5, h9, gx, gy, 0.5 * (nxt - prev), nxt - 2.0 * z + prev), dim=1
        )
        hp = torch.cat((heat[:, :, :1], heat[:, :, :-1]), dim=2)
        hn = torch.cat((heat[:, :, 1:], heat[:, :, -1:]), dim=2)
        hs = (
            1.0 - c.detail_heatmap_temporal_smoothing
        ) * heat + c.detail_heatmap_temporal_smoothing * (
            0.25 * hp + 0.50 * heat + 0.25 * hn
        )
        detail = self.detail_base(x) + self.detail_latent(bands) + self.detail_heat(hs)
        detail = self.detail_blks(detail)
        detail = self.detail_temporal_blks(detail)
        residual = self.detail_ending(self.detail_norm(detail))
        gate = c.detail_gate_floor + (1.0 - c.detail_gate_floor) * hs
        detail = c.detail_residual_scale * gate * residual
        return base[..., :height, :width], detail[..., :height, :width]


class FeatureRestorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config["adapter"]
        w = cfg["width"]
        self.backbone = RestorationBackbone(config["backbone"])
        self.image_encoder = nn.Sequential(
            nn.Conv3d(46, 32, 1),
            nn.SiLU(),
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1)),
            nn.SiLU(),
        )
        self.intro = nn.Conv3d(80, w, 1)
        self.body = nn.Sequential(*[TemporalNAFBlock(w) for _ in range(cfg["blocks"])])
        self.norm = LayerNorm3d(w)
        self.correction = nn.Conv3d(w, 16, 1)
        self.detail_gate = nn.Conv3d(w, 1, 1)
        self.quality = nn.Sequential(
            nn.Conv3d(w + 32 + 16, 48, 1),
            nn.SiLU(),
            TemporalNAFBlock(48),
            TemporalNAFBlock(48),
            LayerNorm3d(48),
            nn.Conv3d(48, 2, 1),
        )

    def forward(self, z10, features, heat):
        base, detail = self.backbone(z10, heat)
        image = self.image_encoder(features.to(z10))
        hidden = self.norm(
            self.body(self.intro(torch.cat([z10, base, detail, image], 1)))
        )
        gate = 2 * torch.sigmoid(self.detail_gate(hidden))
        delta = 0.05 * self.correction(hidden)
        # Do not algebraically regroup: preserve the original FP32 rounding.
        restored = z10 + base + detail + 1.0 * ((gate - 1) * detail + delta)
        q = self.quality(
            torch.cat(
                [hidden.detach(), image.detach(), (restored - z10).detach().abs()], 1
            )
        ).float()
        expected = 0.04 * F.softplus(q[:, :1])
        cfg = self.config["adapter"]
        risk = torch.sigmoid(
            (expected - cfg["rgb_error_tolerance"]) / cfg["rgb_error_softness"]
        )
        return (
            restored,
            1 - risk,
            {"expected_rgb_error": expected, "latent_risk": torch.sigmoid(q[:, 1:])},
        )


class LatentFeatures(nn.Module):
    def __init__(self, width=48, blocks=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(16, width, 3, padding=1),
            *[TemporalNAFBlock(width) for _ in range(blocks)],
            LayerNorm3d(width),
            nn.Conv3d(width, 47, 1),
        )

    def forward(self, z):
        x = self.net(z)
        return x[:, :46].float(), x[:, 46:].sigmoid().float()


class SubjectHead(nn.Module):
    def __init__(self, width=32, blocks=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(63, width, 3, padding=1),
            *[TemporalNAFBlock(width) for _ in range(blocks)],
            LayerNorm3d(width),
            nn.Conv3d(width, 1, 1),
        )

    def forward(self, z, f, h):
        return self.net(torch.cat([z, f.to(z), h.to(z)], 1)).float()


def load_models(directory, device="cuda"):
    directory = Path(directory)
    models = []
    for name, cls in [
        ("restorer", FeatureRestorer),
        ("features", LatentFeatures),
        ("subject", SubjectHead),
    ]:
        config = json.loads((directory / f"{name}.json").read_text())
        m = cls(config) if name == "restorer" else cls(**config)
        m.load_state_dict(
            load_file(str(directory / f"{name}.safetensors")), strict=True
        )
        models.append(m.to(device).eval().requires_grad_(False))
    return tuple(models)
