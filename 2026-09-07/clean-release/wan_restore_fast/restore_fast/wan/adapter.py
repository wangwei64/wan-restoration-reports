"""External official Wan2.1 T2V-1.3B interface. No Wan source or weights are bundled."""

from pathlib import Path
import math, sys, time
import numpy as np
import torch
from ..config import external_paths, GenerationConfig


class OfficialWanAdapter:
    def __init__(self, repo=None, checkpoint=None, device="cuda", config=None):
        self.config = config or GenerationConfig()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU required; the verified profile uses BF16 and Triton."
            )
        self.repo, self.checkpoint = external_paths(repo, checkpoint)
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae import WanVAE

        wc = WAN_CONFIGS["t2v-1.3B"]
        required = [
            self.checkpoint / wc.t5_checkpoint,
            self.checkpoint / wc.t5_tokenizer,
            self.checkpoint / wc.vae_checkpoint,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Incomplete external Wan checkpoint: " + ", ".join(missing)
            )
        # Original FP32 DiT parameters + BF16 autocast, exactly as the chosen release.
        self.native_model = (
            WanModel.from_pretrained(str(self.checkpoint))
            .eval()
            .requires_grad_(False)
            .to(self.device)
        )
        if self.native_model.dim != 1536 or len(self.native_model.blocks) != 30:
            raise ValueError("This release is validated for Wan2.1 T2V-1.3B only.")
        self.official_text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=self.device,
            checkpoint_path=str(required[0]),
            tokenizer_path=str(required[1]),
        )
        self.vae = WanVAE(
            z_dim=16, vae_pth=str(required[2]), dtype=torch.float32, device=self.device
        )
        self.native_context = None
        self.native_negative_context = None

    def set_request(self, prompt, seed):
        self.prompt = prompt
        self.seed = seed

    def encode_prompt(self):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.native_context = self.official_text_encoder([self.prompt], self.device)
        self.native_negative_context = self.official_text_encoder(
            [self.config.negative_prompt], self.device
        )
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def initial_latents(self):
        c = self.config
        shape = (1, 16, (c.num_frames - 1) // 4 + 1, c.height // 8, c.width // 8)
        return torch.randn(
            shape,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
            device=self.device,
            dtype=torch.float32,
        )

    def scheduler(self):
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        s = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        s.set_timesteps(self.config.steps, device=self.device, shift=self.config.shift)
        return s

    @staticmethod
    def clean_estimate(latent, flow, sigma):
        return (
            latent.float()
            - sigma.to(device=latent.device, dtype=torch.float32) * flow.float()
        )

    @staticmethod
    def step(scheduler, flow, timestep, latent):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return scheduler.step(flow, timestep, latent, return_dict=False)[0]

    def predict(self, latent, idx, scheduler, controller=None):
        t = scheduler.timesteps[idx]
        n = math.prod(latent.shape[-3:]) // 4
        if controller is not None:
            controller.sigma = float(scheduler.sigmas[idx])
        if controller is not None and controller.batched_cfg:
            controller.branch = "joint"
            c, u = self.native_model(
                [latent[0], latent[0]],
                t=t.repeat(2),
                context=[self.native_context[0], self.native_negative_context[0]],
                seq_len=n,
            )
            return (u + self.config.guidance_scale * (c - u))[None].float()
        if controller is not None:
            controller.branch = "cond"
        c = self.native_model(
            [latent[0]], t=t[None], context=self.native_context, seq_len=n
        )[0]
        if controller is not None:
            controller.branch = "uncond"
        u = self.native_model(
            [latent[0]], t=t[None], context=self.native_negative_context, seq_len=n
        )[0]
        return (u + self.config.guidance_scale * (c - u))[None].float()

    def decode(self, latent):
        torch.cuda.synchronize()
        start = time.perf_counter()
        decoded = self.vae.decode([latent[0].to(self.device, dtype=torch.float32)])[0]
        video = decoded.detach().cpu().permute(1, 2, 3, 0).numpy()
        frames = [
            np.uint8(np.clip((frame + 1.0) * 127.5, 0.0, 255.0)) for frame in video
        ]
        torch.cuda.synchronize()
        return frames, time.perf_counter() - start
