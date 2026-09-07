"""The single supported route: Wan10 -> restoration -> adaptive Wan11..50 -> fusion."""

import copy, hashlib, json, time
from pathlib import Path
import torch
from safetensors.torch import save_file
from .config import PACKAGE_ROOT, GenerationConfig
from .models.restorer import load_models
from .routing import route
from .refresh import RefinementState
from .video import write_video


class RestorationPipeline:
    def __init__(self, wan=None, weights=None, config=None, repo=None, checkpoint=None):
        start = time.perf_counter()
        self.config = config or GenerationConfig()
        if (self.config.steps, self.config.warmup_steps) != (50, 10):
            raise ValueError(
                "The released architecture requires exactly 10 warmup and 50 total steps."
            )
        if wan is None:
            from .wan.adapter import OfficialWanAdapter

            wan = OfficialWanAdapter(repo, checkpoint, config=self.config)
        self.wan = wan
        self.weights = Path(weights) if weights else PACKAGE_ROOT / "weights"
        self.restorer, self.features, self.subject_head = load_models(
            self.weights, self.wan.device
        )
        self.calibration = json.loads((self.weights / "calibration.json").read_text())
        from .wan.controller import SparseWanController

        self.controller = SparseWanController(self.wan.native_model)
        self.load_seconds = time.perf_counter() - start

    def close(self):
        self.controller.close()

    @torch.inference_mode()
    def generate(self, prompt, seed=None, out=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must contain text")
        seed = self.config.default_seed if seed is None else seed
        if out is None:
            out = (
                PACKAGE_ROOT
                / "outputs"
                / f'{time.strftime("%Y%m%d_%H%M%S")}_{hashlib.sha256((prompt+str(time.time_ns())).encode()).hexdigest()[:8]}'
            )
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        runner = self.wan
        sparse = self.controller
        sparse.reset()
        runner.set_request(prompt, seed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        stages = {}
        observations = []
        steps = []
        stages["text_encode"] = runner.encode_prompt()
        sparse.configure_prompt(runner, prompt)
        latent = runner.initial_latents()
        scheduler = runner.scheduler()
        warm = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for idx in range(10):
                if idx == 9:
                    sparse.mode = "capture"
                    sparse.slot = "left"
                    sparse.observe = True
                flow = runner.predict(latent, idx, scheduler, sparse)
                clean = runner.clean_estimate(latent, flow, scheduler.sigmas[idx])
                if idx == 9:
                    warm = (copy.deepcopy(scheduler), latent.clone(), flow.clone())
                latent = runner.step(scheduler, flow, scheduler.timesteps[idx], latent)
                if idx == 8:
                    z9 = clean.clone()
                steps.append({"step": idx + 1, "active_ratio": 1.0})
            sparse.observe = False
            z10 = clean.clone()
            post10 = latent.clone()
            torch.cuda.synchronize()
            stages["native_warmup10_and_localization"] = time.perf_counter() - tick
            tick = time.perf_counter()
            f, h = self.features(z10)
            endpoint, confidence, aux = self.restorer(z10, f, h)
            endpoint = endpoint.float()
            confidence = confidence.float()
            torch.cuda.synchronize()
            stages["features_restoration_confidence"] = time.perf_counter() - tick
            tick = time.perf_counter()
            probability = self.subject_head(z10, f, h).sigmoid()
            saliency = sparse.subject(probability)
            active, token_mask, semantic, routing_info, routing_tensors = route(
                z10,
                z9,
                aux["expected_rgb_error"],
                saliency,
                h,
                probability,
                self.calibration,
                self.config.quality_tolerance,
            )
            torch.cuda.synchronize()
            stages["routing"] = time.perf_counter() - tick
            sparse.partition(token_mask, semantic)
            sparse.enable_batched_cfg()
            tick = time.perf_counter()
            refinement = RefinementState(
                warm,
                scheduler,
                post10,
                endpoint,
                h,
                {**routing_tensors, "latent_risk": aux["latent_risk"]},
                token_mask,
                sparse.subject_idx,
                sparse.low_idx,
                self.config.refinement_budget,
            )
            warm = None
            if len(sparse.indices):
                sparse.mode = "capture"
                sparse.slot = "right"
                torch.cuda.synchronize()
                observe_start = time.perf_counter()
                runner.predict(endpoint, 49, scheduler, sparse)
                torch.cuda.synchronize()
                observations.append(
                    {
                        "at_step": 50,
                        "slot": "right",
                        "seconds": time.perf_counter() - observe_start,
                    }
                )
            sparse.mode = "sparse"
            torch.cuda.synchronize()
            stages["solver_and_restored_context"] = time.perf_counter() - tick
            tick = time.perf_counter()
            for idx in range(10, 50):
                sparse.select(refinement.select(idx))
                if len(sparse.indices):
                    flow = runner.predict(latent, idx, scheduler, sparse)
                    refinement.update(flow, latent, idx, sparse.indices)
                next_latent = refinement.dense(scheduler.sigmas[idx + 1])
                restored_state = endpoint + (post10 - endpoint) * (
                    scheduler.sigmas[idx + 1] / scheduler.sigmas[10]
                )
                latent = active * next_latent + (1 - active) * restored_state
                steps.append(
                    {
                        "step": idx + 1,
                        "active_ratio": len(sparse.indices) / token_mask.numel(),
                    }
                )
            torch.cuda.synchronize()
            stages["sparse_iterations"] = time.perf_counter() - tick
            # Binary ownership already gives exact normalized, disjoint weights.
            wan_weight = active
            ws = wan_weight * semantic
            wl = wan_weight * (1 - semantic)
            wr = 1 - wan_weight
            final = wan_weight * latent + wr * endpoint
        sparse.mode = "native"
        frames, stages["final_vae_decode"] = runner.decode(final)
        tick = time.perf_counter()
        write_video(out / "final.mp4", frames, self.config.fps)
        stages["video_encoding"] = time.perf_counter() - tick
        torch.cuda.synchronize()
        online = time.perf_counter() - started
        report = {
            "prompt": prompt,
            "seed": seed,
            "online_seconds": online,
            "model_load_seconds": self.load_seconds,
            "stages": stages,
            "steps": steps,
            "context_observations": observations,
            "profile": self.config.to_dict(),
            "automatic_subject_selection": sparse.automatic_subject_report,
            "subject_training_labels": "offline YOLOE teacher masks; no online detector",
            "routing": routing_info,
            "active_ratio": sum(s["active_ratio"] for s in steps[10:]) / 40,
            "initial_active_ratio": float(active.mean()),
            "weight_sum_max_error": float((ws + wl + wr - 1).abs().max()),
            "weight_overlap_max": float((ws * wl).abs().max()),
            "external_segmentation": False,
            "vae_preview_decodes": 0,
            "extra_tail_steps": [],
            "new_models": False,
            "new_training": False,
            "gt_inputs": False,
            "legacy_quality_head_removed": True,
            "unused_diffusers_dit_removed": True,
            **refinement.report(),
        }
        assert report["weight_sum_max_error"] == 0 and report["weight_overlap_max"] == 0
        tensors = {
            "latent": final,
            "wan": latent,
            "restored": endpoint,
            "x0_step10": z10,
            "x0_step9": z9,
            "confidence": confidence,
            "expected_error": aux["expected_rgb_error"],
            "subject_weight": ws,
            "low_confidence_weight": wl,
            "restored_weight": wr,
            "active": active,
            "subject_saliency": semantic,
            **routing_tensors,
            **refinement.tensors(),
        }
        save_file(
            {k: v[0].cpu().contiguous() for k, v in tensors.items()},
            str(out / "outputs.safetensors"),
        )
        (out / "run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "directory": str(out.resolve()),
            "video": str((out / "final.mp4").resolve()),
            "report": report,
        }

    @torch.inference_mode()
    def native_reference(self, prompt, seed, out):
        """Fresh same-model/same-seed native GT50 for verification only."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        self.controller.reset()
        runner = self.wan
        runner.set_request(prompt, seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        stages = {"text_encode": runner.encode_prompt()}
        latent = runner.initial_latents()
        scheduler = runner.scheduler()
        tick = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for idx in range(50):
                flow = runner.predict(latent, idx, scheduler)
                latent = runner.step(scheduler, flow, scheduler.timesteps[idx], latent)
        torch.cuda.synchronize()
        stages["native_steps50"] = time.perf_counter() - tick
        frames, stages["final_vae_decode"] = runner.decode(latent)
        tick = time.perf_counter()
        write_video(out / "final.mp4", frames, self.config.fps)
        stages["video_encoding"] = time.perf_counter() - tick
        torch.cuda.synchronize()
        r = {
            "prompt": prompt,
            "seed": seed,
            "online_seconds": time.perf_counter() - start,
            "stages": stages,
            "native_steps": 50,
            "model_loading_included": False,
        }
        save_file(
            {"latent": latent[0].cpu().contiguous()}, str(out / "outputs.safetensors")
        )
        (out / "run.json").write_text(json.dumps(r, indent=2))
        return r
