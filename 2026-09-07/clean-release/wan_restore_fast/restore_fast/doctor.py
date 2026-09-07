"""Read-only release checks. Does not load Wan or start generation."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from .config import PACKAGE_ROOT, external_paths


def check(repo=None, checkpoint=None):
    errors = []
    versions = {}
    modules = {
        "torch": "torch",
        "triton": "triton",
        "flash-attn": "flash_attn",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "safetensors": "safetensors",
        "opencv-python": "cv2",
        "imageio-ffmpeg": "imageio_ffmpeg",
        "sentencepiece": "sentencepiece",
        "einops": "einops",
        "ftfy": "ftfy",
        "easydict": "easydict",
        "imageio": "imageio",
        "tqdm": "tqdm",
    }
    for distribution, module in modules.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing Python module: {module}")
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    gpu = None
    if importlib.util.find_spec("torch"):
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("CUDA GPU unavailable")
            elif not torch.cuda.is_bf16_supported():
                errors.append("GPU does not support BF16")
            else:
                gpu = {
                    "name": torch.cuda.get_device_name(),
                    "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "cuda": torch.version.cuda,
                }
        except (ImportError, OSError) as e:
            errors.append(
                f"PyTorch cannot load; use the documented Linux CUDA environment: {e}"
            )
    locations = None
    try:
        source, weights = external_paths(repo, checkpoint)
        locations = {"source": str(source), "checkpoint": str(weights)}
        for name in [
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/umt5-xxl/tokenizer_config.json",
            "diffusion_pytorch_model.safetensors",
        ]:
            if not (weights / name).exists():
                errors.append(f"Missing external Wan file: {name}")
        verified = PACKAGE_ROOT / "tests/environment.json"
        if verified.exists():
            expected = json.loads(verified.read_text(encoding="utf-8"))
            for name, digest in expected["wan_core_sha256"].items():
                if (
                    not (source / name).exists()
                    or hashlib.sha256((source / name).read_bytes()).hexdigest()
                    != digest
                ):
                    errors.append(
                        f"External Wan core differs from verified source: {name}"
                    )
    except FileNotFoundError as e:
        errors.append(str(e))
    checked = []
    try:
        manifest = json.loads(
            (PACKAGE_ROOT / "weights/manifest.json").read_text(encoding="utf-8")
        )
        for record in manifest["files"]:
            path = PACKAGE_ROOT / "weights" / record["file"]
            ok = (
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            )
            checked.append({"file": record["file"], "verified": ok})
            if not ok:
                errors.append(
                    f"Packaged weight/config failed verification: {record['file']}"
                )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        errors.append(f"Weight manifest unreadable: {e}")
    return {
        "ready": not errors,
        "platform": platform.platform(),
        "gpu": gpu,
        "external_wan": locations,
        "versions": versions,
        "packaged_weights": checked,
        "errors": errors,
    }
