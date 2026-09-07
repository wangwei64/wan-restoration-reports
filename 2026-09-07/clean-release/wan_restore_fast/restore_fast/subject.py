"""Automatic prompt/latent agreement. No labels, masks, categories or LLM calls."""

import re
import torch
from torch.nn import functional as F

# Grammatical stop words, not a subject/class dictionary. Visual descriptors
# and action words remain eligible and are judged by spatial evidence.
FUNCTION_WORDS = set(
    "a an the and or but if then than as at by for from in into of on onto over to under up out with without is are was were be been being am do does did has have had its their his her our your my it they them he she we you i this that these those there here some any each every no not very more most less much many".split()
)


def word_groups(tokens):
    groups = []
    current = []
    pieces = []

    def flush():
        if not current:
            return
        word = "".join(pieces).strip()
        word = re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)
        if any(c.isalpha() for c in word) and word.casefold() not in FUNCTION_WORDS:
            groups.append({"word": word, "indices": current.copy()})
        current.clear()
        pieces.clear()

    for i, token in enumerate(tokens):
        if token.startswith("<") and token.endswith(">"):
            flush()
            continue
        start = token.startswith(("▁", "Ġ"))
        if start:
            flush()
        piece = token.lstrip("▁Ġ")
        if not piece:
            continue
        if not any(c.isalnum() for c in piece):
            flush()
            continue
        current.append(i)
        pieces.append(piece)
    flush()
    merged = {}
    for g in groups:
        key = g["word"].casefold()
        if key in merged:
            merged[key]["indices"].extend(g["indices"])
        else:
            merged[key] = g
    return list(merged.values())


def automatic_saliency(maps, head_probability):
    """Soft spatial agreement, no top-k classes and no manually named subject."""
    p = (
        F.avg_pool3d(head_probability.float(), (1, 2, 2), (1, 2, 2))
        .flatten()
        .clamp(0, 1)
    )
    if not maps or maps[0].shape[1] == 0:
        return torch.full_like(p, 0.5), {
            "fallback": "no usable prompt words; neutral text conditioning",
            "weights": [],
            "agreement": [],
        }
    m = torch.stack(maps).mean(0)
    inside = (m * p[:, None]).sum(0) / p.sum().clamp_min(1e-6)
    outside = (m * (1 - p[:, None])).sum(0) / (1 - p).sum().clamp_min(1e-6)
    agreement = inside - outside
    mass = agreement.clamp_min(0).square()
    if float(mass.sum()) <= 1e-12:
        return torch.full_like(p, 0.5), {
            "fallback": "no positive foreground agreement; neutral text conditioning",
            "weights": mass.tolist(),
            "agreement": agreement.tolist(),
        }
    weights = mass / mass.sum()
    saliency = (m * weights[None]).sum(1)
    return saliency, {
        "fallback": None,
        "weights": weights.tolist(),
        "agreement": agreement.tolist(),
    }
