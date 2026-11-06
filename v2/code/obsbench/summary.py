"""Seed-level summaries: mean, sample std, and t-based 95% CI across seeds."""

from __future__ import annotations

import numpy as np
from scipy import stats

CI_LEVEL = 0.95


def summarize(values) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if v.size > 1 else 0.0
    half = float(stats.t.ppf(1 - (1 - CI_LEVEL) / 2, v.size - 1) * std / np.sqrt(v.size)) if v.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci_lo": mean - half, "ci_hi": mean + half, "n": int(v.size)}


def rounded(obj, digits: int = 4):
    """Round floats recursively so results files carry the precision the paper reports."""
    if isinstance(obj, float):
        return round(obj, digits) if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [rounded(v, digits) for v in obj]
    if isinstance(obj, np.floating):
        return rounded(float(obj), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
