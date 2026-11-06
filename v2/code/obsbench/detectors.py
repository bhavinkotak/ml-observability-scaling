"""Univariate drift detectors with explicit outcome handling.

Binned detectors (PSI, KL, chi-square) use reference-quantile bins (numeric) or the reference
vocabulary plus one out-of-vocabulary bin (categorical). Each detector maps a reference sample and a
batch of windows to a score where larger means more drift; KS and chi-square also return p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

from .config import N_BINS, N_MIN, SMOOTH

NUMERIC_DETECTORS = ("psi", "kl", "chi2", "ks", "w1")
CATEGORICAL_DETECTORS = ("psi", "kl", "chi2")


def quantile_edges(ref: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Interior bin edges at reference quantiles b/B; duplicate edges (ties) are merged."""
    return np.unique(np.quantile(ref, np.arange(1, n_bins) / n_bins))


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-closed bins (-inf, e1], (e1, e2], ..., (e_k, inf)."""
    return np.searchsorted(edges, x, side="left")


def row_counts(idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-row bin counts for a (R, n) integer index matrix."""
    r = idx.shape[0]
    flat = (idx + n_bins * np.arange(r)[:, None]).ravel()
    return np.bincount(flat, minlength=r * n_bins).reshape(r, n_bins)


def binned_scores(ref_counts: np.ndarray, win_counts: np.ndarray, smooth: float = SMOOTH) -> dict[str, np.ndarray]:
    """PSI and KL on smoothed proportions; Pearson chi-square on raw 2 x B counts."""
    win_counts = np.atleast_2d(win_counts).astype(float)
    ref_counts = ref_counts.astype(float)
    b = ref_counts.size
    m = ref_counts.sum()
    n = win_counts.sum(axis=1, keepdims=True)
    r = (ref_counts + smooth) / (m + smooth * b)
    c = (win_counts + smooth) / (n + smooth * b)
    log_ratio = np.log(c / r)
    psi = ((c - r) * log_ratio).sum(axis=1)
    kl = (c * log_ratio).sum(axis=1)
    tot = ref_counts + win_counts
    keep = tot > 0
    e_ref = tot * m / (m + n)
    e_win = tot * n / (m + n)
    with np.errstate(divide="ignore", invalid="ignore"):
        cells = np.where(keep, (ref_counts - e_ref) ** 2 / e_ref + (win_counts - e_win) ** 2 / e_win, 0.0)
    x2 = cells.sum(axis=1)
    df = keep.sum(axis=1) - 1
    p = np.where(df >= 1, stats.chi2.sf(x2, np.maximum(df, 1)), 1.0)
    return {"psi": psi, "kl": kl, "chi2_stat": x2, "chi2_p": p}


def ks_w1(ref_sorted: np.ndarray, window: np.ndarray, scale: float) -> tuple[float, float]:
    """Exact two-sample KS statistic and Wasserstein-1 distance (divided by `scale`); handles ties."""
    w = np.sort(window)
    pooled = np.sort(np.concatenate([ref_sorted, w]))
    f_win = np.searchsorted(w, pooled, side="right") / w.size
    f_ref = np.searchsorted(ref_sorted, pooled, side="right") / ref_sorted.size
    diff = np.abs(f_win - f_ref)
    return float(diff.max()), float((diff[:-1] * np.diff(pooled)).sum() / scale)


@lru_cache(maxsize=64)
def _ks_table(n: int, m: int) -> np.ndarray:
    lcm = int(np.lcm(n, m))
    return stats.kstwo.sf(np.arange(lcm + 1) / lcm, np.round(n * m / (n + m)))


def ks_pvalue(d: np.ndarray, n: int, m: int) -> np.ndarray:
    """Asymptotic two-sided p-value, matching scipy.stats.ks_2samp(method='asymp').

    D is a multiple of 1/lcm(n, m), so p-values are read from a precomputed table (same values, much faster).
    """
    lcm = int(np.lcm(n, m))
    if lcm > 1_000_000:
        return stats.kstwo.sf(d, np.round(n * m / (n + m)))
    return _ks_table(n, m)[np.rint(np.asarray(d) * lcm).astype(np.int64)]


@dataclass
class Reference:
    """Precomputed reference summary for one feature."""

    kind: str  # "numeric" or "categorical"
    values: np.ndarray
    edges: np.ndarray | None = None
    vocab: np.ndarray | None = None
    counts: np.ndarray | None = None
    scale: float = 1.0

    @classmethod
    def numeric(cls, ref: np.ndarray, n_bins: int = N_BINS) -> Reference:
        ref = np.sort(ref[~np.isnan(ref)])
        edges = quantile_edges(ref, n_bins)
        counts = np.bincount(bin_index(ref, edges), minlength=edges.size + 1)
        sd = float(ref.std())
        return cls("numeric", ref, edges=edges, counts=counts, scale=sd if sd > 0 else 1.0)

    @classmethod
    def categorical(cls, ref_codes: np.ndarray) -> Reference:
        vocab = np.unique(ref_codes)
        counts = np.append(np.bincount(np.searchsorted(vocab, ref_codes), minlength=vocab.size), 0)
        return cls("categorical", ref_codes, vocab=vocab, counts=counts)

    def bin(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "numeric":
            return bin_index(x, self.edges)
        pos = np.searchsorted(self.vocab, x).clip(0, self.vocab.size - 1)
        return np.where(self.vocab[pos] == x, pos, self.vocab.size)  # unseen category -> OOV bin


def score_windows(ref: Reference, windows: np.ndarray, with_ks: bool = True) -> dict[str, np.ndarray]:
    """Scores for a (R, n) matrix of complete windows; larger score means more drift."""
    windows = np.atleast_2d(windows)
    counts = row_counts(ref.bin(windows), ref.counts.size)
    out = binned_scores(ref.counts, counts)
    out["chi2"] = -np.log10(np.maximum(out["chi2_p"], 1e-300))
    if ref.kind == "numeric" and with_ks:
        kw = np.array([ks_w1(ref.values, w, ref.scale) for w in windows])
        out["ks"], out["w1"] = kw[:, 0], kw[:, 1]
        out["ks_p"] = ks_pvalue(out["ks"], windows.shape[1], ref.values.size)
    return out


@dataclass
class DetectorResult:
    """Outcome of one monitoring call. status: ok | insufficient_data | error."""

    status: str
    scores: dict[str, float]
    n_used: int
    missing_rate: float
    message: str = ""


def monitor_feature(ref: Reference, window: np.ndarray) -> DetectorResult:
    """Single-window call used by the monitoring cycle; never raises on bad input.

    Missing values are dropped and their rate reported; windows with fewer than N_MIN usable
    values abstain (status 'insufficient_data'), which the alerting layer treats as unknown.
    """
    try:
        if ref.kind == "numeric":
            window = np.asarray(window, dtype=float)
            ok = np.isfinite(window)
        else:
            window = np.asarray(window)
            ok = window >= 0  # categorical missing values are coded as their own category upstream
        miss = 1.0 - float(ok.mean()) if window.size else 1.0
        used = window[ok]
        if used.size < N_MIN:
            return DetectorResult("insufficient_data", {}, int(used.size), miss)
        s = score_windows(ref, used[None, :])
        return DetectorResult("ok", {k: float(v[0]) for k, v in s.items()}, int(used.size), miss)
    except (ValueError, FloatingPointError) as exc:  # malformed input is reported, not raised
        return DetectorResult("error", {}, 0, 1.0, message=str(exc))
