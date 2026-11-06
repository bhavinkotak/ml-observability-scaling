"""Simulated model portfolio: daily KS tests on N models x F features with per-test, Bonferroni,
Benjamini-Hochberg, and hierarchical (per-model Bonferroni, then BH across models) alarm policies.

All drift here is simulated: Gaussian features, sudden mean shifts that persist after onset.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .detectors import ks_pvalue
from .summary import summarize

POLICIES = ("per_test", "bonferroni", "bh", "hierarchical")
_OFFSET = 64.0  # row separation for the flattened searchsorted; |x| stays far below this


def _gauss(rng: np.random.Generator, n_models: int, n_feat: int, size: int, rho: float) -> np.ndarray:
    x = rng.standard_normal((n_models, n_feat, size))
    if rho > 0:
        common = rng.standard_normal((n_models, 1, size))
        x = math.sqrt(rho) * common + math.sqrt(1 - rho) * x
    return x


def ks_batch(ref_flat: np.ndarray, m: int, windows: np.ndarray) -> np.ndarray:
    """Two-sample KS statistic for K continuous (tie-free) windows against K sorted references.

    ref_flat holds the K sorted references, row k shifted by k * _OFFSET, so one global searchsorted
    evaluates every reference ECDF at its own window points.
    """
    k, n = windows.shape
    w = np.sort(windows, axis=1)
    offs = (np.arange(k) * _OFFSET)[:, None]
    g = np.searchsorted(ref_flat, (w + offs).ravel(), side="right").reshape(k, n) - (np.arange(k) * m)[:, None]
    g = g / m
    i = np.arange(1, n + 1) / n
    return np.maximum((i - g).max(axis=1), (g - (i - 1.0 / n)).max(axis=1))


def simulate(n_models: int, seed: int, window: int = C.PORTFOLIO_WINDOW, rho: float = 0.0,
             missing: float = 0.0, n_feat: int = C.N_FEATURES, m: int = C.PORTFOLIO_REF,
             days: int = C.N_DAYS) -> dict:
    rng = np.random.default_rng([seed, n_models, window, int(rho * 100), int(missing * 100)])
    k = n_models * n_feat
    ref = np.sort(_gauss(rng, n_models, n_feat, m, rho).reshape(k, m), axis=1)
    ref_flat = (ref + (np.arange(k) * _OFFSET)[:, None]).ravel()
    del ref
    n_events = math.ceil(C.DRIFT_MODEL_FRACTION * n_models)
    ev_models = rng.choice(n_models, n_events, replace=False)
    ev_onset = rng.integers(C.ONSET_LO, C.ONSET_HI + 1, n_events)
    shift = np.zeros((n_models, n_feat))
    onset = np.full(n_models, np.iinfo(np.int64).max)
    for mdl, t0 in zip(ev_models, ev_onset, strict=True):
        nf = rng.integers(1, C.DRIFT_FEATURES_MAX + 1)
        feats = rng.choice(n_feat, nf, replace=False)
        shift[mdl, feats] = rng.uniform(C.DELTA_LO, C.DELTA_HI, nf) * rng.choice([-1.0, 1.0], nf)
        onset[mdl] = t0
    pvals = np.empty((days, n_models, n_feat))
    drifted = np.zeros((days, n_models, n_feat), dtype=bool)
    for t in range(days):
        active = (t >= onset)[:, None] & (shift != 0)
        w = _gauss(rng, n_models, n_feat, window, rho) + (shift * active)[:, :, None]
        d = ks_batch(ref_flat, m, w.reshape(k, window))
        pvals[t] = ks_pvalue(d, window, m).reshape(n_models, n_feat)
        drifted[t] = active
    if missing > 0:
        pvals[rng.random(pvals.shape) < missing] = np.nan  # detector timeout / missing window -> unknown
    return {"p": pvals, "drifted": drifted, "ev_models": ev_models, "ev_onset": ev_onset, "shift": shift}


def bh_reject(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up on a 1-D array of p-values (no NaN)."""
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.sort(p)
    below = order <= q * np.arange(1, p.size + 1) / p.size
    if not below.any():
        return np.zeros(p.size, dtype=bool)
    return p <= order[np.nonzero(below)[0].max()]


def decide(p: np.ndarray, policy: str, alpha: float = C.ALPHA, q: float = C.Q_FDR) -> np.ndarray:
    """Rejected (model, feature) cells for one day. NaN p-values are unknown and never rejected.

    For 'hierarchical' only the feature with the smallest p-value of each rejected model is flagged.
    """
    valid = np.isfinite(p)
    rej = np.zeros(p.shape, dtype=bool)
    if policy == "per_test":
        rej[valid] = p[valid] <= alpha
    elif policy == "bonferroni":
        rej[valid] = p[valid] <= alpha / max(int(valid.sum()), 1)
    elif policy == "bh":
        rej[valid] = bh_reject(p[valid], q)
    elif policy == "hierarchical":
        n_valid = valid.sum(axis=1)
        has = n_valid > 0
        pmin = np.where(valid, p, np.inf).min(axis=1)
        p_model = np.where(has, np.minimum(1.0, n_valid * np.where(has, pmin, 1.0)), 1.0)
        model_rej = np.zeros(p.shape[0], dtype=bool)
        model_rej[has] = bh_reject(p_model[has], q)
        flagged = np.where(valid, p, np.inf).argmin(axis=1)
        rej[np.nonzero(model_rej)[0], flagged[model_rej]] = True
    else:
        raise ValueError(policy)
    return rej


def evaluate(sim: dict, policy: str, missing_policy: str = "exclude") -> dict:
    p = sim["p"] if missing_policy == "exclude" else np.where(np.isnan(sim["p"]), 1.0, sim["p"])
    drifted = sim["drifted"]
    days = p.shape[0]
    fa_feat, fa_model, fdp, n_rej, unknown = [], [], [], [], []
    first = {int(mdl): None for mdl in sim["ev_models"]}
    for t in range(days):
        rej = decide(p[t], policy)
        fa = rej & ~drifted[t]
        fa_feat.append(fa.sum())
        fa_model.append((rej.any(axis=1) & ~drifted[t].any(axis=1)).sum())
        n_rej.append(rej.sum())
        fdp.append(fa.sum() / max(rej.sum(), 1))
        unknown.append(np.isnan(p[t]).sum())
        hit = (rej & drifted[t]).any(axis=1)
        for mdl in first:
            if first[mdl] is None and hit[mdl]:
                first[mdl] = t
    delays = np.array([np.nan if first[int(mdl)] is None else first[int(mdl)] - t0
                       for mdl, t0 in zip(sim["ev_models"], sim["ev_onset"], strict=True)])
    detected = np.isfinite(delays)
    return {
        "fa_feature_per_day": float(np.mean(fa_feat)),
        "fa_model_per_day": float(np.mean(fa_model)),
        "rejections_per_day": float(np.mean(n_rej)),
        "fdp_mean": float(np.mean(fdp)),
        "unknown_tests_per_day": float(np.mean(unknown)),
        "n_events": int(delays.size),
        "recall_3d": float(np.mean(detected & (delays < C.RECALL_HORIZON))),
        "recall_horizon": float(detected.mean()),
        "missed_3d": int(np.sum(~(detected & (delays < C.RECALL_HORIZON)))),
        "mean_delay_days": float(np.nanmean(delays)) if detected.any() else float("nan"),
        "tests_per_day": int(p[0].size),
    }


SCENARIOS = [("base", n, C.PORTFOLIO_WINDOW, 0.0, 0.0) for n in C.PORTFOLIO_SIZES] + [
    ("corr", max(C.PORTFOLIO_SIZES), C.PORTFOLIO_WINDOW, C.CORR_RHO, 0.0),
    ("missing", max(C.PORTFOLIO_SIZES), C.PORTFOLIO_WINDOW, 0.0, C.MISSING_RATE),
] + [(f"window{w}", max(C.PORTFOLIO_SIZES), w, 0.0, 0.0) for w in C.ABLATION_WINDOWS]


def aggregate(rows: list[dict]) -> dict:
    import pandas as pd

    df = pd.DataFrame(rows)
    metrics = ["fa_feature_per_day", "fa_model_per_day", "rejections_per_day", "fdp_mean", "recall_3d",
               "recall_horizon", "missed_3d", "mean_delay_days", "unknown_tests_per_day", "n_events",
               "tests_per_day"]
    out: dict = {}
    for (sc, n, mp, pol), g in df.groupby(["scenario", "n_models", "missing_policy", "policy"]):
        out[f"{sc}|N={n}|{mp}|{pol}"] = {m: summarize(g[m]) for m in metrics}
    # Primary endpoints (H1) per seed, base scenario at the largest N
    base = df[(df.scenario == "base") & (df.n_models == max(C.PORTFOLIO_SIZES))].set_index(["policy", "seed"])
    pt, bh = base.loc["per_test"], base.loc["bh"]
    r_fa = 1 - bh["fa_feature_per_day"] / pt["fa_feature_per_day"]
    rr = bh["recall_3d"] / pt["recall_3d"]
    out["H1"] = {"R_FA": summarize(r_fa), "RR": summarize(rr),
                 "R_FA_per_seed": [float(v) for v in r_fa], "RR_per_seed": [float(v) for v in rr]}
    return out
