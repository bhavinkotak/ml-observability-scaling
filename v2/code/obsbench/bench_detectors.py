"""Detector benchmark on real feature distributions with injected (simulated) drift.

Produces calibrated TPR/FPR, false alarm rates of rule-of-thumb thresholds, a PSI bin ablation,
a streaming detection-delay study, and per-call compute time.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .data import DATASETS, feature_arrays
from .detectors import CATEGORICAL_DETECTORS, NUMERIC_DETECTORS, Reference, binned_scores, bin_index, \
    ks_w1, row_counts, score_windows
from .drift import lattice_step, sample_windows, to_lattice
from .summary import summarize

DEFAULT_RULES = {  # rule name -> (score key, comparator threshold, direction)
    "psi>0.1": ("psi", C.PSI_RULES[0], "gt"),
    "psi>0.25": ("psi", C.PSI_RULES[1], "gt"),
    "ks_p<0.05": ("ks_p", C.ALPHA, "lt"),
    "chi2_p<0.05": ("chi2_p", C.ALPHA, "lt"),
}


def _fire(scores: dict, rule: str) -> np.ndarray | None:
    key, thr, direction = DEFAULT_RULES[rule]
    if key not in scores:
        return None
    return scores[key] > thr if direction == "gt" else scores[key] < thr


def _features():
    for di, name in enumerate(DATASETS):
        num, cat = feature_arrays(name)
        for fi, (col, x) in enumerate(num.items()):
            yield di, fi, name, col, "numeric", x
        for fi, (col, x) in enumerate(cat.items()):
            yield di, 100 + fi, name, col, "categorical", x


def _split(x: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(x.size)
    half = x.size // 2
    ref_pool, test_pool = x[perm[:half]], x[perm[half:]]
    ref = ref_pool[rng.choice(ref_pool.size, size=C.REF_SIZE, replace=False)]
    return ref, test_pool


def _psi_only(ref_vals: np.ndarray, windows: np.ndarray, n_bins: int) -> np.ndarray:
    r = Reference.numeric(ref_vals, n_bins)
    return binned_scores(r.counts, row_counts(r.bin(windows), r.counts.size))["psi"]


def run_benchmark(seed: int) -> list[dict]:
    rows: list[dict] = []
    for di, fi, dname, col, kind, x in _features():
        rng = np.random.default_rng([seed, di, fi])
        ref_vals, pool = _split(x, rng)
        ref = Reference.numeric(ref_vals) if kind == "numeric" else Reference.categorical(ref_vals)
        dets = NUMERIC_DETECTORS if kind == "numeric" else CATEGORICAL_DETECTORS
        drifts = C.NUMERIC_DRIFTS if kind == "numeric" else C.CATEGORICAL_DRIFTS
        step = lattice_step(x) if kind == "numeric" else None
        sd, med = ref.scale, float(np.median(ref_vals))
        base = {"seed": seed, "dataset": dname, "feature": col, "kind": kind,
                "discrete": kind == "categorical" or step is not None}

        def draw(drift, mag, n, r):
            return sample_windows(pool, drift, mag, n, r, sd, med, step, rng)

        for n in C.WINDOW_SIZES:
            cal, ev = draw("none", 0.0, n, C.N_NULL_CAL), draw("none", 0.0, n, C.N_NULL_EVAL)
            s_cal, s_ev = score_windows(ref, cal), score_windows(ref, ev)
            tau = {d: np.quantile(s_cal[d], 1 - C.FPR_TARGET, method="higher") for d in dets}
            for d in dets:
                rows.append({**base, "n": n, "drift": "none", "magnitude": 0.0, "detector": d,
                             "metric": "fpr_calibrated", "value": float(np.mean(s_ev[d] > tau[d]))})
            for rule in DEFAULT_RULES:
                fired = _fire(s_ev, rule)
                if fired is not None:
                    rows.append({**base, "n": n, "drift": "none", "magnitude": 0.0, "detector": rule,
                                 "metric": "fpr_default", "value": float(fired.mean())})
            abl_tau, abl_null = {}, {}
            if kind == "numeric":
                for b in C.PSI_BIN_ABLATION:
                    abl_tau[b] = np.quantile(_psi_only(ref_vals, cal, b), 1 - C.FPR_TARGET, method="higher")
                    abl_null[b] = _psi_only(ref_vals, ev, b)
                    rows.append({**base, "n": n, "drift": "none", "magnitude": 0.0, "detector": f"psi_B{b}",
                                 "metric": "fpr_default", "value": float(np.mean(abl_null[b] > C.PSI_RULES[0]))})
            for drift in drifts:
                for mag in C.MAGNITUDES:
                    w = draw(drift, mag, n, C.N_DRIFT)
                    s = score_windows(ref, w)
                    for d in dets:
                        rows.append({**base, "n": n, "drift": drift, "magnitude": mag, "detector": d,
                                     "metric": "tpr_calibrated", "value": float(np.mean(s[d] > tau[d]))})
                    for rule in DEFAULT_RULES:
                        fired = _fire(s, rule)
                        if fired is not None:
                            rows.append({**base, "n": n, "drift": drift, "magnitude": mag, "detector": rule,
                                         "metric": "tpr_default", "value": float(fired.mean())})
                    for b in abl_tau:
                        rows.append({**base, "n": n, "drift": drift, "magnitude": mag, "detector": f"psi_B{b}",
                                     "metric": "tpr_calibrated",
                                     "value": float(np.mean(_psi_only(ref_vals, w, b) > abl_tau[b]))})
    return rows


def run_streaming(seed: int) -> list[dict]:
    """Gradual mean-shift ramp after STREAM_PRE null batches; per-batch thresholds calibrated at STREAM_FPR."""
    rows: list[dict] = []
    t_total = C.STREAM_PRE + C.STREAM_POST
    deltas = np.r_[np.zeros(C.STREAM_PRE), C.STREAM_DELTA_MAX * np.arange(1, C.STREAM_POST + 1) / C.STREAM_POST]
    for di, fi, dname, col, kind, x in _features():
        if kind != "numeric":
            continue
        rng = np.random.default_rng([seed, di, fi, 7])
        ref_vals, pool = _split(x, rng)
        ref = Reference.numeric(ref_vals)
        step, origin = lattice_step(x), float(pool.min())
        cal = pool[rng.integers(0, pool.size, size=(C.STREAM_NULL_CAL, C.STREAM_N))]
        s_cal = score_windows(ref, cal)
        tau = {d: np.quantile(s_cal[d], 1 - C.STREAM_FPR, method="higher") for d in NUMERIC_DETECTORS}
        xs = pool[rng.integers(0, pool.size, size=(C.STREAM_REPS, t_total, C.STREAM_N))]
        xs = to_lattice(xs + deltas[None, :, None] * ref.scale, step, origin, rng,
                        np.unique(pool) if step is not None and step != 1.0 else None)
        xs[:, :C.STREAM_PRE] = pool[rng.integers(0, pool.size, size=(C.STREAM_REPS, C.STREAM_PRE, C.STREAM_N))]
        s = score_windows(ref, xs.reshape(-1, C.STREAM_N))
        for d in NUMERIC_DETECTORS:
            alarm = (s[d] > tau[d]).reshape(C.STREAM_REPS, t_total)
            post = alarm[:, C.STREAM_PRE:]
            detected = post.any(axis=1)
            delay = np.where(detected, post.argmax(axis=1) + 1, np.nan)
            rows.append({"seed": seed, "dataset": dname, "feature": col, "detector": d,
                         "pre_change_false_alarms": float(alarm[:, :C.STREAM_PRE].sum(axis=1).mean()),
                         "detection_rate": float(detected.mean()),
                         "mean_delay_batches": float(np.nanmean(delay)) if detected.any() else np.nan,
                         "median_delay_batches": float(np.nanmedian(delay)) if detected.any() else np.nan})
    return rows


def run_timing() -> dict:
    """Median wall time per call (microseconds), reference summary precomputed, single thread."""
    rng = np.random.default_rng(12345)
    out = {}
    for size in C.TIMING_SIZES:
        ref_vals, win = rng.normal(size=size), rng.normal(0.1, 1.0, size=size)
        ref = Reference.numeric(ref_vals)
        calls = {
            "binned_psi_kl_chi2": lambda: binned_scores(ref.counts, np.bincount(bin_index(win, ref.edges),
                                                                                minlength=ref.counts.size)),
            "ks_scipy": lambda: stats.ks_2samp(ref_vals, win, method="asymp"),
            "w1_scipy": lambda: stats.wasserstein_distance(ref_vals, win),
            "ks_and_w1_ours": lambda: ks_w1(ref.values, win, ref.scale),
        }
        for name, fn in calls.items():
            fn()
            ts = []
            for _ in range(C.TIMING_REPS):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            out[f"{name}_n{size}_us"] = float(np.median(ts) * 1e6)
    return out


def aggregate(bench: pd.DataFrame, stream: pd.DataFrame) -> dict:
    res: dict = {"calibrated": {}, "default_null_fpr": {}, "default_tpr": {}, "psi_bins": {}, "streaming": {},
                 "calibrated_fpr_check": {}}
    feat_mean = bench.groupby(["kind", "n", "drift", "magnitude", "detector", "metric", "seed"])["value"].mean()
    fm = feat_mean.reset_index()
    for (kind, n, drift, mag, det, metric), g in fm.groupby(["kind", "n", "drift", "magnitude", "detector",
                                                              "metric"]):
        key = f"{kind}|n={n}|{drift}|{mag}|{det}"
        bucket = {"tpr_calibrated": "calibrated", "fpr_calibrated": "calibrated_fpr_check",
                  "fpr_default": "default_null_fpr", "tpr_default": "default_tpr"}[metric]
        if det.startswith("psi_B"):
            bucket = "psi_bins"
            key = f"{metric}|{key}"
        res[bucket][key] = summarize(g["value"])
    # Mean TPR over magnitudes per drift type, n, detector (headline table)
    tpr = fm[(fm.metric == "tpr_calibrated") & ~fm.detector.str.startswith("psi_B")]
    res["calibrated_by_type"] = {
        f"{kind}|n={n}|{drift}|{det}": summarize(g.groupby("seed")["value"].mean())
        for (kind, n, drift, det), g in tpr.groupby(["kind", "n", "drift", "detector"])}
    # Same, split by discrete (integer or regular-grid) versus continuous numeric features
    raw = bench[(bench.metric == "tpr_calibrated") & (bench.kind == "numeric") & ~bench.detector.str.startswith("psi_B")]
    by_disc = raw.groupby(["discrete", "n", "drift", "detector", "seed"])["value"].mean().reset_index()
    res["calibrated_by_type_discrete"] = {
        f"{'discrete' if disc else 'continuous'}|n={n}|{drift}|{det}": summarize(g["value"])
        for (disc, n, drift, det), g in by_disc.groupby(["discrete", "n", "drift", "detector"])}
    res["n_features"] = {
        "numeric_discrete": int(bench[(bench.kind == "numeric") & bench.discrete].feature.nunique()),
        "numeric_continuous": int(bench[(bench.kind == "numeric") & ~bench.discrete].feature.nunique()),
        "categorical": int(bench[bench.kind == "categorical"].feature.nunique()),
        "total": int(bench.feature.nunique())}
    # H2: PSI>0.1 null false alarm rate, n=100 minus n=2000, numeric features
    nul = fm[(fm.metric == "fpr_default") & (fm.detector == "psi>0.1") & (fm.kind == "numeric")]
    small = nul[nul.n == min(C.WINDOW_SIZES)].set_index("seed")["value"]
    large = nul[nul.n == max(C.WINDOW_SIZES)].set_index("seed")["value"]
    res["H2_psi_fpr_gap"] = summarize((small - large).values)
    res["H2_psi_fpr_small_n"] = summarize(small.values)
    res["H2_psi_fpr_large_n"] = summarize(large.values)
    # Null false alarm counts (numerator/denominator) for the default rules, pooled over seeds and features
    nulls = bench[bench.metric == "fpr_default"]
    per_cfg = C.N_NULL_EVAL
    res["default_null_counts"] = {
        f"{kind}|n={n}|{det}": {"false_alarms": int(round((g["value"] * per_cfg).sum())),
                                "windows": int(len(g) * per_cfg)}
        for (kind, n, det), g in nulls.groupby(["kind", "n", "detector"])}
    # Streaming
    sm = stream.groupby(["detector", "seed"])[["pre_change_false_alarms", "detection_rate", "mean_delay_batches",
                                                "median_delay_batches"]].mean().reset_index()
    for det, g in sm.groupby("detector"):
        res["streaming"][det] = {c: summarize(g[c]) for c in ["pre_change_false_alarms", "detection_rate",
                                                              "mean_delay_batches", "median_delay_batches"]}
        res["streaming"][det]["delta_at_detection"] = summarize(
            C.STREAM_DELTA_MAX * g["mean_delay_batches"] / C.STREAM_POST)
    return res
