"""Electricity (Elec2) with its real temporal order: do input-drift alarms agree with performance alarms?"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

from . import config as C
from .data import load
from .detectors import Reference, ks_pvalue, ks_w1, score_windows
from .portfolio import bh_reject
from .summary import summarize

DRIFT_FEATURES = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
MODEL_FEATURES = DRIFT_FEATURES + ["period", "day"]
ACC_DROP = 0.05
PSI_ALARM = 0.25


def load_stream():
    df = load("electricity")
    df = df.assign(day=df["day"].astype(int)).iloc[np.lexsort((df["period"].to_numpy(), df["date"].to_numpy()))]
    x = df[MODEL_FEATURES].to_numpy(dtype=float)
    y = (df["class"].astype(str) == "UP").to_numpy(dtype=int)
    return x, y


def perf_test(err_b: float, n_b: int, err_v: float, n_v: int) -> float | None:
    """One-sided two-proportion z-test p-value for batch error > validation error; None if labels are missing."""
    if n_b < C.MIN_LABELS:
        return None
    pooled = (err_b * n_b + err_v * n_v) / (n_b + n_v)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_b + 1 / n_v))
    return float(stats.norm.sf((err_b - err_v) / se)) if se > 0 else 1.0


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == a[0]) and np.all(b == b[0]):
        return float("nan")  # undefined when both raters are constant
    return float(cohen_kappa_score(a, b))


def run_one(run: int, start_week: int, x: np.ndarray, y: np.ndarray) -> dict:
    w = C.WEEK
    tr = slice(start_week * w, (start_week + C.TRAIN_WEEKS - C.VAL_WEEKS) * w)
    va = slice(tr.stop, (start_week + C.TRAIN_WEEKS) * w)
    model = RandomForestClassifier(n_estimators=C.RF_TREES, random_state=run, n_jobs=1).fit(x[tr], y[tr])
    err_v = float(np.mean(model.predict(x[va]) != y[va]))
    n_v = va.stop - va.start
    ref_rows = slice(tr.start, va.stop)
    refs = {f: Reference.numeric(x[ref_rows, MODEL_FEATURES.index(f)]) for f in DRIFT_FEATURES}
    weeks = []
    for k in range(C.MONITOR_WEEKS):
        b = slice(va.stop + k * w, va.stop + (k + 1) * w)
        if b.stop > len(y):
            break
        err_b = float(np.mean(model.predict(x[b]) != y[b]))
        p_perf = perf_test(err_b, b.stop - b.start, err_v, n_v)
        ks_p, ks_d, psi = [], [], []
        for f in DRIFT_FEATURES:
            col = x[b, MODEL_FEATURES.index(f)]
            d, _ = ks_w1(refs[f].values, col, refs[f].scale)
            ks_d.append(d)
            ks_p.append(float(ks_pvalue(np.array(d), col.size, refs[f].values.size)))
            psi.append(float(score_windows(refs[f], col[None, :], with_ks=False)["psi"][0]))
        weeks.append({"week": k, "acc": 1 - err_b, "acc_drop": err_b - err_v, "p_perf": p_perf,
                      "perf_alarm": p_perf is not None and p_perf < C.ALPHA,
                      "perf_drop_alarm": (err_b - err_v) >= ACC_DROP,
                      "input_alarm_bh": bool(bh_reject(np.array(ks_p), C.Q_FDR).any()),
                      "input_alarm_psi": bool(max(psi) > PSI_ALARM),
                      "max_ks": max(ks_d), "max_psi": max(psi)})
    perf = np.array([wk["perf_alarm"] for wk in weeks])
    drop = np.array([wk["perf_drop_alarm"] for wk in weeks])
    inp = np.array([wk["input_alarm_bh"] for wk in weeks])
    inp_psi = np.array([wk["input_alarm_psi"] for wk in weeks])
    acc_drop = np.array([wk["acc_drop"] for wk in weeks])
    return {
        "run": run, "start_week": start_week, "val_acc": 1 - err_v, "n_weeks": len(weeks),
        "perf_alarm_rate": float(perf.mean()), "perf_drop_alarm_rate": float(drop.mean()),
        "input_alarm_rate": float(inp.mean()), "input_psi_alarm_rate": float(inp_psi.mean()),
        "both": int((perf & inp).sum()), "input_only": int((~perf & inp).sum()),
        "perf_only": int((perf & ~inp).sum()), "neither": int((~perf & ~inp).sum()),
        "kappa_bh_vs_perf": kappa(inp, perf), "kappa_psi_vs_perf": kappa(inp_psi, perf),
        "kappa_psi_vs_drop": kappa(inp_psi, drop),
        "spearman_maxpsi_accdrop": float(stats.spearmanr([wk["max_psi"] for wk in weeks], acc_drop)[0]),
        "spearman_maxks_accdrop": float(stats.spearmanr([wk["max_ks"] for wk in weeks], acc_drop)[0]),
        "unknown_perf_weeks": int(sum(wk["p_perf"] is None for wk in weeks)),
        "weeks": weeks,
    }


def run() -> dict:
    x, y = load_stream()
    vic = x[:, MODEL_FEATURES.index("vicprice")]
    fill_end = int(np.nonzero(vic != vic[0])[0][0])  # vic* and transfer hold a constant fill value before this row
    runs = [run_one(i, s, x, y) for i, s in enumerate(C.TRAIN_START_WEEKS)]
    keys = ["val_acc", "perf_alarm_rate", "perf_drop_alarm_rate", "input_alarm_rate", "input_psi_alarm_rate",
            "kappa_bh_vs_perf", "kappa_psi_vs_perf", "kappa_psi_vs_drop", "spearman_maxpsi_accdrop",
            "spearman_maxks_accdrop"]
    totals = {k: int(sum(r[k] for r in runs)) for k in ["both", "input_only", "perf_only", "neither", "n_weeks",
                                                        "unknown_perf_weeks"]}
    return {"summary": {k: summarize([r[k] for r in runs]) for k in keys}, "totals": totals,
            "constant_fill_end_row": fill_end, "constant_fill_end_week": fill_end / C.WEEK, "n_rows": int(y.size),
            "H3_kappa": summarize([r["kappa_bh_vs_perf"] for r in runs]), "runs": runs}
