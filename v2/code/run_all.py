"""Single entry point: fetch/cache data, run every experiment with fixed seeds, write results/, figures, tables.

Usage: python code/run_all.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from obsbench import bench_detectors, config as C, data, electricity, figures, portfolio  # noqa: E402
from obsbench.summary import rounded  # noqa: E402
from obsbench.verdict import verdict  # noqa: E402


def _bench(seed):
    return bench_detectors.run_benchmark(seed), bench_detectors.run_streaming(seed)


def _portfolio_task(args):
    name, n_models, window, rho, missing, seed = args
    sim = portfolio.simulate(n_models, seed, window=window, rho=rho, missing=missing)
    rows = []
    for mp in (("exclude", "p1") if missing > 0 else ("exclude",)):
        for pol in portfolio.POLICIES:
            rows.append({"scenario": name, "n_models": n_models, "window": window, "rho": rho, "missing": missing,
                         "missing_policy": mp, "policy": pol, "seed": seed, **portfolio.evaluate(sim, pol, mp)})
    print(f"  portfolio {name} N={n_models} n={window} seed={seed} done", flush=True)
    return rows


def _dump(name, obj):
    (C.RESULTS / name).write_text(json.dumps(rounded(obj), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data.write_sources()
    print(f"[{time.time() - t0:.0f}s] data cached, SOURCES.md written", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        bench_out = list(ex.map(_bench, C.SEEDS))
        print(f"[{time.time() - t0:.0f}s] detector benchmark + streaming done", flush=True)
        tasks = [(*sc, seed) for sc in portfolio.SCENARIOS for seed in C.SEEDS]
        tasks.sort(key=lambda t: -t[1] * t[2])  # largest jobs first
        port_rows = [r for rows in ex.map(_portfolio_task, tasks) for r in rows]
        print(f"[{time.time() - t0:.0f}s] portfolio simulation done", flush=True)

    bench = pd.DataFrame([r for rows, _ in bench_out for r in rows])
    stream = pd.DataFrame([r for _, rows in bench_out for r in rows])
    bench.to_csv(C.RESULTS / "detector_benchmark_long.csv", index=False, float_format="%.4f")
    stream.to_csv(C.RESULTS / "streaming_delay_long.csv", index=False, float_format="%.4f")
    bench_res = bench_detectors.aggregate(bench, stream)
    _dump("detector_benchmark.json", bench_res)

    port_rows.sort(key=lambda r: (r["scenario"], r["n_models"], r["missing_policy"], r["policy"], r["seed"]))
    pd.DataFrame(port_rows).to_csv(C.RESULTS / "portfolio_per_seed.csv", index=False, float_format="%.4f")
    port_res = portfolio.aggregate(port_rows)
    _dump("portfolio.json", port_res)

    elec = electricity.run()
    _dump("electricity.json", elec)
    print(f"[{time.time() - t0:.0f}s] electricity agreement study done", flush=True)

    timing = bench_detectors.run_timing()
    _dump("timing.json", {**timing, "machine": platform.machine(), "processor": platform.processor(),
                          "python": platform.python_version(), "numpy": np.__version__})

    v = verdict(json.loads((C.RESULTS / "portfolio.json").read_text()), bench_res, elec)
    _dump("verdict.json", v)
    figures.make_all()
    elapsed = time.time() - t0
    _dump("run_info.json", {"elapsed_s": elapsed, "workers": a.workers, "seeds": list(C.SEEDS)})
    print(f"[{elapsed:.0f}s] verdict: {v['verdict']} (H2 {v['H2']['verdict']}, H3 {v['H3']['verdict']})")


if __name__ == "__main__":
    main()
