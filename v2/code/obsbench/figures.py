"""Figures (matplotlib PDFs) and LaTeX tables, generated only from files in results/."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

COL_W = 3.45  # inches, IEEE column (8.86 cm) minus a small margin
WIDE_W = 7.0
DET_LABEL = {"psi": "PSI", "kl": "KL", "chi2": r"$\chi^2$", "ks": "KS", "w1": "W1"}
DET_TEX = {"psi": "PSI", "kl": "KL", "chi2": "$\\chi^2$", "ks": "KS", "w1": "W1"}
POL_LABEL = {"per_test": "per-test", "bonferroni": "Bonferroni", "bh": "BH", "hierarchical": "hierarchical"}
RULE_LABEL = {"psi>0.1": "PSI>0.1", "psi>0.25": "PSI>0.25", "ks_p<0.05": "KS $p$<0.05",
              "chi2_p<0.05": r"$\chi^2$ $p$<0.05"}
RULE_COLOR = {"psi>0.1": "tab:blue", "psi>0.25": "tab:orange", "ks_p<0.05": "tab:green", "chi2_p<0.05": "tab:red"}
DRIFT_LABEL = {"mean_shift": "mean shift", "scale_change": "scale change", "selection_shift": "selection shift",
               "frequency_shift": "frequency shift (categorical)"}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3})


def _load(name):
    return json.loads((C.RESULTS / name).read_text())


def fig_null_fpr(bench):
    fig, axes = plt.subplots(1, 3, figsize=(WIDE_W, 1.9))
    ns = list(C.WINDOW_SIZES)
    for ax, kind in zip(axes[:2], ("numeric", "categorical"), strict=True):
        for rule in ("psi>0.1", "psi>0.25", "ks_p<0.05", "chi2_p<0.05"):
            key = [f"{kind}|n={n}|none|0.0|{rule}" for n in ns]
            if key[0] not in bench["default_null_fpr"]:
                continue
            m = [bench["default_null_fpr"][k]["mean"] for k in key]
            s = [bench["default_null_fpr"][k]["std"] for k in key]
            ax.errorbar(ns, m, yerr=s, marker="o", capsize=2, color=RULE_COLOR[rule], label=RULE_LABEL[rule])
        ax.axhline(C.ALPHA, color="grey", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("window size n")
        ax.set_title(f"({'ab'[kind == 'categorical']}) {kind} features")
        ax.set_ylabel("no-drift false alarm rate")
        ax.legend(frameon=False)
    ax = axes[2]
    for b in C.PSI_BIN_ABLATION:
        key = [f"fpr_default|numeric|n={n}|none|0.0|psi_B{b}" for n in ns]
        ax.errorbar(ns, [bench["psi_bins"][k]["mean"] for k in key], yerr=[bench["psi_bins"][k]["std"] for k in key],
                    marker="o", capsize=2, label=f"B={b}")
    ax.axhline(C.ALPHA, color="grey", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("window size n")
    ax.set_title("(c) PSI>0.1 by bin count, numeric")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_null_fpr.pdf")
    plt.close(fig)


def fig_tpr(bench, n=500):
    drifts = list(C.NUMERIC_DRIFTS) + list(C.CATEGORICAL_DRIFTS)
    fig, axes = plt.subplots(1, 4, figsize=(WIDE_W, 2.1), sharey=True)
    for ax, drift in zip(axes, drifts, strict=True):
        kind = "categorical" if drift == "frequency_shift" else "numeric"
        for det in DET_LABEL:
            keys = [f"{kind}|n={n}|{drift}|{m}|{det}" for m in C.MAGNITUDES]
            if keys[0] not in bench["calibrated"]:
                continue
            ax.errorbar(C.MAGNITUDES, [bench["calibrated"][k]["mean"] for k in keys],
                        yerr=[bench["calibrated"][k]["std"] for k in keys], marker="o", capsize=2,
                        label=DET_LABEL[det])
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(C.MAGNITUDES, [str(m) for m in C.MAGNITUDES])
        ax.set_xlabel("magnitude")
        ax.set_title(DRIFT_LABEL[drift])
    axes[0].set_ylabel(f"TPR at FPR {C.FPR_TARGET} (n={n})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(C.FIGURES / "fig_tpr_calibrated.pdf")
    plt.close(fig)


def fig_portfolio(port):
    ns = list(C.PORTFOLIO_SIZES)
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.2))
    for pol in POL_LABEL:
        fa = [port[f"base|N={n}|exclude|{pol}"]["fa_feature_per_day"] for n in ns]
        rc = [port[f"base|N={n}|exclude|{pol}"]["recall_3d"] for n in ns]
        axes[0].errorbar(ns, [max(f["mean"], 1e-2) for f in fa], yerr=[f["std"] for f in fa], marker="o",
                         capsize=2, label=POL_LABEL[pol])
        axes[1].errorbar(ns, [r["mean"] for r in rc], yerr=[r["std"] for r in rc], marker="o", capsize=2,
                         label=POL_LABEL[pol])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("models N")
    axes[0].set_ylabel("false feature alarms per day")
    axes[0].set_title("(a) false alarms")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("models N")
    axes[1].set_ylabel("event recall within 3 days")
    axes[1].set_title("(b) detection")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0),
               handlelength=1.5, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    fig.savefig(C.FIGURES / "fig_portfolio.pdf")
    plt.close(fig)


def fig_electricity(elec):
    run = elec["runs"][0]
    wk = run["weeks"]
    t = np.arange(len(wk))
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True)
    acc = np.array([w["acc"] for w in wk])
    perf = np.array([w["perf_alarm"] for w in wk])
    axes[0].plot(t, acc, color="k", marker=".", label="weekly accuracy")
    axes[0].axhline(run["val_acc"], color="grey", ls=":", label="validation accuracy")
    axes[0].scatter(t[perf], acc[perf], color="tab:red", zorder=3, s=10, label="performance alarm")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.0,
                   columnspacing=0.6, fontsize=6)
    psi = np.array([w["max_psi"] for w in wk])
    inp = np.array([w["input_alarm_bh"] for w in wk])
    axes[1].plot(t, psi, color="k", marker=".", label="max PSI over features")
    axes[1].scatter(t[inp], psi[inp], color="tab:blue", zorder=3, s=10, label="input alarm (KS+BH)")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 3.0, 4.0, 6.0)))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_ylabel("max PSI")
    axes[1].set_xlabel(f"week after training (run 0, start week {run['start_week']})")
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0,
                   columnspacing=0.6, fontsize=6)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_electricity.pdf")
    plt.close(fig)


def _pm(s, d=2):
    if s is None or s.get("mean") is None:
        return "n/a"
    return f"{s['mean']:.{d}f}$\\pm${(s['std'] or 0.0):.{d}f}"


def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def tables(bench, port, elec, timing):
    C.TABLES.mkdir(parents=True, exist_ok=True)
    # Table: calibrated TPR by drift type, averaged over magnitudes
    dets = list(DET_LABEL)
    lines = ["\\begin{tabular}{llccccc}", "\\toprule",
             "Drift type & $n$ & " + " & ".join(DET_TEX[d] for d in dets) + " \\\\", "\\midrule"]
    for drift in list(C.NUMERIC_DRIFTS) + list(C.CATEGORICAL_DRIFTS):
        kind = "categorical" if drift == "frequency_shift" else "numeric"
        for n in C.WINDOW_SIZES:
            cells = []
            for d in dets:
                s = bench["calibrated_by_type"].get(f"{kind}|n={n}|{drift}|{d}")
                cells.append(_pm(s) if s else "n/a")
            lines.append(f"{DRIFT_LABEL[drift].replace(' (categorical)', ' (cat.)')} & {n} & " + " & ".join(cells)
                         + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.TABLES / "tab_tpr.tex").write_text("\n".join(lines))
    # Table: default-rule null false alarms, numerator/denominator
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Features & Rule & " +
             " & ".join(f"$n={n}$" for n in C.WINDOW_SIZES) + " \\\\", "\\midrule"]
    for kind in ("numeric", "categorical"):
        for rule in ("psi>0.1", "psi>0.25", "ks_p<0.05", "chi2_p<0.05"):
            cells = []
            for n in C.WINDOW_SIZES:
                c = bench["default_null_counts"].get(f"{kind}|n={n}|{rule}")
                if c is None:
                    break
                cells.append(f"{c['false_alarms']}/{c['windows']}")
            if cells:
                r = rule.replace("psi", "PSI").replace("ks_p", "KS $p$").replace("chi2_p", "$\\chi^2$ $p$")
                lines.append(f"{kind} & {r.replace('>', '$>$').replace('<', '$<$')} & " + " & ".join(cells)
                             + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.TABLES / "tab_null_counts.tex").write_text("\n".join(lines))
    # Table: portfolio base scenario
    lines = ["\\begin{tabular}{rlrrrrr}", "\\toprule",
             "$N$ & Policy & FA/day & Model FA/day & FDP & Recall@3d & Delay (d) \\\\", "\\midrule"]
    for n in C.PORTFOLIO_SIZES:
        for pol in POL_LABEL:
            r = port[f"base|N={n}|exclude|{pol}"]
            lines.append(f"{n} & {POL_LABEL[pol]} & {_pm(r['fa_feature_per_day'])} & {_pm(r['fa_model_per_day'])} & "
                         f"{_pm(r['fdp_mean'], 3)} & {_pm(r['recall_3d'])} & {_pm(r['mean_delay_days'])} \\\\")
        lines.append("\\midrule" if n != max(C.PORTFOLIO_SIZES) else "\\bottomrule")
    lines.append("\\end{tabular}")
    (C.TABLES / "tab_portfolio.tex").write_text("\n".join(lines))
    # Table: breakpoint scenarios at the largest N
    big = max(C.PORTFOLIO_SIZES)
    sc = [("base", "exclude", "independent, $n=200$"), ("corr", "exclude", f"$\\rho={C.CORR_RHO}$"),
          ("missing", "exclude", "10\\% missing, exclude"), ("missing", "p1", "10\\% missing, $p{=}1$"),
          ("window100", "exclude", "$n=100$"), ("window500", "exclude", "$n=500$")]
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Scenario & Policy & FA/day & FDP & Recall@3d \\\\",
             "\\midrule"]
    for name, mp, label in sc:
        for pol in ("per_test", "bh", "hierarchical"):
            r = port[f"{name}|N={big}|{mp}|{pol}"]
            lines.append(f"{label} & {POL_LABEL[pol]} & {_pm(r['fa_feature_per_day'], 1)} & {_pm(r['fdp_mean'], 3)}"
                         f" & {_pm(r['recall_3d'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.TABLES / "tab_scenarios.tex").write_text("\n".join(lines))
    # Table: electricity runs
    lines = ["\\begin{tabular}{rrrrrr}", "\\toprule",
             "Start wk & Val. acc. & Perf. alarms & Input alarms & $\\kappa$ & $\\rho_s$ \\\\", "\\midrule"]
    for r in elec["runs"]:
        nw = r["n_weeks"]
        pa = round(r["perf_alarm_rate"] * nw)
        ia = round(r["input_alarm_rate"] * nw)
        lines.append(f"{r['start_week']} & {r['val_acc']:.3f} & {pa}/{nw} & {ia}/{nw} & "
                     f"{_f(r['kappa_bh_vs_perf'])} & {_f(r['spearman_maxpsi_accdrop'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.TABLES / "tab_electricity.tex").write_text("\n".join(lines))
    # Table: streaming delay and timing
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Detector & Pre-change FA & Detected & Delay (batches) & $\\delta$ at alarm \\\\", "\\midrule"]
    for d in dets:
        s = bench["streaming"][d]
        lines.append(f"{DET_TEX[d]} & {_pm(s['pre_change_false_alarms'], 3)} & {_pm(s['detection_rate'], 3)} & "
                     f"{_pm(s['mean_delay_batches'])} & {_pm(s['delta_at_detection'], 3)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.TABLES / "tab_delay.tex").write_text("\n".join(lines))
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Call & $n=m=1000$ & $n=m=10000$ \\\\", "\\midrule"]
    names = {"binned_psi_kl_chi2": "binned PSI+KL+$\\chi^2$ (ours)", "ks_scipy": "KS (SciPy, asymptotic)",
             "w1_scipy": "W1 (SciPy)", "ks_and_w1_ours": "KS+W1 (ours, shared sort)"}
    for k, label in names.items():
        lines.append(f"{label} & {timing[f'{k}_n1000_us']:.0f} & {timing[f'{k}_n10000_us']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.TABLES / "tab_timing.tex").write_text("\n".join(lines))


def make_all():
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    bench, port, elec, timing = (_load("detector_benchmark.json"), _load("portfolio.json"),
                                 _load("electricity.json"), _load("timing.json"))
    fig_null_fpr(bench)
    fig_tpr(bench)
    fig_portfolio(port)
    fig_electricity(elec)
    tables(bench, port, elec, timing)
