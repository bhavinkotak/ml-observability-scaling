import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from obsbench import config as C  # noqa: E402
from obsbench.detectors import Reference, binned_scores, ks_pvalue, ks_w1, monitor_feature, score_windows  # noqa: E402
from obsbench.drift import frequency_shift_probs, lattice_step, sample_windows, to_lattice  # noqa: E402
from obsbench.electricity import kappa, perf_test  # noqa: E402
from obsbench.portfolio import bh_reject, decide, evaluate, ks_batch, simulate  # noqa: E402
from obsbench.verdict import _rule  # noqa: E402


@pytest.mark.parametrize("discrete", [False, True])
def test_ks_and_w1_match_scipy(discrete):
    rng = np.random.default_rng(0)
    ref = rng.normal(size=700)
    win = rng.normal(0.2, 1.3, size=150)
    if discrete:
        ref, win = np.round(ref * 2), np.round(win * 2)
    d, w1 = ks_w1(np.sort(ref), win, scale=2.0)
    res = stats.ks_2samp(ref, win, method="asymp")
    assert d == pytest.approx(res.statistic, abs=1e-12)
    assert ks_pvalue(np.array(d), win.size, ref.size) == pytest.approx(res.pvalue, rel=1e-9)
    assert w1 == pytest.approx(stats.wasserstein_distance(ref, win) / 2.0, rel=1e-9)


def test_ks_pvalue_table_matches_kstwo():
    d = np.arange(1, 1001) / 1000
    for n in (100, 200, 500):
        assert np.array_equal(ks_pvalue(d, n, 1000), stats.kstwo.sf(d, round(n * 1000 / (n + 1000))))


def test_ks_batch_matches_scipy_on_continuous_data():
    rng = np.random.default_rng(1)
    k, m, n = 6, 300, 80
    ref = np.sort(rng.normal(size=(k, m)), axis=1)
    flat = (ref + (np.arange(k) * 64.0)[:, None]).ravel()
    win = rng.normal(0.3, 1.0, size=(k, n))
    d = ks_batch(flat, m, win)
    for i in range(k):
        assert d[i] == pytest.approx(stats.ks_2samp(ref[i], win[i]).statistic, abs=1e-12)


def test_binned_scores_zero_for_identical_and_chi2_matches_scipy():
    counts = np.array([10, 20, 30, 40])
    s = binned_scores(counts, counts[None, :])
    assert s["psi"][0] == pytest.approx(0.0, abs=1e-12)
    assert s["kl"][0] == pytest.approx(0.0, abs=1e-12)
    win = np.array([[15, 25, 20, 40]])
    s = binned_scores(counts, win)
    chi = stats.chi2_contingency(np.vstack([counts, win[0]]), correction=False)
    assert s["chi2_stat"][0] == pytest.approx(chi.statistic)
    assert s["chi2_p"][0] == pytest.approx(chi.pvalue)
    assert s["psi"][0] > 0


def test_monitor_feature_outcome_handling():
    rng = np.random.default_rng(2)
    ref = Reference.numeric(rng.normal(size=500))
    short = monitor_feature(ref, rng.normal(size=C.N_MIN - 1))
    assert short.status == "insufficient_data" and short.scores == {}
    with_nan = np.r_[rng.normal(size=100), [np.nan] * 25]
    res = monitor_feature(ref, with_nan)
    assert res.status == "ok" and res.n_used == 100 and res.missing_rate == pytest.approx(0.2)
    cat = Reference.categorical(np.array([0, 1, 1, 2] * 50))
    res = monitor_feature(cat, np.array([0, 1, 9, 9] * 20))  # 9 is unseen -> out-of-vocabulary bin
    assert res.status == "ok" and res.scores["chi2_p"] < 1e-6


def test_drift_injection_keeps_lattice_and_probabilities():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 50, size=2000).astype(float)
    assert lattice_step(x) == 1.0
    y = to_lattice(x + 0.3, 1.0, 0.0, rng)
    assert np.all(np.mod(y, 1) == 0)
    assert y.mean() - x.mean() == pytest.approx(0.3, abs=0.05)
    cats, p = frequency_shift_probs(np.array([0] * 90 + [1] * 10), 0.4)
    assert p.sum() == pytest.approx(1.0) and p[1] == pytest.approx(0.6 * 0.1 + 0.2)
    null = sample_windows(x, "none", 0.0, 50, 4, x.std(), np.median(x), 1.0, rng)
    assert null.shape == (4, 50) and np.isin(null, x).all()
    grid = np.round(np.arange(48) / 47, 6)  # rounded regular grid, as Elec2 'period'
    step = lattice_step(grid)
    assert step is not None
    z = to_lattice(rng.choice(grid[:40], 500) + 0.05, step, 0.0, rng, grid)
    assert np.isin(z, grid).all()


def test_bh_matches_scipy_adjusted_pvalues():
    rng = np.random.default_rng(4)
    p = np.r_[rng.uniform(size=200), rng.uniform(0, 1e-4, size=20)]
    ours = bh_reject(p, 0.05)
    ref = stats.false_discovery_control(p, method="bh") <= 0.05
    assert np.array_equal(ours, ref)


def test_decide_ignores_unknown_and_hierarchical_flags_argmin():
    p = np.array([[0.2, 1e-9, np.nan], [np.nan, np.nan, np.nan], [0.5, 0.6, 0.7]])
    for pol in ("per_test", "bonferroni", "bh", "hierarchical"):
        rej = decide(p, pol)
        assert not rej[np.isnan(p)].any()
        assert rej[0, 1]
    assert decide(p, "hierarchical").sum() == 1


def test_simulation_null_calibration_small():
    sim = simulate(10, seed=0, days=6)
    sim["drifted"][:] = False
    out = evaluate(sim, "per_test")
    assert 0 < out["fa_feature_per_day"] < 0.15 * 200


def test_perf_test_unknown_and_kappa_and_rule():
    assert perf_test(0.3, C.MIN_LABELS - 1, 0.2, 500) is None
    assert perf_test(0.4, 300, 0.2, 600) < 0.05
    assert np.isnan(kappa(np.ones(5, bool), np.ones(5, bool)))
    assert kappa(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) == pytest.approx(1.0)
    assert _rule(0.95, 0.99, 0.9) == "SUPPORTED"
    assert _rule(0.85, 0.95, 0.9) == "INCONCLUSIVE"
    assert _rule(0.5, 0.6, 0.9) == "REFUTED"
    assert _rule(0.1, 0.3, 0.4, higher_is_support=False) == "SUPPORTED"


def test_score_windows_shapes():
    rng = np.random.default_rng(5)
    ref = Reference.numeric(rng.normal(size=400))
    s = score_windows(ref, rng.normal(size=(3, 60)))
    assert all(s[k].shape == (3,) for k in ("psi", "kl", "chi2", "ks", "w1", "ks_p"))
