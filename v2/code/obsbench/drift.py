"""Simulated drift injected into windows resampled from real feature distributions."""

from __future__ import annotations

import numpy as np
from scipy import stats


def lattice_step(x: np.ndarray, max_levels: int = 200) -> float | None:
    """Grid step if the feature is integer-valued or lies on a regular grid of few levels."""
    if np.all(np.equal(np.mod(x, 1), 0)):
        return 1.0
    u = np.unique(x)
    if u.size > max_levels or u.size < 2:
        return None
    d = np.diff(u)
    h = d.min()
    # grids stored with rounding (e.g. Elec2 'period' = k/47 to 6 decimals) differ from exact multiples by ~1e-4 h
    return float(h) if np.allclose(d / h, np.round(d / h), atol=1e-2) else None


def to_lattice(x: np.ndarray, step: float | None, origin: float, rng: np.random.Generator,
               grid: np.ndarray | None = None) -> np.ndarray:
    """Unbiased stochastic rounding back onto the feature's grid (keeps discrete features discrete).

    `grid` holds the stored level values of a non-integer grid; results within 1% of a step are snapped to them.
    """
    if step is None:
        return x
    k = (x - origin) / step
    lo = np.floor(k)
    out = origin + step * (lo + (rng.random(x.shape) < (k - lo)))
    if grid is not None and step != 1.0:
        idx = np.searchsorted(grid, out).clip(1, grid.size - 1)
        left, right = grid[idx - 1], grid[idx]
        near = np.where(np.abs(out - left) <= np.abs(right - out), left, right)
        out = np.where(np.abs(near - out) < 0.01 * step, near, out)
    return out


def mean_shift(x, delta, sd, step, origin, rng, grid=None):
    """Additive offset of delta reference standard deviations."""
    return to_lattice(x + delta * sd, step, origin, rng, grid)


def scale_change(x, delta, center, step, origin, rng, grid=None):
    """Spread around the reference median multiplied by (1 + delta)."""
    return to_lattice(center + (x - center) * (1.0 + delta), step, origin, rng, grid)


def selection_weights(pool: np.ndarray, beta: float) -> np.ndarray:
    """Covariate shift by selection: P(row) proportional to exp(beta * z), z = rank-based normal score."""
    z = stats.norm.ppf((stats.rankdata(pool) - 0.5) / pool.size)
    w = np.exp(beta * z)
    return w / w.sum()


def frequency_shift_probs(pool_codes: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Categorical frequencies mixed toward uniform: p' = (1 - delta) p + delta / K."""
    cats, cnt = np.unique(pool_codes, return_counts=True)
    p = cnt / cnt.sum()
    return cats, (1.0 - delta) * p + delta / cats.size


def sample_windows(pool: np.ndarray, kind: str, magnitude: float, n: int, r: int, ref_sd: float,
                   ref_median: float, step: float | None, rng: np.random.Generator) -> np.ndarray:
    """R windows of size n from the test pool with the given drift kind applied (kind='none' for null)."""
    origin = float(pool.min())
    if kind == "selection_shift":
        return pool[rng.choice(pool.size, size=(r, n), p=selection_weights(pool, magnitude))]
    if kind == "frequency_shift":
        cats, p = frequency_shift_probs(pool, magnitude)
        return cats[rng.choice(cats.size, size=(r, n), p=p)]
    x = pool[rng.integers(0, pool.size, size=(r, n))]
    if kind == "none":
        return x
    grid = np.unique(pool) if step is not None and step != 1.0 else None
    if kind == "mean_shift":
        return mean_shift(x, magnitude, ref_sd, step, origin, rng, grid)
    if kind == "scale_change":
        return scale_change(x, magnitude, ref_median, step, origin, rng, grid)
    raise ValueError(f"unknown drift kind {kind}")
