import numpy as np


def _resample_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    means = _resample_means(values, n_boot, seed)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(diff, n_boot, seed, alpha)
