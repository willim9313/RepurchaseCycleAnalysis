"""
Unimodality test module.
Conduct various unimodality tests (Dip, Silverman, Heuristics) on transformed data.
"""
from typing import Dict, Any, Optional, Literal, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats
import diptest
import logging

logger = logging.getLogger(__name__)

DEFAULT_UNIMODALITY_PARAMS = {
    "alpha": 0.05,
    "max_sample_for_test": 10000,
    "value_col": "interval_days_transformed",
    "dip_bootstrap_samples": 200,
    "methods_by_mode": {
        "small": ["dip", "silverman"],
        "medium": ["dip_subsampled", "silverman"],
        "large": ["kde_extrema", "smoothness_emd"],
    },
    # Silverman test parameters
    "silverman_grid_size": 512,
    "silverman_search_iters": 30,     # binary search steps for critical bandwidth
    "silverman_bootstrap_samples": 200,
    # Smoothness/EMD heuristic parameters
    "emd_bootstrap_samples": 200,
    "emd_sample_size": 4000,          # for approximating gaussian null distribution
    "check_flat_distribution": True,
    "flat_kurtosis_threshold": -1.0,  # uniform kurtosis ≈ -1.2
    "flat_cv_threshold": 0.6,         # coefficient of variation threshold
}


def _resolve_mode_auto(
    n: int,
    cfg: Dict[str, Any]
) -> Literal["small", "medium", "large"]:
    """
    setup mode according to data size and global thresholds

    Parameters
    ----------
    n : int
        data size
    cfg : Dict[str, Any]
        global configuration

    Returns
    -------
    mode : str
        "small" | "medium" | "large"
    """
    thresholds = cfg.get("data_size_thresholds", {"small": 1e4, "medium": 1e6})
    th_small = thresholds.get("small", 1e4)
    th_medium = thresholds.get("medium", 1e6)

    if n <= th_small:
        return "small"
    elif n <= th_medium:
        return "medium"
    else:
        return "large"


def _subsample(values: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """
    return sub-sampled values if size exceeds max_n, otherwise return original values.
    sub-sampling to max_n values.

    Parameters
    ----------
    values : np.ndarray
        original values
    max_n : int
        maximum allowed size
    seed : int
        random seed for reproducibility

    Returns
    -------
    np.ndarray
        sub-sampled values
    """
    n = len(values)
    if n <= max_n:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return values[idx]


def _kde_peak_count(values: np.ndarray, grid_size: int = 512) -> int:
    """
    Apply Gaussian KDE to estimate density in 1D, count number of local maxima.

    Parameters
    ----------
    values : np.ndarray
        1D array of values to analyze
    grid_size : int, default 512
        number of grid points for density estimation

    Returns
    -------
    peak: int
        number of peaks detected, strong signal of multi-modality if > 1
    """
    if values.ndim != 1:
        values = values.ravel()

    if len(values) < 8:
        return 1

    # Check for constant values (zero variance) - KDE cannot handle this
    if np.ptp(values) == 0 or np.std(values) == 0:
        return 1

    kde = stats.gaussian_kde(values)
    x_min, x_max = float(values.min()), float(values.max())
    if x_max == x_min:
        return 1

    grid = np.linspace(x_min, x_max, grid_size)
    density = kde(grid)

    # Find local maxima: density[i-1] < density[i] > density[i+1]
    peaks = 0
    for i in range(1, grid_size - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peaks += 1
    return peaks


def _dip_p_value(values: np.ndarray, seed: int, cfg: Dict[str, Any]) -> float:
    """
    using diptest package to compute dip test p-value.

    Parameters
    ----------
    values : np.ndarray
        1D array of values to test
    seed : int
        random seed for reproducibility
    cfg : Dict[str, Any]
        configuration dictionary (not used here, but for API consistency)

    Returns
    -------
    p_value : float
        p-value from dip test
    """
    values = np.asarray(values, dtype=float).ravel()
    if len(values) < 8:
        return 1.0

    out = diptest.diptest(values)
    pval = float(out[1])
    return float(max(0.0, min(1.0, pval)))


# Silverman's unimodality test (critical bandwidth + bootstrap)
def _gaussian_kde_manual_density(
    grid: np.ndarray,
    values: np.ndarray,
    bandwidth: float
) -> np.ndarray:
    """
    Manual Gaussian KDE density on a fixed grid:
    f(x) = mean(phi((x-xi)/h)) / h

    Parameters
    ----------
    grid : np.ndarray
        1D array of grid points to evaluate density
    values : np.ndarray
        1D array of data points
    bandwidth : float
        bandwidth h for KDE

    Returns
    -------
    dens : np.ndarray
        estimated density at each grid point
    """
    values = values.astype(float).ravel()
    h = float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be > 0")

    # (grid_size, n) matrix; OK for small/medium sizes
    z = (grid[:, None] - values[None, :]) / h
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    dens = phi.mean(axis=1) / h
    return dens


def _count_modes_from_density(density: np.ndarray) -> int:
    """
    Count local maxima in a 1D density array.
    """
    if density.size < 3:
        return 1
    peaks = 0
    for i in range(1, density.size - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peaks += 1
    return peaks if peaks > 0 else 1


def _critical_bandwidth_unimodal(
    values: np.ndarray,
    grid_size: int,
    search_iters: int
) -> float:
    """
    Find the critical bandwidth h_crit for uni-modality:
    minimal h such that KDE has <= 1 mode.
    Use binary search on h.

    Parameters
    ----------
    values : np.ndarray
        1D array of data points
    grid_size : int
        number of grid points for density estimation
    search_iters : int
        number of binary search iterations

    Returns
    -------
    h_crit : float
    """
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if n < 8:
        # trivial
        return float(np.std(values, ddof=1) or 1.0)

    x_min, x_max = float(values.min()), float(values.max())
    if x_min == x_max:
        return 1e-6

    grid = np.linspace(x_min, x_max, grid_size)

    # Search range for bandwidth
    std = float(np.std(values, ddof=1) or 1.0)
    h_low = max(std * 1e-6, 1e-6)
    h_high = std * 5.0 + 1e-6

    # Ensure upper bound is unimodal
    for _ in range(20):
        dens_high = _gaussian_kde_manual_density(grid, values, h_high)
        if _count_modes_from_density(dens_high) <= 1:
            break
        h_high *= 2.0

    # Binary search
    for _ in range(search_iters):
        h_mid = 0.5 * (h_low + h_high)
        dens_mid = _gaussian_kde_manual_density(grid, values, h_mid)
        modes = _count_modes_from_density(dens_mid)
        if modes <= 1:
            h_high = h_mid
        else:
            h_low = h_mid

    return float(h_high)


def _sample_from_kde_at_bandwidth(
    rng: np.random.Generator,
    values: np.ndarray,
    bandwidth: float,
    size: int
) -> np.ndarray:
    """
    Silverman bootstrap sampling:
    pick xi with replacement, add N(0, h) noise.

    Parameters
    ----------
    rng : np.random.Generator
        random number generator
    values : np.ndarray
        1D array of data points
    bandwidth : float
        bandwidth h for KDE
    size : int
        number of samples to generate

    Returns
    -------
    samples : np.ndarray
        generated samples
    """
    values = np.asarray(values, dtype=float).ravel()
    idx = rng.integers(0, len(values), size=size)
    base = values[idx]
    noise = rng.normal(loc=0.0, scale=float(bandwidth), size=size)
    return base + noise


def _silverman_unimodality_p_value(
    values: np.ndarray,
    seed: int,
    cfg: Dict[str, Any]
) -> float:
    """
    Silverman's test (k=1, unimodality):
    - compute critical bandwidth h_obs
    - bootstrap from KDE with bandwidth h_obs
    - compute h_boot each time
    - p = P(h_boot >= h_obs)

    This is a standard approach for Silverman 1981 style testing.

    Parameters
    ----------
    values : np.ndarray
        1D array of data points
    seed : int
        random seed for reproducibility
    cfg : Dict[str, Any]
        configuration dictionary

    Returns
    -------
    p_value : float
        estimated p-value for unimodality test
    """
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if n < 8:
        return 1.0

    grid_size = int(cfg.get("silverman_grid_size", 512))
    search_iters = int(cfg.get("silverman_search_iters", 30))
    b = int(cfg.get("silverman_bootstrap_samples", 200))

    rng = np.random.default_rng(seed)

    h_obs = _critical_bandwidth_unimodal(values, grid_size=grid_size, search_iters=search_iters)

    h_boot = np.empty(b, dtype=float)
    for i in range(b):
        sim = _sample_from_kde_at_bandwidth(rng, values, bandwidth=h_obs, size=n)
        h_boot[i] = _critical_bandwidth_unimodal(sim, grid_size=grid_size, search_iters=search_iters)

    p = float(np.mean(h_boot >= h_obs))
    return float(max(0.0, min(1.0, p)))


# smoothness_emd heuristic (bootstrap Wasserstein distance to fitted Gaussian)
def _emd_unimodality_p_value(
    values: np.ndarray,
    seed: int,
    cfg: Dict[str, Any]
) -> float:
    """
    Heuristic unimodality test using Wasserstein (EMD) distance:

    Null: best-fit unimodal Gaussian N(mu, sigma).
    Statistic: W1(values, gaussian_sample).

    Bootstrap:
    - Generate bootstrap samples from the fitted Gaussian with size n
    - For each, compute W1 vs another gaussian sample (or vs the fitted gaussian sample)
    - p = P(W1_boot >= W1_obs)

    Notes:
    - This is not a classical unimodality test, but works as a large-N-friendly heuristic
      that penalizes strong multi-peaked / heavy-tailed deviations.
    """
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)
    if n < 8:
        return 1.0

    b = int(cfg.get("emd_bootstrap_samples", 200))
    m = int(cfg.get("emd_sample_size", 4000))

    rng = np.random.default_rng(seed)
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1) or 1.0)

    # Observed statistic: W1 between data and gaussian sample
    g_obs = rng.normal(loc=mu, scale=sigma, size=min(m, max(2000, n)))
    w_obs = float(stats.wasserstein_distance(values, g_obs))

    # Bootstrap under Gaussian null:
    w_boot = np.empty(b, dtype=float)
    for i in range(b):
        x1 = rng.normal(loc=mu, scale=sigma, size=n)
        x2 = rng.normal(loc=mu, scale=sigma, size=min(m, max(2000, n)))
        w_boot[i] = float(stats.wasserstein_distance(x1, x2))

    p = float(np.mean(w_boot >= w_obs))
    return float(max(0.0, min(1.0, p)))


def _run_method(
    method: str,
    values: np.ndarray,
    seed: int,
    cfg: Dict[str, Any],
    max_sample: int
) -> Tuple[str, float]:
    """
    Run one method and return (method_used_name, p_value).
    """
    method = method.strip().lower()

    if method == "dip":
        p = _dip_p_value(values, seed=seed, cfg=cfg)
        return "dip", p

    if method == "dip_subsampled":
        v = _subsample(values, max_n=max_sample, seed=seed)
        p = _dip_p_value(v, seed=seed, cfg=cfg)
        return "dip_subsampled", p

    if method == "silverman":
        # Silverman is expensive; allow subsample via max_sample_for_test implicitly
        v = _subsample(values, max_n=max_sample, seed=seed)
        p = _silverman_unimodality_p_value(v, seed=seed, cfg=cfg)
        return "silverman", p

    if method == "kde_extrema":
        peak_count = _kde_peak_count(values, grid_size=int(cfg.get("silverman_grid_size", 512)))
        # same as your previous heuristic
        p = 0.0 if peak_count > 1 else 1.0
        return "kde_extrema", float(p)

    if method == "smoothness_emd":
        # fast-ish for large N, but still bootstrap
        v = _subsample(values, max_n=max_sample, seed=seed)
        p = _emd_unimodality_p_value(v, seed=seed, cfg=cfg)
        return "smoothness_emd", p

    raise ValueError(f"Unknown unimodality method: {method}")


def _is_flat_distribution(
    values: np.ndarray,
    kurtosis_threshold: float = -1.0,
    cv_threshold: float = 0.6
) -> bool:
    """
    Check if the distribution is flat/uniform-like based on kurtosis and CV only.

    平坦分佈特徵：
    1. Kurtosis 很負（< -1.0，uniform ≈ -1.2）
    2. CV (std/mean) 適中（排除 heavy-tail 和常數）
    3. Skewness 接近 0（對稱）

    Bimodal 通常有：
    - 較高的 bimodality coefficient
    - 或者 kurtosis 沒那麼負（兩個峰會拉高 kurtosis）

    Returns
    -------
    bool : True if distribution is flat/uniform-like
    """
    if len(values) < 10:
        return False

    kurt = stats.kurtosis(values, fisher=True)
    skew = stats.skew(values)

    mean_val = np.mean(values)
    median_val = np.median(values)
    if mean_val == 0:
        return False
    cv = np.std(values, ddof=1) / abs(mean_val)

    # === 新增：mean-median 差異檢查 ===
    # 如果 mean 和 median 差距過大，可能是 bimodal 或 skewed
    mean_median_ratio = abs(mean_val - median_val) / max(np.std(values, ddof=1), 1e-6)
    if mean_median_ratio > 0.3:  # mean 和 median 差超過 0.3 個標準差
        return False

    # === 收緊 skewness 門檻 ===
    is_platykurtic = kurt < kurtosis_threshold
    is_moderate_cv = 0.3 < cv < cv_threshold
    is_symmetric = abs(skew) < 0.3  # 從 0.5 改為 0.3，更嚴格

    return is_platykurtic and is_moderate_cv and is_symmetric


def run_unimodality_test(
    df: pd.DataFrame,
    mode: str = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Conduct uni-modality test on the given data slice.
    (Dip / Silverman / Heuristics / etc.)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe after scale transformation(single-category slice)
    mode : {"small", "medium", "large", "auto"}, default "small"
        - "small": use dip surrogate(full load)
        - "medium": use dip surrogate, subsample to max_sample_for_test if needed.
        - "large": use KDE extrema heuristic (no bootstrap, faster).
        - "auto" : automatically switch among the above three based on data size and data_size_thresholds.
    params : Dict[str, Any], optional
        additional parameters to control the test behavior.

    Returns
    -------
    unimodality_test_result : dict
        {
            "dip_p": float,
            "method_used": str,
            "decision": "unimodal" | "multimodal",
        }
    """
    if general_params is None:
        general_params = {}
    if mod_params is None:
        mod_params = {}

    cfg = {**DEFAULT_UNIMODALITY_PARAMS, **mod_params}

    logger.info("=== Unimodality Test Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    value_col = cfg["value_col"]
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame.")

    values = df[value_col].dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("No valid (non-NaN) values for uni-modality test.")

    # === 前置檢查：平坦分佈直接判定為 unimodal ===
    if cfg.get("check_flat_distribution", True):
        is_flat = _is_flat_distribution(
            values,
            kurtosis_threshold=cfg.get("flat_kurtosis_threshold", -1.0),
            cv_threshold=cfg.get("flat_cv_threshold", 0.6),
            #grid_size=int(cfg.get("silverman_grid_size", 512))
        )
        if is_flat:
            kurt_val = float(stats.kurtosis(values, fisher=True))
            logger.info(
                f"Flat distribution detected (kurtosis={kurt_val:.3f}). "
                f"Treating as unimodal."
            )
            return {
                "dip_p": 1.0,
                "method_used": "flat_distribution_check",
                "decision": "unimodal",
                "kurtosis": kurt_val,
            }

    if mode == "auto":
        mode_eff = _resolve_mode_auto(n, general_params)
    else:
        mode_eff = mode.lower()

    if mode_eff not in {"small", "medium", "large"}:
        raise ValueError(f"Unsupported mode: {mode_eff}")

    alpha = float(cfg["alpha"])
    max_sample = int(cfg["max_sample_for_test"])
    seed = int(general_params.get("random_seed", 42))

    # Select methods by mode (this is what was missing before)
    methods_by_mode = cfg.get("methods_by_mode", {})
    methods: List[str] = methods_by_mode.get(mode_eff, [])

    if not methods:
        raise ValueError(f"No methods configured for mode '{mode_eff}'. Check methods_by_mode.")

    # Run all methods, collect p-values
    used: List[str] = []
    pvals: List[float] = []

    for m in methods:
        name, p = _run_method(
            method=m,
            values=values,
            seed=seed,
            cfg=cfg,
            max_sample=max_sample,
        )
        used.append(name)
        pvals.append(float(p))

    # Spec-style aggregation:
    # - "if p < alpha OR multiple extrema -> multimodal"
    # Here we have multiple p-values; take the strongest evidence (min p).
    dip_p = float(max(0.0, min(1.0, np.min(pvals))))
    decision = "multimodal" if dip_p < alpha else "unimodal"

    return {
        "dip_p": dip_p,
        "method_used": "+".join(used),
        "decision": decision,
    }
