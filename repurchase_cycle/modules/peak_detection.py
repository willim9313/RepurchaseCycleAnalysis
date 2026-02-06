"""
Peak detection module
Using KDE to detect peaks in transformed interval days data.
Find peaks and calculate their characteristics (e.g. widths, prominences).
"""
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from KDEpy import FFTKDE
from scipy.stats import gaussian_kde
from scipy.signal import (
    find_peaks,
    peak_widths,
    peak_prominences,
    argrelmax
)
from sklearn.cluster import MeanShift
import logging

logger = logging.getLogger(__name__)


DEFAULT_PEAK_PARAMS: Dict[str, Any] = {
    # --- shared ---
    "height_min": 0.001,
    "prominence_min": 0.01,
    "grid_size": 512,

    # --- bandwidth split (fix #5) ---
    # small: scipy gaussian_kde bw_method factor
    "kde_bandwidth_factor": 0.5,

    # medium: KDEpy FFTKDE bw (same unit as x)
    "kde_bandwidth": 0.5,

    # large: MeanShift bandwidth (same unit as x)
    "meanshift_bandwidth": 1.0,

    # --- implementation detail for medium/large ---
    # if None -> auto from grid_size
    "argrelmax_order": None,

    # large: how to pick "high density" grid points for clustering
    # choose one of: "height_min" or "quantile"
    "large_density_filter": "quantile",
    "large_density_quantile": 0.85,
}


def _get_interval_values(
    df: pd.DataFrame,
    col_candidates: List[str]
) -> np.ndarray:
    """
    retrieve interval days values from transformed_df, prioritizing by col_candidates order.
    if none of the columns exist, raise KeyError.
    """
    for col in col_candidates:
        if col in df.columns:
            values = df[col].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if values.size == 0:
                raise ValueError(f"Column '{col}' has only NaN values.")
            return np.sort(values)
    raise KeyError(
        f"Cannot find any of columns {col_candidates} in transformed_df."
    )


def _build_grid(x: np.ndarray, grid_size: int) -> np.ndarray:
    """
    build evaluation grid for KDE according to data range.
    keep a bit of padding around.
    """
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_min == x_max:
        # return a small range if all points are the same to avoid division by zero
        x_min -= 1.0
        x_max += 1.0
    padding = 0.02 * (x_max - x_min)
    return np.linspace(x_min - padding, x_max + padding, grid_size)


def _fit_kde_scipy(
    x: np.ndarray,
    bandwidth_factor: float,
    grid: np.ndarray
) -> np.ndarray:
    """
    Using scipy KDE (scipy.stats.gaussian_kde) to fit KDE.
    bandwidth_factor is treated as bw_method scalar (covariance factor).
    Returns density values on the provided grid.
    """
    # Check for constant values (zero variance) - KDE cannot handle this
    if np.ptp(x) == 0 or np.std(x) == 0:
        # Return a single peak at the constant value
        density = np.zeros_like(grid, dtype=float)
        const_val = x[0]
        # Find the closest grid point to the constant value
        closest_idx = np.argmin(np.abs(grid - const_val))
        density[closest_idx] = 1.0
        return density

    kde = gaussian_kde(x, bw_method=bandwidth_factor)
    return kde(grid)


def _fit_kde_kdepy(
    x: np.ndarray,
    bandwidth: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using KDEpy FFTKDE to fit KDE.
    bandwidth is a bandwidth value in the same unit as x.
    Returns grid and density values.
    """
    x_2d = x.reshape(-1, 1)
    kde = FFTKDE(bw=bandwidth)
    grid, density = kde.fit(x_2d).evaluate(grid_size)
    return grid.ravel(), density


def _auto_argrelmax_order(grid_size: int, user_value: Optional[int]) -> int:
    """
    Determine argrelmax order.
    If user_value is provided, use it; otherwise choose an automatic value.
    """
    if user_value is not None:
        return int(user_value)
    return max(3, int(grid_size // 100))


def _peaks_table_from_indices(
    grid: np.ndarray,
    density: np.ndarray,
    peak_indices: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Given peak indices on a density curve, compute pos/height/width/prominence.
    """
    if peak_indices.size == 0:
        return []

    step = float(grid[1] - grid[0]) if grid.size > 1 else 1.0

    prominences, left_bases, right_bases = peak_prominences(density, peak_indices)
    widths_result = peak_widths(density, peak_indices, rel_height=0.5)
    widths_x = widths_result[0] * step

    peaks_table: List[Dict[str, float]] = []
    for idx, prom, width in zip(peak_indices, prominences, widths_x):
        peaks_table.append(
            {
                "pos": float(grid[idx]),
                "height": float(density[idx]),
                "width": float(width),
                "prominence": float(prom),
            }
        )

    peaks_table.sort(key=lambda r: r["pos"])
    return peaks_table


def _detect_peaks_small(
    x: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """
    small: KDE(scipy) + scipy.find_peaks
    """
    grid = _build_grid(x, int(cfg["grid_size"]))
    density = _fit_kde_scipy(x, float(cfg["kde_bandwidth_factor"]), grid)

    peak_indices, properties = find_peaks(
        density,
        height=float(cfg["height_min"]),
        prominence=float(cfg["prominence_min"]),
    )
    peaks_table = _peaks_table_from_indices(grid, density, peak_indices)
    return peaks_table, grid, density


def _detect_peaks_medium(
    x: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """
    medium: KDEpy FFT + argrelmax
    """
    grid, density = _fit_kde_kdepy(
        x=x,
        bandwidth=float(cfg["kde_bandwidth"]),
        grid_size=int(cfg["grid_size"]),
    )

    order = _auto_argrelmax_order(int(cfg["grid_size"]), cfg.get("argrelmax_order"))
    candidate_idx = argrelmax(density, order=order)[0]

    if candidate_idx.size == 0:
        return [], grid, density

    # apply height filter
    height_min = float(cfg["height_min"])
    idx = candidate_idx[density[candidate_idx] >= height_min]
    if idx.size == 0:
        return [], grid, density

    # compute prominence then filter by prominence_min
    prominences, _, _ = peak_prominences(density, idx)
    prom_min = float(cfg["prominence_min"])
    idx = idx[prominences >= prom_min]
    if idx.size == 0:
        return [], grid, density

    peaks_table = _peaks_table_from_indices(grid, density, idx)
    return peaks_table, grid, density


def _detect_peaks_large(
    x: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    """
    large: MeanShift clustering on density
    - Fit KDE on grid (KDEpy FFT for speed)
    - Select high-density grid points
    - MeanShift on those grid points (1D)
    - Map centers back to nearest grid index, compute true prominence/width on density curve
    """
    grid, density = _fit_kde_kdepy(
        x=x,
        bandwidth=float(cfg["kde_bandwidth"]),
        grid_size=int(cfg["grid_size"]),
    )

    # pick high-density points
    filter_mode = str(cfg.get("large_density_filter", "quantile")).lower()
    if filter_mode == "height_min":
        thr = float(cfg["height_min"])
    else:
        q = float(cfg.get("large_density_quantile", 0.85))
        thr = float(np.quantile(density, q))

    mask = density >= thr
    grid_hi = grid[mask]

    if grid_hi.size == 0:
        return [], grid, density

    X = grid_hi.reshape(-1, 1)
    ms = MeanShift(bandwidth=float(cfg["meanshift_bandwidth"]), bin_seeding=True)
    ms.fit(X)
    centers = ms.cluster_centers_.ravel()

    # Map centers to nearest density-curve index
    center_idx = np.array([int(np.argmin(np.abs(grid - c))) for c in centers], dtype=int)
    center_idx = np.unique(center_idx)  # remove duplicates if very close

    # Optionally: ensure these are actual local maxima on density curve
    # We enforce by snapping to nearest local max in a neighborhood
    order = _auto_argrelmax_order(int(cfg["grid_size"]), cfg.get("argrelmax_order"))
    local_max_idx = set(argrelmax(density, order=order)[0].tolist())

    # if local_max_idx is empty, retry with smaller order
    if len(local_max_idx) == 0:
        for fallback_order in [3, 2, 1]:
            local_max_idx = set(argrelmax(density, order=fallback_order)[0].tolist())
            if len(local_max_idx) > 0:
                logger.warning(f"No local max found with order={order}, falling back to order={fallback_order}")
                order = fallback_order
                break

    snapped_idx: List[int] = []
    for ci in center_idx.tolist():
        if ci in local_max_idx:
            snapped_idx.append(ci)
            continue
        # search within a larger neighborhood - 使用 grid_size 的一定比例
        search_range = max(order, int(cfg["grid_size"] // 20))  # 改為 grid_size // 20 ≈ 25 points
        left = max(0, ci - search_range)
        right = min(len(density) - 1, ci + search_range)
        candidates = [i for i in range(left, right + 1) if i in local_max_idx]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: density[i])
        snapped_idx.append(best)

    if len(snapped_idx) == 0:
        return [], grid, density

    idx = np.unique(np.array(snapped_idx, dtype=int))

    # apply height/prominence filters (same semantics as other modes)
    height_min = float(cfg["height_min"])
    idx = idx[density[idx] >= height_min]
    if idx.size == 0:
        return [], grid, density

    prominences, _, _ = peak_prominences(density, idx)
    prom_min = float(cfg["prominence_min"])

    idx = idx[prominences >= prom_min]
    if idx.size == 0:
        return [], grid, density

    peaks_table = _peaks_table_from_indices(grid, density, idx)
    return peaks_table, grid, density


def run_peak_detection(
    profile_name: Optional[str],
    df: pd.DataFrame,
    mode: str = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "./reports",
    transform_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, float]], str]:
    """
    Run peak detection on transformed interval days data.

    Parameters
    ----------
    profile_name : str, optional
        analysis config name used for report title etc.
        if None, use generic names.
    df : pd.DataFrame
        transformed interval days DataFrame, must contain repurchase cycle columns.
        By default, it will look for the following columns in order:
        - 'interval_days_transformed'
        - 'interval_days'
    mode : {'small', 'medium', 'large'}
        - small  : scipy KDE + scipy.signal.find_peaks
        - medium : KDEpy FFTKDE + argrelmax
        - large  : MeanShift clustering on density
    params : dict, optional
        override default parameters for peak detection module.
        see DEFAULT_PEAK_PARAMS for available parameters.
    output_dir : str, default "./data/reports"
        output directory for saved plots.

    Returns
    -------
    peaks_table : List[Dict[str, float]]
        each peak's feature indicators:
        - pos : x position of the peak (repurchase days coordinate)
        - height : height of the KDE curve at that point
        - width : estimated peak width (same unit as x-axis)
        - prominence : prominence of the peak (or height approximation in large mode)
    kde_plot_with_peaks : str
        path to the saved KDE+peaks plot image.
    """
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    if mod_params is None:
        mod_params = {}

    cfg = {**DEFAULT_PEAK_PARAMS, **mod_params}
    logger.info("=== Peak Detection Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    mode = mode.lower()
    if mode not in {"small", "medium", "large"}:
        raise ValueError("mode must be one of {'small', 'medium', 'large'}")

    # 1. retrieve transformed interval days values
    x = _get_interval_values(df, ["interval_days_transformed", "interval_days"])

    # 2. determine KDE and peak detection method based on mode
    if mode == "small":
        peaks_table, grid, density = _detect_peaks_small(x, cfg)
    elif mode == "medium":
        peaks_table, grid, density = _detect_peaks_medium(x, cfg)
    else:
        peaks_table, grid, density = _detect_peaks_large(x, cfg)

    # 3. export plot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f"{profile_name}_peak_detection_kde_{mode}.png"
        if profile_name else
        f"peak_detection_kde_{mode}.png"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, density, label="KDE density")
    if len(peaks_table) > 0:
        peak_positions = np.array([p["pos"] for p in peaks_table], dtype=float)
        peak_heights = np.array([p["height"] for p in peaks_table], dtype=float)
        ax.scatter(peak_positions, peak_heights, marker="x", s=80, c="red", label="Peaks", zorder=5)

        for p in peaks_table:
            ax.annotate(
                f'{p["pos"]:.2f}',
                xy=(p["pos"], p["height"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    ax.set_xlabel("Transformed interval days")
    ax.set_ylabel("Density")
    ax.set_title(f"Repurchase Cycle KDE with Peaks ({mode})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    # inverse transform if needed
    if transform_meta and transform_meta.get("method") != "none":
        from repurchase_cycle.modules.transform import _inverse_transform
        method = transform_meta["method"]
        transform_params = transform_meta.get("transform_params", {})

        for peak in peaks_table:
            # inverse transform the transformed pos back to original scale
            pos_series = pd.Series([peak["pos"]])
            original_pos = _inverse_transform(pos_series, method, transform_params)
            peak["pos_original"] = float(original_pos.iloc[0])
            peak["pos_transformed"] = peak["pos"]  # keep the transformed value for internal use
    return peaks_table, plot_path
