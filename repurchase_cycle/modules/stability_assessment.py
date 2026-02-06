import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

DEFAULT_STABILITY_PARAMS: Dict[str, Any] = {
    "n_bootstrap": 100,
    "sample_fraction": 0.8,
    "support_threshold": 0.6,
    "value_col": "interval_days_transformed",
    "match_tol": None,  # auto-determined
    "grid_size": 512
}

MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "small": {"n_bootstrap": 100, "sample_fraction": 0.8},
    "medium": {"n_bootstrap": 80, "sample_fraction": 0.7},
    "large": {"n_bootstrap": 50, "sample_fraction": 0.6},
}


def _save_stability_plot(
    profile_name: Optional[str],
    target_positions: np.ndarray,
    support_ratios: np.ndarray,
    support_threshold: float,
    output_dir: str
) -> str:
    """Generate and save the stability assessment plot."""
    if len(support_ratios) == 0:
        return ""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{profile_name}_stability_assessment_peaks.png" if profile_name else "stability_assessment_peaks.png"
    plot_path = os.path.join(output_dir, filename)

    order = np.argsort(target_positions)
    sorted_pos = target_positions[order]
    sorted_ratio = support_ratios[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['green' if r >= support_threshold else 'salmon' for r in sorted_ratio]
    ax.bar(range(len(sorted_pos)), sorted_ratio, color=colors)
    ax.set_xticks(range(len(sorted_pos)))
    ax.set_xticklabels([f"{p:.2f}" for p in sorted_pos], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(support_threshold, linestyle="--", color="red", label=f"Threshold ({support_threshold})")
    ax.set_ylabel("Support Ratio")
    ax.set_xlabel("Peak Position (transformed scale)")
    ax.set_title("Peak Stability Assessment (Bootstrap)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def run_stability_assessment(
    profile_name: Optional[str],
    df: pd.DataFrame,
    peaks_table: List[Dict[str, float]],
    mode: str = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "./reports",
    transform_meta: Optional[Dict[str, Any]] = None,  # 新增參數
) -> Tuple[List[Dict[str, float]], str]:
    """
    Validate peak stability through bootstrap.

    Parameters
    ----------
    profile_name : str, optional
        Analysis profile name (used for report titles, etc.).
    df : pd.DataFrame
        Transformed data (transformed_df), must contain the target numeric column.
    peaks_table : List[Dict[str, float]]
        List of peaks from the peak detection module, for example:
        [
            {"pos": 5.0, "height": 0.12, "width": 0.9, "prominence": 0.03},
            ...
        ]
        Only the "pos" key is used here.
    mode : {"small", "medium", "large"}, default "small"
        Corresponding to data size strategies in the spec:
        - "small":      Bootstrap KDE (more iterations)
        - "medium":     Bootstrap + Subsample Consensus
        - "large":      Multi-bandwidth-like fast approximation (fewer iterations)
    general_params : dict, optional
        General parameter dictionary (not used in this function).
        - random_state : int
    mod_params : dict, optional
        Parameter override dictionary, using default values if not provided.
        Available keys:
        - "value_col": str, default "interval_days_transformed"
            The column name to sample/build KDE from.
        - "n_bootstrap": int
        - "sample_fraction": float in (0, 1]
        - "support_threshold": float in [0, 1]
        - "match_tol": float, peak position matching tolerance (same scale)
        - "grid_size": int, default 512
    output_dir : str, default "./data/reports"
        Folder to save the stability assessment plot.

    Returns
    -------
    stable_peaks_table : List[Dict[str, float]]
        List of stable peaks passing the support threshold, each containing:
        - "pos": float
        - "support_ratio": float
    stability_plot : str
        Stability plot file path (PNG). Returns an empty string if no peaks or insufficient data.
    """
    if general_params is None:
        general_params = {}
    if mod_params is None:
        mod_params = {}

    cfg = {
        **DEFAULT_STABILITY_PARAMS,
        **MODE_DEFAULTS.get(mode, {}),
        **(mod_params or {}),
    }
    logger.info("=== Stability Assessment Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    # Parse and set parameters (including mode defaults)
    value_col: str = cfg.get("value_col", "interval_days_transformed")
    n_bootstrap: int = int(cfg["n_bootstrap"])
    sample_fraction: float = float(cfg["sample_fraction"])
    support_threshold: float = float(cfg["support_threshold"])
    grid_size: int = int(cfg["grid_size"])
    random_state = general_params.get("random_state", 42)

    # early exit if no peaks
    if not peaks_table:
        return [], ""

    target_positions = np.array([float(p["pos"]) for p in peaks_table], dtype=float)
    n_peaks = target_positions.shape[0]

    # Extract numeric data
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in DataFrame columns: {list(df.columns)}")

    values = df[value_col].dropna().to_numpy()

    if values.size == 0:
        return [], ""

    # Decide peak matching tolerance
    data_min, data_max = float(values.min()), float(values.max())
    data_range = data_max - data_min if data_max > data_min else 1.0
    raw_match_tol = cfg.get("match_tol")
    if raw_match_tol is None or raw_match_tol == "None" or raw_match_tol == "null":
        match_tol: float = 0.05 * data_range
    else:
        match_tol: float = float(raw_match_tol)

    # Bootstrap + KDE + peak detection
    rng = np.random.default_rng(random_state)
    support_counts = np.zeros(n_peaks, dtype=int)
    # To avoid recalculating boundaries each time, fix the grid
    x_grid = np.linspace(data_min, data_max, grid_size)
    effective_bootstrap = 0  # Track the actual successful bootstrap count

    sample_size = max(2, int(values.size * sample_fraction))

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, values.size, size=sample_size)
        sample = values[sample_indices]

        try:
            kde = gaussian_kde(sample)
        except Exception:
            continue

        effective_bootstrap += 1
        density = kde(x_grid)
        peak_indices, _ = find_peaks(density)

        if peak_indices.size == 0:
            continue

        bootstrap_positions = x_grid[peak_indices]

        # check each original peak against bootstrap peaks whether any is within match_tol
        for i, pos in enumerate(target_positions):
            if np.any(np.abs(bootstrap_positions - pos) <= match_tol):
                support_counts[i] += 1

    # calculate effective bootstrap count
    if effective_bootstrap == 0:
        return [], ""

    support_ratios = support_counts.astype(float) / float(effective_bootstrap)

    stable_peaks_table: List[Dict[str, float]] = [
        {"pos": float(pos), "support_ratio": float(ratio)}
        for pos, ratio in zip(target_positions, support_ratios)
        if ratio >= support_threshold
    ]

    plot_path = _save_stability_plot(
        profile_name, target_positions, support_ratios,
        support_threshold, output_dir
    )

    # Before returning, add original scale positions
    if transform_meta and transform_meta.get("method") != "none":
        from repurchase_cycle.modules.transform import _inverse_transform
        method = transform_meta["method"]
        transform_params = transform_meta.get("transform_params", {})

        for peak in stable_peaks_table:
            pos_series = pd.Series([peak["pos"]])
            original_pos = _inverse_transform(pos_series, method, transform_params)
            peak["pos_original"] = float(original_pos.iloc[0])
            peak["pos_transformed"] = peak["pos"]

    return stable_peaks_table, plot_path
