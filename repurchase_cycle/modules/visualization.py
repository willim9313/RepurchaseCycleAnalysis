import os
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import repurchase_cycle.modules.ptitprince as pt
import logging

logger = logging.getLogger(__name__)

DEFAULT_VIZ_PARAMS = {
    "sample_ratio": 0.05,
    "kde_bandwidths": [0.3, 0.6, 1.0],
    "plot_types": ["hist_kde", "violin", "box", "cdf", "raincloud"],
    "orient": "h",
    "palette": "Set2",
    "sigma": 0.2,
    "data_hue": None,
    "multi_category": False,
    "base_plots_dir": "./plots"
}


def _setup_plot_style() -> None:
    """
    Setup plot style for all plots.
    """
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.titleweight": "bold",
        "axes.titlepad": 16,
        "axes.labelpad": 8,
    })


def _auto_select_mode(
    n_rows: int,
    explicit_mode: str,
    small_threshold: int = int(1e4),
    medium_threshold: int = int(1e6),
) -> str:
    """
    according to data size and user specified mode to decide actual mode.

    Parameters
    ----------
    n_rows : int
        data size (number of rows)
    explicit_mode : str
        user specified mode: small/medium/large/auto
    small_threshold : int
        threshold for small mode
    medium_threshold : int
        threshold for medium mode

    Returns
    -------
    str
        selected mode: small/medium/large

    Notes:
    - If explicit_mode is small/medium/large, use it directly.
    - If explicit_mode is auto, determine based on n_rows:
        n <= small_threshold  -> small
        small_threshold < n <= medium_threshold -> medium
        n > medium_threshold -> large
    """
    if explicit_mode in {"small", "medium", "large"}:
        return explicit_mode

    if explicit_mode not in {"auto", None, ""}:
        logger.warning("Unknown mode %s, fallback to 'auto'", explicit_mode)

    if n_rows <= small_threshold:
        return "small"
    if n_rows <= medium_threshold:
        return "medium"
    return "large"


def _build_approx_series_large(
    s: pd.Series,
    max_points: int = 5000,
    min_bins: int = 20,
    max_bins: int = 120,
) -> pd.Series:
    """
    Large data approximation via histogram binning.

    Parameters
    ----------
    s : pd.Series
        raw data Series of numerical values.(NA removed)
    max_points : int
        Maximum number of points to retain after approximation to control
        computation and plotting load.
    min_bins : int
        Minimum number of histogram bins.
    max_bins : int
        Maximum number of histogram bins.

    Returns
    -------
    pd.Series approximated Series of numerical values.
    """
    s = s.dropna()
    n = len(s)
    if n == 0:
        return s

    # bin counts variate between min_bins and max_bins
    n_bins = int(np.clip(np.sqrt(n), min_bins, max_bins))
    counts, bin_edges = np.histogram(s.values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    approx_values: List[float] = []
    for center, count in zip(bin_centers, counts):
        if count <= 0:
            continue
        # Distribute counts proportionally to max_points
        n_rep = int(round(count / n * max_points))
        if n_rep == 0:
            n_rep = 1
        approx_values.extend([center] * n_rep)

    return pd.Series(approx_values, name=s.name)


def _get_working_series(
    df: pd.DataFrame,
    col: str,
    mode: str,
    sample_ratio: float,
    random_seed: int,
) -> pd.Series:
    """
    retrieve the actual Series used for plotting according to mode.

    - small: full data
    - medium: sampling (sample_ratio)
    - large: histogram-based approximation
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    s = df[col].dropna()
    n = len(s)
    if n == 0:
        return s

    rng = np.random.default_rng(random_seed)

    if mode == "small":
        return s

    if mode == "medium":
        frac = np.clip(sample_ratio, 0.0, 1.0)
        if frac <= 0:
            frac = 0.05
        n_sample = max(1, int(round(n * frac)))
        idx = rng.choice(n, size=n_sample, replace=False)
        return s.iloc[idx]

    if mode == "large":    
        return _build_approx_series_large(s)

    # fallback
    logger.warning("Unknown mode %s, fallback to 'small'", mode)
    return s


def _plot_raincloud(
    df: pd.DataFrame,
    output_path: str,
    data_hue: Optional[str] = None,
    palette: Optional[str] = None,
    width_viol: float = 0.6,
    orient: str = "h",
    sigma: float = 0.2,
    **kwargs
) -> None:
    """
    Plot raincloud chart using ptitprince library.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot. Must include 'interval_days_transformed' column.
    output_path : str
        File path to save the generated plot.
    data_hue : Optional[str], default None
        Column name for hue grouping. If None, no hue grouping is applied.
    palette : Optional[str], default None
        Color palette for the plot.
    width_viol : float, default 0.6
        Width of the violin plot component.
    orient : str, default "h"
        Orientation of the plot, "h" for horizontal, "v" for vertical.
    sigma : float, default 0.2
        Bandwidth for the kernel density estimation.
    """
    if "cat" not in df.columns:
        logger.warning("Column 'cat' not found in DataFrame, skipping raincloud plot.")
        return
    if "interval_days_transformed" not in df.columns:
        logger.warning("Column 'interval_days_transformed' not found, skipping raincloud plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    pt.RainCloud(
        x="cat",
        y="interval_days_transformed",
        hue=data_hue,
        data=df,
        palette=palette,
        width_viol=width_viol,
        ax=ax,
        orient=orient,
        bw=sigma
    )

    plt.title("Raincloud plot")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_hist_kde(
    s: pd.Series,
    kde_bandwidths: List[float],
    output_path: str,
) -> None:
    """
    plot histogram + multiple KDE overlays.
    """
    fig, ax = plt.subplots()

    # histogram
    sns.histplot(
        s,
        stat="density",
        kde=False,
        bins="auto",
        edgecolor=None,
        alpha=0.35,
        ax=ax,
    )

    # multiple KDE lines (using bw_adjust to simulate different bandwidths)
    base_label = "KDE (bw_adjust={:.2f})"
    for bw in kde_bandwidths:
        sns.kdeplot(
            s,
            bw_adjust=bw,
            linewidth=2,
            alpha=0.9,
            ax=ax,
            label=base_label.format(bw),
        )

    ax.set_title("Histogram + KDE (interval_days)")
    ax.set_xlabel("Interval Days")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_violin(
    s: pd.Series,
    output_path: str,
) -> None:
    """
    single dimensional violin plot.
    """
    fig, ax = plt.subplots()

    sns.violinplot(
        y=s,
        inner="quartile",
        cut=0,
        linewidth=1.2,
        ax=ax,
    )
    ax.set_title("Violin Plot (interval_days)")
    ax.set_ylabel("Interval Days")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_box(
    s: pd.Series,
    output_path: str,
) -> None:
    """
    Single dimensional boxplot.
    """
    fig, ax = plt.subplots()

    sns.boxplot(
        y=s,
        width=0.3,
        ax=ax,
    )
    ax.set_title("Box Plot (interval_days)")
    ax.set_ylabel("Interval Days")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_cdf(
    s: pd.Series,
    output_path: str,
) -> None:
    """
    Empirical CDF plot.
    """
    fig, ax = plt.subplots()

    x = np.sort(s.values)
    n = len(x)
    y = np.linspace(0.0, 1.0, n, endpoint=True)

    ax.step(x, y, where="post")
    ax.set_title("Empirical CDF (interval_days)")
    ax.set_xlabel("Interval Days")
    ax.set_ylabel("Cumulative Probability")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_visualization(
    df: pd.DataFrame,
    mode: str = "small",
    viz_prefix: Optional[str] = None,
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "./data/reports",
) -> Tuple[str, Dict[str, float]]:
    """
    Plot various visualizations for interval days.
    Plot histogram + KDE + violin + boxplot + CDF etc. for manual inspection.
    Note that the data column used for plotting is currently fixed to interval_days_transformed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the interval_days column (already transformed).
    mode : {"small", "medium", "large", "auto"}, default "small"
        small  : 全資料繪製
        medium : Sampled according to sample_ratio
        large  : Approximate plotting after bucketing
        auto   : Automatically select small/medium/large based on data size
    general_params : dict, optional
        - data_size_thresholds: Dict[str, float], optional
            {"small": 1e4, "medium": 1e6}
        - "random_seed": int, default 42
    mod_params : dict, optional
        override default visualization parameters. See DEFAULT_VIZ_PARAMS for details.
    viz_prefix: Optional[str] = None,
        Prefix string for image filenames. Default is None, meaning no prefix.
    output_dir : str, default "./data/reports"
        Directory for saving images.
    Returns
    -------
    plots_dir : str
        Directory path where the final images are saved.
    summary_stats : dict
        {
            "n": float,
            "mean": float,
            "median": float,
            "std": float,
            "skew": float,
        }
    """
    # 0. setup config
    if general_params is None:
        general_params = {
            "data_size_thresholds": {"small": int(1e4), "medium": int(1e6)},
            "random_seed": 42,
        }
    if mod_params is None:
        mod_params = {}
    cfg = {**DEFAULT_VIZ_PARAMS, **mod_params}
    logger.info("=== Visualization Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    n_rows = len(df)
    actual_mode = _auto_select_mode(
        n_rows=n_rows,
        explicit_mode=mode,
        small_threshold=general_params["data_size_thresholds"]["small"],
        medium_threshold=general_params["data_size_thresholds"]["medium"],
    )
    logger.info(
        "run_visualization: n_rows=%d, mode=%s (resolved=%s)",
        n_rows,
        mode,
        actual_mode,
    )

    # Prepare data
    s_all = df["interval_days_transformed"].dropna()
    if s_all.empty:
        raise ValueError("No non-null values found in column 'interval_days_transformed'.")

    s_plot = _get_working_series(
        df=df,
        col="interval_days_transformed",
        mode=actual_mode,
        sample_ratio=cfg["sample_ratio"],
        random_seed=general_params["random_seed"],
    )

    # summary stats always use the full data (not affected by sampling, bucketing, or multiple categories)
    summary_stats = {
        "n": float(n_rows),
        "mean": float(s_all.mean()),
        "median": float(s_all.median()),
        "std": float(s_all.std(ddof=1)),
        "skew": float(s_all.skew()),
    }

    plots_dir = os.path.join(output_dir, "visualization")
    os.makedirs(plots_dir, exist_ok=True)
    prefix = f"{viz_prefix}_" if viz_prefix else ""

    # Plotting
    _setup_plot_style()

    if "raincloud" in cfg["plot_types"]:
        _plot_raincloud(
            df=df,
            output_path=os.path.join(plots_dir, f"{prefix}interval_raincloud.png"),
            data_hue=cfg["data_hue"],
            palette=cfg["palette"],
            width_viol=0.6,
            orient=cfg["orient"],
            sigma=cfg["sigma"]
        )

    # hist + KDE
    if "hist_kde" in cfg["plot_types"]:
        _plot_hist_kde(
            s=s_plot,
            kde_bandwidths=cfg["kde_bandwidths"],
            output_path=os.path.join(plots_dir, f"{prefix}interval_hist_kde.png"),
        )

    # violin
    if "violin" in cfg["plot_types"]:
        _plot_violin(
            s=s_plot,
            output_path=os.path.join(plots_dir, f"{prefix}interval_violin.png"),
        )

    # boxplot
    if "box" in cfg["plot_types"]:
        _plot_box(
            s=s_plot,
            output_path=os.path.join(plots_dir, f"{prefix}interval_box.png"),
        )

    # CDF
    if "cdf" in cfg["plot_types"]:
        _plot_cdf(
            s=s_plot,
            output_path=os.path.join(plots_dir, f"{prefix}interval_cdf.png"),
        )

    return plots_dir, summary_stats
