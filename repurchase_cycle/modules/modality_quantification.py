import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Sequence, Tuple

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODALITY_PARAMS: Dict[str, Any] = {
    "k_range": [1, 6],
    "selection_metric": "BIC",
    "subsample_size": 20000,
    "max_iter": 500,
    "n_init": 5,
    "kde_grid_size": 512,
    "dp_weight_threshold": 0.01,
}


def _resolve_k_range(raw_k_range: Any) -> Sequence[int]:
    """
    transform k_range parameter into list of integers.

    Support:
    - None -> [1, 2, 3, 4, 5, 6]
    - int -> range(1, int+1)
    - (start, end) or [start, end] -> range(start, end+1)
    - direct list[int] / tuple[int]
    """
    if raw_k_range is None:
        return list(range(1, 7))

    if isinstance(raw_k_range, int):
        return list(range(1, raw_k_range + 1))

    if isinstance(raw_k_range, (list, tuple)):
        if (
            len(raw_k_range) == 2
            and all(isinstance(v, int) for v in raw_k_range)
            and raw_k_range[0] < raw_k_range[1]
        ):
            start, end = raw_k_range
            return list(range(start, end + 1))

    # fallback: try to convert each element to int
    return list(int(k) for k in raw_k_range)


def _select_best_k(
    aic_scores: Dict[int, float],
    bic_scores: Dict[int, float],
    metric: str,
) -> int:
    """
    select the best K based on AIC or BIC scores.
    the smaller the score, the better.
    """
    metric = (metric or "BIC").upper()
    if metric == "AIC":
        scores = aic_scores
    else:
        scores = bic_scores

    # if scores is empty, return 1(e.g., no valid K fitted)
    if not scores:
        return 1

    return min(scores.items(), key=lambda kv: kv[1])[0]


def _estimate_kde_n_peaks(
    x: np.ndarray,
    grid_size: int = 512,
    prominence_factor: float = 0.05,
    height_factor: float = 0.02,
) -> int:
    """
    Estimate the pke count using KDE.
    Used to compare consistency with GMM results.

    Parameters
    ----------
    x : np.ndarray
        1d data array, already cleaned of NaNs.
    grid_size : int
        evaluation grid size for KDE
    prominence_factor : float
        peak prominence relative to the maximum density
    height_factor : float
        peak height relative to the maximum density

    Returns
    -------
    int
        Estimated number of peaks.
    """
    if x.size < 2 or np.allclose(x, x[0]):
        # Almost constant or single value -> consider as unimodal
        return 1

    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), grid_size)
    density = kde(grid)

    if density.max() <= 0:
        return 1

    prominence = prominence_factor * density.max()
    height = height_factor * density.max()

    peaks, _ = find_peaks(density, prominence=prominence, height=height)
    # at least consider as 1 peak
    return max(len(peaks), 1)


def run_modality_quantification(
    df: pd.DataFrame,
    mode: str = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Using GMM / DP-GMM to estimate the number of modalities in the interval_days data.
    Compare with KDE peak count for consistency check.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the `interval_days` column, already scale-transformed.
    mode : {"small", "medium", "large"}
        Corresponding to the spec's small / medium / large data volume processing strategies:
        - small  : GMM full loop (1..K_max)
        - medium : Subsample GMM + AIC/BIC
        - large  : DP-GMM / Variational Inference (BayesianGaussianMixture)
    params : dict, optional
        override default module-level parameters, for example:
        - "k_range": [1, 6] or [1, 3, 5], etc.
        - "selection_metric": "AIC" or "BIC" (default "BIC")
        - "subsample_size": 20000 (only for medium)
        - "random_state": 42
        - "max_iter": 500
        - "n_init": 5
        - "kde_grid_size": 512
        - "dp_weight_threshold": 0.01 (only for large)

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (
          {
              "best_n_components": int,
              "aic_scores": list[float],
              "bic_scores": list[float]
          },
          {
              "kde_n_peaks": int,
              "gmm_n_components": int,
              "status": "consistent" or "inconsistent"
          }
        )
    """
    # 1. Read and preprocess data
    if "interval_days" not in df.columns:
        raise ValueError("run_modality_quantification requires df to contain the 'interval_days' column.")

    x = df["interval_days"].to_numpy(dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        # Extreme case: no data at all, return default result directly
        return (
            {
                "best_n_components": 1,
                "aic_scores": [],
                "bic_scores": [],
            },
            {
                "kde_n_peaks": 1,
                "gmm_n_components": 1,
                "status": "consistent",
            }
        )

    x = x.reshape(-1, 1)
    n_samples = x.shape[0]

    # 2. Unify default parameters
    if mod_params is None:
        mod_params = {}
    if general_params is None:
        general_params = {}

    cfg = {**DEFAULT_MODALITY_PARAMS, **mod_params}
    logger.info("=== Modality Quantification Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    k_values = _resolve_k_range(cfg["k_range"])
    if not k_values:
        k_values = [1]

    random_state = general_params.get("random_state", 42)
    max_iter = cfg["max_iter"]
    n_init = cfg["n_init"]

    # 3. Execute GMM / DP-GMM based on mode
    aic_scores_map: Dict[int, float] = {}
    bic_scores_map: Dict[int, float] = {}

    mode = mode.lower()
    if mode not in {"small", "medium", "large"}:
        raise ValueError(f"Unknown mode: {mode}. Expected 'small', 'medium', or 'large'.")

    # --- small / medium: traditional GMM ---
    if mode in {"small", "medium"}:
        x_fit = x

        # medium mode: if sample size is too large, do random sub-sampling
        if mode == "medium" and n_samples > cfg["subsample_size"]:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_samples, size=cfg["subsample_size"], replace=False)
            x_fit = x[idx]

        for k in k_values:
            # if sample size < k, skip fitting(gmm will error out)
            if x_fit.shape[0] < k:
                continue

            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state,
            )
            try:
                gm.fit(x_fit)
                aic_scores_map[k] = float(gm.aic(x_fit))
                bic_scores_map[k] = float(gm.bic(x_fit))
            except (ConvergenceWarning, ValueError, np.linalg.LinAlgError):
                # skip invalid K that cannot be fitted
                continue
            except Exception:
                raise

        best_k = _select_best_k(aic_scores_map, bic_scores_map, cfg["selection_metric"])

    # --- large: using BayesianGaussianMixture to approximate DP-GMM ---
    else:  # mode == "large"
        max_components = max(k_values)
        if x.shape[0] < 2:
            best_k = 1
        else:
            bgm = BayesianGaussianMixture(
                n_components=max_components,
                covariance_type="full",
                max_iter=max_iter,
                n_init=1,
                random_state=random_state,
                weight_concentration_prior_type="dirichlet_process",
            )
            try:
                bgm.fit(x)
                weight_threshold = cfg["dp_weight_threshold"]
                effective_components = int(np.sum(bgm.weights_ > weight_threshold))
                best_k = max(effective_components, 1)

            except Exception as e:
                raise RuntimeError("BayesianGaussianMixture fit failed") from e

    # export list by k_values, not dict, default None for missing K
    aic_scores_list = [aic_scores_map.get(k, None) for k in k_values]
    bic_scores_list = [bic_scores_map.get(k, None) for k in k_values]

    # KDE peak count estimation for consistency check
    kde_n_peaks = _estimate_kde_n_peaks(
        x=x.ravel(),
        grid_size=cfg["kde_grid_size"],
    )

    status = "consistent" if kde_n_peaks == best_k else "inconsistent"

    gmm_result = {
        "best_n_components": int(best_k),
        "aic_scores": aic_scores_list,
        "bic_scores": bic_scores_list,
    }
    consistency_check = {
        "kde_n_peaks": int(kde_n_peaks),
        "gmm_n_components": int(best_k),
        "status": status,
    }
    return gmm_result, consistency_check
