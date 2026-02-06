"""repurchase_cycle.modules.transform"""
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox1p
import logging

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORM_PARAMS = {
    "method_candidates": ["log1p", "yeo_johnson", "none"],
    "auto_select_by_skewness": True,
    "skew_threshold": 0.5
}

MODE_DEFAULT_METHODS = {
    "medium": "log1p",
    "large": "yeo_johnson",
}


def _apply_transform(series: pd.Series, method: str) -> pd.Series:
    """
    apply transformation to a pandas Series based on the specified method.

    parameters
    ----------
    series : pd.Series
        The input data series to be transformed.
    method : str
        The transformation method to apply. Supported methods are:
        - "log1p": Apply log(1 + x) transformation.
        - "yeo_johnson": Apply Yeo-Johnson transformation.
        - "none": No transformation, return the original series.
    """
    s = series.copy()
    # 如果沒有實際數值，直接回傳原 series
    if s.dropna().size == 0:
        return s

    if method == "log1p":
        return pd.Series(np.log1p(s.clip(lower=0)), index=s.index)
    elif method == "yeo_johnson":
        vals = s.fillna(0).values
        transformed = stats.yeojohnson(vals)[0]
        return pd.Series(transformed, index=s.index)
    elif method == "none":
        return s
    else:
        raise ValueError(f"Unsupported transform method: {method}")


def _inverse_transform(
    series: pd.Series,
    method: str,
    transform_params: Optional[Dict] = None
) -> pd.Series:
    """
    Apply inverse transformation to convert values back to original scale.

    Parameters
    ----------
    series : pd.Series
        The transformed data series.
    method : str
        The transformation method that was applied.
    transform_params : dict, optional
        Parameters needed for inverse transform (e.g., lambda for yeo_johnson).

    Returns
    -------
    pd.Series
        Data in original scale.
    """
    s = series.copy()
    if s.dropna().size == 0:
        return s

    if method == "log1p":
        return pd.Series(np.expm1(s), index=s.index)
    elif method == "yeo_johnson":
        if transform_params and "lmbda" in transform_params:
            lmbda = transform_params["lmbda"]
            return pd.Series(_inverse_yeo_johnson(s.values, lmbda), index=s.index)
        else:
            logger.warning("Yeo-Johnson inverse requires lambda parameter. Returning original values.")
            return s
    elif method == "none":
        return s
    else:
        raise ValueError(f"Unsupported transform method for inverse: {method}")


def _inverse_yeo_johnson(y, lmbda):
    """Inverse Yeo-Johnson transform."""
    y = np.asarray(y)
    result = np.zeros_like(y, dtype=np.float64)

    pos_mask = y >= 0
    neg_mask = ~pos_mask

    if lmbda == 0:
        result[pos_mask] = np.expm1(y[pos_mask])
    else:
        result[pos_mask] = np.power(y[pos_mask] * lmbda + 1, 1 / lmbda) - 1

    if lmbda == 2:
        result[neg_mask] = -np.expm1(-y[neg_mask])
    else:
        result[neg_mask] = 1 - np.power(-(2 - lmbda) * y[neg_mask] + 1, 1 / (2 - lmbda))

    return result


def _select_best_method(series: pd.Series, candidates: list) -> str:
    """
    Select the transformation method that minimizes skewness for small datasets.

    Parameters
    ----------
    series : pd.Series
        The input data series to be transformed.
    candidates : list
        List of transformation methods to consider.

    Returns
    -------
    str
        The transformation method that results in the lowest skewness.
    """
    if series.dropna().size == 0:
        return "none"

    best_method = "none"
    best_skew = float("inf")

    for method in candidates:
        transformed = _apply_transform(series, method)
        skew = abs(stats.skew(transformed, nan_policy="omit"))
        if skew < best_skew:
            best_skew = skew
            best_method = method

    return best_method


def run_transform(
    df: pd.DataFrame,
    mode: str = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Automated transformation of 'interval_days' column based on skewness.
    methods include [log1p, yeo_johnson, none].
    For small data, it will try multiple methods and select the best one.
    For medium and large data, it will use default methods.
    Supports modes small / medium / large for different data sizes and strategies.


    Parameters
    ----------
    df : pd.DataFrame
        cleaned data, must contain the column 'interval_days'
    mode : str, default='small'
        Data size mode, can be ['small', 'medium', 'large']
    params : dict, optional
        Custom parameters for the transform module, including:
            - method_candidates: list[str] = ["log1p", "yeo_johnson", "none"]
            - auto_select_by_skewness: bool = True
            - skew_threshold: float = 2.0

    Returns
    -------
    transformed_df : pd.DataFrame
        The transformed DataFrame, including the column 'interval_days_transformed'
    transform_meta : dict
        Contains:
            - method: The transformation method used
            - skewness_before: Skewness before transformation
            - skewness_after: Skewness after transformation
    """
    # 0. Initialize config
    if mod_params is None:
        mod_params = {}
    cfg = {**DEFAULT_TRANSFORM_PARAMS, **mod_params}
    logger.info("=== Transform Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    if "interval_days" not in df.columns:
        raise ValueError("Input DataFrame must contain 'interval_days' column.")

    series = df["interval_days"].copy()
    skew_before = stats.skew(series, nan_policy="omit")

    if series.dropna().size == 0:
        transformed_df = df.copy()
        transformed_df["interval_days_transformed"] = series
        return transformed_df, {
            "method": "none",
            "skewness_before": float(skew_before),
            "skewness_after": float(skew_before),
        }

    # 優先檢查 skewness，若已經夠低則不轉換
    if abs(skew_before) < cfg["skew_threshold"]:
        method = "none"
        logger.info(f"Skewness {skew_before:.4f} < threshold {cfg['skew_threshold']}, skipping transform.")
    elif mode in MODE_DEFAULT_METHODS:
        method = MODE_DEFAULT_METHODS[mode]
    elif mode == "small" and cfg["auto_select_by_skewness"]:
        method = _select_best_method(series, cfg["method_candidates"])
    else:
        method = "none"

    # Apply transformation and record parameters if needed
    if method == "yeo_johnson":
        vals = series.fillna(0).values
        transformed, lmbda = stats.yeojohnson(vals)
        transformed_series = pd.Series(transformed, index=series.index)
        transform_params = {"lmbda": float(lmbda)}
    else:
        transformed_series = _apply_transform(series, method)
        transform_params = {}

    skew_after = stats.skew(transformed_series, nan_policy="omit")

    transformed_df = df.copy()
    transformed_df["interval_days_transformed"] = transformed_series

    transform_meta = {
        "method": method,
        "skewness_before": float(skew_before),
        "skewness_after": float(skew_after),
        "transform_params": transform_params,
    }

    return transformed_df, transform_meta
