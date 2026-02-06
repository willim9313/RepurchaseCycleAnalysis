from typing import Tuple, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import duckdb
import logging

logger = logging.getLogger(__name__)

DEFAULT_CLEANING_PARAMS = {
    "remove_negatives": True,
    "missing_strategy": "drop",      # "drop", "impute_mean"
    "outlier_method": "IQR",         # "IQR", "MAD", "QUANTILE"
    "outlier_threshold": 1.5,        # for IQR and MAD
    "quantile_bounds": [0.01, 0.99],  # for QUANTILE method
    "min_group_size_for_stats": 3    # fallback when group too small
}


def _compute_outlier_mask(
    s: pd.Series,
    method: str,
    threshold: float,
    quantile_bounds: Optional[list] = None,
) -> pd.Series:
    """
    Given a series(cat level), return a boolean mask indicating True=keep,
    False=outlier.

    Parameters
    ----------
    s : pd.Series, series of quantity values
    method : str, outlier detection method ("IQR", "MAD", "QUANTILE")
    threshold : float, threshold corresponding to the method

    Returns
    -------
    pd.Series of bool: boolean series with the same index as s, True=keep,
      False=outlier
    """
    method = method.upper()

    if s.empty:
        return pd.Series([False] * 0, index=s.index)

    if method == "IQR":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (s >= lower) & (s <= upper)

    elif method == "MAD":
        median = s.median()
        mad = np.median(np.abs(s - median))
        if mad == 0:
            # if MAD is zero, all values are nearly identical, keep all
            return pd.Series([True] * len(s), index=s.index)
        z = np.abs(s - median) / mad
        return z < threshold

    elif method == "QUANTILE":
        if quantile_bounds is None:
            raise ValueError("quantile_bounds must be provided for QUANTILE method")
        lower, upper = s.quantile(quantile_bounds)
        return (s >= lower) & (s <= upper)

    else:
        raise ValueError(f"Unsupported outlier_method: {method}")


def run_data_cleaning(
    df: pd.DataFrame,
    mode: Literal["small", "medium", "large"] = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean the data by removing erroneous and outlier values in "interval_days".

    Parameters
    ----------
    df : pd.DataFrame
        Original input data, must contain ['uid', 'cat', 'interval_days'].
    mode : str, optional
        Execution strategy (small / medium / large), default is 'small'.
        - small : step-by-step/readable, suitable for <1e4 rows
        - medium: fully vectorized pandas, suitable for ~1e6
        - large : DuckDB/sql path (currently still returns pandas, reserved for large data)
    params : dict, optional
        Module settings, e.g.:
        {
          "remove_negatives": true | false,
          "missing_strategy": "drop" | "impute_mean",
          "outlier_method": "IQR" | "MAD" | "QUANTILE",
          "outlier_threshold": 1.5,
          "min_group_size_for_stats": 3
        }

    Returns
    -------
    cleaned_df : pd.DataFrame
        DataFrame after cleaning.
    discard_summary : dict
        {
          'total_rows': int,
          'removed_negatives': int,
          'removed_missing': int,
          'removed_outliers': int
        }
    """
    # 0. Defaults & validation
    if mod_params is None:
        mod_params = {}
    cfg = {**DEFAULT_CLEANING_PARAMS, **mod_params}

    logger.info("=== Data Cleaning Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    required_cols = {"uid", "cat", "interval_days"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    df = df.copy()
    total_rows = len(df)

    # 1. Remove negatives
    removed_negatives = 0
    if cfg["remove_negatives"]:
        neg_mask = df["interval_days"] < 0
        removed_negatives = int(neg_mask.sum())
        df = df[~neg_mask]

    # 2. Missing handling (per category)
    # record initial missing count, because later we may impute instead of drop
    initial_missing = df["interval_days"].isna().sum()

    if cfg["missing_strategy"] == "drop":
        df = df.dropna(subset=["interval_days"])
        removed_missing = int(initial_missing)

    elif cfg["missing_strategy"] == "impute_mean":
        # per-cat impute; small group fallback global mean
        global_mean = df["interval_days"].mean(skipna=True)

        def impute_group_mean(x: pd.Series) -> pd.Series:
            # x is pd.Series of interval_days for a single cat
            non_na = x.dropna()
            if len(non_na) >= cfg["min_group_size_for_stats"]:
                fill_val = non_na.mean()
            else:
                fill_val = global_mean
            return x.fillna(fill_val)

        df["interval_days"] = (
            df.groupby("cat", group_keys=False)["interval_days"]
              .transform(impute_group_mean)
        )
        # No rows are removed in impute mode
        removed_missing = 0

    else:
        raise ValueError(f"Invalid missing_strategy: {cfg['missing_strategy']}")

    # 3. Outlier detection (per category)
    # Under large mode, we will use DuckDB for efficiency(currently still pandas output)
    # Use duckdb's placeholder for future extension.
    if mode in ["small", "medium"]:
        def per_cat_mask(sub_df: pd.DataFrame) -> pd.Series:
            col = sub_df["interval_days"]
            # if group size is too small, keep all to avoid over-filtering
            if len(col) < cfg["min_group_size_for_stats"]:
                return pd.Series([True] * len(col), index=col.index)
            return _compute_outlier_mask(
                s=col,
                method=cfg["outlier_method"],
                threshold=cfg["outlier_threshold"],
                quantile_bounds=cfg["quantile_bounds"]
            )

        # Initialize mask
        keep_mask = pd.Series([True] * len(df), index=df.index)

        # Calculate and update mask for each category
        for cat_val, sub_df in df.groupby("cat"):
            cat_mask = per_cat_mask(sub_df)
            keep_mask.loc[cat_mask.index] = cat_mask

        removed_outliers = int((~keep_mask).values.sum())
        df = df[keep_mask]

    elif mode == "large":
        # first using pandas to compute per-cat bounds, then duckdb to filter
        # 1. calculate per-cat bounds
        bounds = []

        for cat_val, sub in df.groupby("cat"):
            col = sub["interval_days"].dropna()
            if len(col) < cfg["min_group_size_for_stats"]:
                bounds.append(
                    (cat_val, None, None, None)
                )
                continue

            method = cfg["outlier_method"].upper()
            thr = cfg["outlier_threshold"]

            if method == "IQR":
                q1, q3 = col.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - thr * iqr
                upper = q3 + thr * iqr
            elif method == "MAD":
                med = col.median()
                mad = np.median(np.abs(col - med))
                if mad == 0:
                    lower, upper = -np.inf, np.inf
                else:
                    # z < thr  → |x-med|/mad < thr → |x-med| < thr*mad
                    lower, upper = med - thr * mad, med + thr * mad
            elif method == "QUANTILE":
                # 修正：使用 cfg 中的 quantile_bounds 而非硬編碼
                lower, upper = col.quantile(cfg["quantile_bounds"])
            else:
                raise ValueError(f"Unsupported outlier_method: {method}")

            bounds.append((cat_val, lower, upper, len(col)))

        bounds_df = pd.DataFrame(bounds, columns=["cat", "lower", "upper", "n"])

        # 2. using DuckDB to filter and join
        with duckdb.connect(database=":memory:") as con:
            con.register("input_df", df)
            con.register("bounds_df", bounds_df)

            cleaned_df = con.execute("""
                SELECT i.*
                FROM input_df i
                JOIN bounds_df b
                ON i.cat = b.cat
                WHERE
                b.lower IS NULL
                OR (
                        i.interval_days >= b.lower
                    AND i.interval_days <= b.upper
                )
            """).df()
            removed_outliers = int(len(df) - len(cleaned_df))
            df = cleaned_df
    else:
        raise ValueError(f"Invalid mode: {mode}")

    logger.info(f"Removed {removed_negatives} negative values")
    logger.info(f"Removed {removed_missing} missing values")
    logger.info(f"Removed {removed_outliers} outliers")

    discard_summary = {
        "total_rows": int(total_rows),
        "removed_negatives": int(removed_negatives),
        "removed_missing": int(removed_missing),
        "removed_outliers": int(removed_outliers),
    }

    cleaned_df = df.reset_index(drop=True)
    return cleaned_df, discard_summary
