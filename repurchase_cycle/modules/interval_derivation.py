'''repurchase_cycle.modules.interval_derivation'''
from typing import Tuple, Dict, Any, Optional, Literal
import re
import pandas as pd
import duckdb
import logging

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_PARAMS = {
    "uid_col": "UserId",
    "cat_col": "Category",
    "date_col": "OrderDate",
    "groupby_cols": ["UserId", "Category"],
    "keep_first_purchase": False,
    "date_format": None,
    "extra_cols": [],
    "min_intervals_per_group": 2
}


def _parse_dates(
    df: pd.DataFrame,
    date_col: str,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse date column to datetime if not already.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    date_format : str, optional

    Returns
    -------
    pd.DataFrame with parsed date column
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
    return df


def sanitize_context(name: str) -> str:
    """
    clean context, remove illegal characters

    Parameters
    ----------
    name : str
        Original name

    Returns
    -------
    str
        Cleaned safe context string
    """
    if not name:
        return name
    # substitute illegal characters
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', name)
    # remove leading and trailing whitespace
    sanitized = sanitized.strip()
    # merge consecutive underscores or whitespace
    sanitized = re.sub(r'[_\s]+', '_', sanitized)
    # 移除首尾底線
    sanitized = sanitized.strip('_')
    return sanitized


def run_interval_calculation(
    df: pd.DataFrame,
    mode: Literal["small", "medium", "large"] = "small",
    general_params: Optional[Dict[str, Any]] = None,
    mod_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Convert transaction-based data to interval-based data.

    For each (uid, cat) group, calculate the time difference (in days)
    between consecutive purchases.

    Parameters
    ----------
    df : pd.DataFrame
        Original transaction data. Must contain uid, cat, and order_date columns
        (column names configurable via params).
    mode : str, optional
        Execution strategy (small / medium / large), default is 'small'.
        - small : step-by-step groupby + shift, suitable for <1e4 rows
        - medium: vectorized pandas sort + groupby transform, suitable for ~1e6
        - large : DuckDB window function (LAG) for large datasets
    general_params : dict, optional
        General pipeline parameters (not used in this module).
    mod_params : dict, optional
        Module settings:
        {
          "uid_col": str,           # user identifier column name
          "cat_col": str,           # category column name
          "date_col": str,          # date column name
          "groupby_cols": list,     # columns to group by for interval calculation
          "keep_first_purchase": bool,  # whether to keep first purchase (interval=NaN)
          "date_format": str|None,  # date format for parsing
          "extra_cols": list,       # additional columns to preserve
          "min_intervals_per_group": int,
        }

    Returns
    -------
    interval_df : pd.DataFrame
        DataFrame with interval_days calculated. Contains:
        - uid, cat, order_date, prev_order_date, interval_days, purchase_seq
        - any extra_cols specified
    conversion_summary : dict
        {
          'total_transactions': int,
          'unique_users': int,
          'unique_categories': int,
          'output_intervals': int,
          'single_purchase_dropped': int,
          'dropped_due_to_insufficient_intervals': int
        }
    """
    # 0. Defaults & validation
    if mod_params is None:
        mod_params = {}
    cfg = {**DEFAULT_INTERVAL_PARAMS, **mod_params}

    logger.info("=== Interval Calculation Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    uid_col = cfg["uid_col"]
    cat_col = cfg["cat_col"]
    date_col = cfg["date_col"]
    groupby_cols = cfg["groupby_cols"]
    keep_first = cfg["keep_first_purchase"]
    extra_cols = cfg["extra_cols"]

    # Validate required columns
    required_cols = {uid_col, cat_col, date_col}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    df = df.copy()
    total_transactions = len(df)
    unique_users = df[uid_col].nunique()
    unique_categories = df[cat_col].nunique()

    # 1. Parse dates
    df = _parse_dates(df, date_col, cfg["date_format"])

    # 2. Determine columns to keep
    output_cols = list(groupby_cols) + [date_col]
    for col in extra_cols:
        if col in df.columns and col not in output_cols:
            output_cols.append(col)

    # 3. Calculate intervals based on mode
    if mode in ["small", "medium"]:
        # Sort by groupby_cols + date
        df = df.sort_values(groupby_cols + [date_col]).reset_index(drop=True)

        # Calculate previous order date using shift within groups
        df["prev_order_date"] = df.groupby(groupby_cols)[date_col].shift(1)

        # Calculate interval in days
        df["interval_days"] = (df[date_col] - df["prev_order_date"]).dt.total_seconds() / 86400

        # Calculate purchase sequence number
        df["purchase_seq"] = df.groupby(groupby_cols).cumcount() + 1

    elif mode == "large":
        # Use DuckDB for large datasets
        with duckdb.connect(database=":memory:") as con:
            con.register("transactions", df)

            # Build groupby clause
            groupby_clause = ", ".join(groupby_cols)
            orderby_clause = ", ".join(groupby_cols + [date_col])

            query = f"""
                SELECT *,
                    LAG({date_col}) OVER (
                        PARTITION BY {groupby_clause}
                        ORDER BY {date_col}
                    ) AS prev_order_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY {groupby_clause}
                        ORDER BY {date_col}
                    ) AS purchase_seq
                FROM transactions
                ORDER BY {orderby_clause}
            """
            df = con.execute(query).df()

        # Calculate interval_days from prev_order_date
        df["interval_days"] = (
            pd.to_datetime(df[date_col]) - pd.to_datetime(df["prev_order_date"])
        ).dt.total_seconds() / 86400

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # 4. Handle first purchases (where prev_order_date is NaN)
    first_purchase_mask = df["prev_order_date"].isna()
    single_purchase_dropped = 0

    if not keep_first:
        single_purchase_dropped = int(first_purchase_mask.sum())
        df = df[~first_purchase_mask]

    # remove groups with insufficient intervals
    min_intervals = cfg.get("min_intervals_per_group", 2)
    dropped_due_to_intervals = 0  # 初始化變數
    if min_intervals > 1:
        group_sizes = df.groupby(groupby_cols).size()
        valid_groups = group_sizes[group_sizes >= min_intervals].index
        before_filter = len(df)
        df = df.set_index(groupby_cols).loc[valid_groups].reset_index()
        dropped_due_to_intervals = before_filter - len(df)

    # 5. Rename columns to standard names
    df = df.rename(columns={
        uid_col: "uid",
        cat_col: "cat",
        date_col: "order_date",
    })

    # 6. Select output columns
    final_cols = ["uid", "cat", "order_date", "prev_order_date", "interval_days", "purchase_seq"]
    for col in extra_cols:
        if col in df.columns and col not in final_cols:
            final_cols.append(col)

    df = df[[c for c in final_cols if c in df.columns]]

    logger.info(f"Total transactions: {total_transactions}")
    logger.info(f"Unique users: {unique_users}")
    logger.info(f"Unique categories: {unique_categories}")
    logger.info(f"Output intervals: {len(df)}")
    logger.info(f"Single purchase dropped: {single_purchase_dropped}")
    logger.info(f"Dropped due to insufficient intervals: {dropped_due_to_intervals}")
    logger.info(f"Final columns: {df.columns.tolist()}")

    conversion_summary = {
        "total_transactions": int(total_transactions),
        "unique_users": int(unique_users),
        "unique_categories": int(unique_categories),
        "output_intervals": int(len(df)),
        "single_purchase_dropped": int(single_purchase_dropped),
        "dropped_due_to_insufficient_intervals": int(dropped_due_to_intervals)
    }

    interval_df = df.reset_index(drop=True)
    return interval_df, conversion_summary
