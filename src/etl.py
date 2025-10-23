import pandas as pd
import numpy as np
from dateutil import parser
from tqdm import tqdm


def build_intervals(
    df: pd.DataFrame,
    min_purchase: int = 2,
    category_col: str | None = "cat"
) -> pd.DataFrame:
    """
    根據每位用戶的購買紀錄，計算連續兩次購買之間的時間間隔。

    Parameters
    ----------
    df : pd.DataFrame
        原始訂單資料，至少需包含:
        - uid : 使用者編號
        - it_name : 商品名稱
        - purchase_date : 購買日期 (datetime 或可轉換格式)
        - cat : 商品分類 (可選)
        - client_type : 用戶類型 (可選, 不參與計算)
    min_purchase : int, default=2
        每位用戶最少需出現的購買次數，否則不納入分析。
    category_col : str | None, default='cat'
        商品分類欄位名稱，若無可設為 None。

    Returns
    -------
    pd.DataFrame
        含以下欄位的 DataFrame:
        - prev_purchase_date : 上次購買日期
        - interval_days : 與上次購買的間隔天數
        - purchase_seq : 該用戶的第幾次購買
        - purchase_month / prev_purchase_month : 月份衍生欄位
    """

    df = df.copy()

    # --- 1. 日期轉換 ---
    if not np.issubdtype(df["purchase_date"].dtype, np.datetime64):
        df["purchase_date"] = df["purchase_date"].apply(parser.parse)

    # --- 2. 檢查必要欄位 ---
    required_cols = {"uid", "it_name", "purchase_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")

    # --- 3. 排序 ---
    sort_cols = ["uid", "purchase_date"]
    if category_col and category_col in df.columns:
        sort_cols.insert(1, category_col)

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 4. 計算間隔 ---
    group_cols = ["uid"]
    if category_col and category_col in df.columns:
        group_cols.append(category_col)

    df["prev_purchase_date"] = df.groupby(group_cols)["purchase_date"].shift(1)
    df["interval_days"] = (
        df["purchase_date"] - df["prev_purchase_date"]
    ).dt.days
    df["purchase_seq"] = df.groupby(group_cols).cumcount() + 1

    # --- 5. 派生月份 ---
    df["purchase_month"] = df["purchase_date"].dt.to_period("M").astype(str)
    df["prev_purchase_month"] = df["prev_purchase_date"].dt.to_period("M").astype(str)

    # --- 6. 過濾購買次數不足的使用者 ---
    user_counts = df.groupby("uid")["purchase_seq"].max()
    valid_users = user_counts[user_counts >= min_purchase].index
    df = df[df["uid"].isin(valid_users)].reset_index(drop=True)

    # --- 7. 結果回傳 ---
    return df
