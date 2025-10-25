"""ETL 模組：處理訂單資料以計算購買間隔時間。"""
import pandas as pd
import numpy as np
from dateutil import parser


def build_intervals(
    df: pd.DataFrame,
    min_purchase: int = 2,
    category_col: str | None = "Category"
) -> pd.DataFrame:
    """
    根據每位用戶的購買紀錄，計算連續兩次購買之間的時間間隔。

    Parameters
    ----------
    df : pd.DataFrame, 原始訂單資料, 至少需包含:
        - UserId : 使用者編號
        - ItemName : 商品名稱
        - OrderDate : 購買日期 (datetime 或可轉換格式)
        - Category : 商品分類 (可選), 如無可以用以下category_col指定
        - ClientType : 用戶類型 (可選, 不參與計算)
    min_purchase : int, default=2
        每位用戶最少需出現的購買次數，否則不納入分析。
    category_col : str | None, default='Category'
        商品分類欄位名稱，若無可設為 None。

    Returns
    -------
    pd.DataFrame
        含以下欄位的 DataFrame:
        - prev_order_date : 上次購買日期
        - interval_days : 與上次購買的間隔天數
        - purchase_seq : 該用戶的第幾次購買
        - purchase_month / prev_purchase_month : 月份衍生欄位
    """

    df = df.copy()

    # --- 1. 日期轉換 ---
    if not np.issubdtype(df["OrderDate"].dtype, np.datetime64):
        df["OrderDate"] = df["OrderDate"].apply(parser.parse)

    # --- 2. 檢查必要欄位 ---
    required_cols = {"UserId", "ItemName", "OrderDate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")

    # --- 3. 排序 ---
    sort_cols = ["UserId", "OrderDate"]
    if category_col and category_col in df.columns:
        sort_cols.insert(1, category_col)

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # --- 4. 計算間隔 ---
    group_cols = ["UserId"]
    if category_col and category_col in df.columns:
        group_cols.append(category_col)

    df["prev_order_date"] = df.groupby(group_cols)["OrderDate"].shift(1)
    df["interval_days"] = (
        df["OrderDate"] - df["prev_order_date"]
    ).dt.days
    df["purchase_seq"] = df.groupby(group_cols).cumcount() + 1

    # --- 5. 派生月份 ---
    df["purchase_month"] = df["OrderDate"].dt.to_period("M").astype(str)
    df["prev_purchase_month"] = df["prev_order_date"].dt.to_period("M").astype(str)

    # --- 6. 過濾購買次數不足的使用者 ---
    user_counts = df.groupby("UserId")["purchase_seq"].max()
    valid_users = user_counts[user_counts >= min_purchase].index
    df = df[df["UserId"].isin(valid_users)].reset_index(drop=True)

    # --- 7. 結果回傳 ---
    return df
