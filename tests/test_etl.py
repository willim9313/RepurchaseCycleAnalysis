import pandas as pd
import pytest
from src.etl import build_intervals


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"uid": "U1", "cat": "Snack", "it_name": "Chips", "purchase_date": "2024-01-01"},
        {"uid": "U1", "cat": "Snack", "it_name": "Soda", "purchase_date": "2024-01-20"},
        {"uid": "U1", "cat": "Snack", "it_name": "Candy", "purchase_date": "2024-02-18"},
        {"uid": "U2", "cat": "Drink", "it_name": "Coffee", "purchase_date": "2024-01-03"},
    ])


def test_build_intervals_basic(sample_df):
    df_out = build_intervals(sample_df, min_purchase=2)

    # 基本欄位檢查
    expected_cols = {
        "uid", "cat", "it_name", "purchase_date",
        "prev_purchase_date", "interval_days",
        "purchase_seq", "purchase_month", "prev_purchase_month"
    }
    assert expected_cols.issubset(df_out.columns)

    # 資料筆數應只剩多於一次購買的使用者
    assert df_out["uid"].nunique() == 1  # U2 應該被排除
    assert all(df_out["uid"] == "U1")

    # 檢查間隔值是否正確
    intervals = df_out["interval_days"].dropna().tolist()
    assert intervals == [19, 29]


def test_missing_columns_should_raise():
    df = pd.DataFrame([{"uid": "U1", "purchase_date": "2024-01-01"}])
    with pytest.raises(ValueError):
        build_intervals(df)


def test_handles_multiple_categories():
    df = pd.DataFrame([
        {"uid": "U1", "cat": "Snack", "it_name": "Chips", "purchase_date": "2024-01-01"},
        {"uid": "U1", "cat": "Drink", "it_name": "Tea", "purchase_date": "2024-01-10"},
        {"uid": "U1", "cat": "Snack", "it_name": "Candy", "purchase_date": "2024-01-20"},
        {"uid": "U1", "cat": "Drink", "it_name": "Juice", "purchase_date": "2024-01-25"},
    ])

    df_out = build_intervals(df)
    # 每個分類都應獨立計算 interval
    assert set(df_out["cat"].unique()) == {"Snack", "Drink"}
    assert not df_out["interval_days"].isna().all()
