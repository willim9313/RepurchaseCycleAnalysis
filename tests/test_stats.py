import numpy as np
import pandas as pd
import pytest

from src.stats import summarize_intervals, analyze_distribution


@pytest.fixture
def df_intervals():
    """
    兩個分類的 interval 範例：
    - A: [10, 20, 30]  → n=3, mean=20, median=20, iqr=10
    - B: [5, 5, 5, 5]  → n=4, mean=5,  median=5,  iqr=0
    另外包含 0、負數與 NaN，應被自動忽略。
    """
    return pd.DataFrame(
        {
            "uid":  ["U1", "U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4", "U5"],
            "cat":  ["A",  "A",  "A",  "B",  "B",  "B",  "B",  "A",  "A",  "B"],
            "interval_days": [10, 20, 30, 5, 5, 5, 5, 0, -1, np.nan],  # 0, -1, NaN 應忽略
        }
    )


def test_summarize_intervals_basic(df_intervals):
    out = summarize_intervals(df_intervals, interval_col="interval_days", group_cols=("cat",), min_samples=2)

    # 基本欄位存在
    expected_cols = {"cat", "n", "mean", "median", "std", "iqr", "cv", "skew", "kurtosis"}
    assert expected_cols.issubset(out.columns)

    # 兩個群組都應該保留
    assert set(out["cat"]) == {"A", "B"}

    # A 群組檢查
    row_a = out.loc[out["cat"] == "A"].iloc[0]
    assert row_a["n"] == 3
    assert np.isclose(row_a["mean"], 20.0)
    assert np.isclose(row_a["median"], 20.0)
    assert np.isclose(row_a["iqr"], 10.0)

    # B 群組檢查
    row_b = out.loc[out["cat"] == "B"].iloc[0]
    assert row_b["n"] == 4
    assert np.isclose(row_b["mean"], 5.0)
    assert np.isclose(row_b["median"], 5.0)
    assert np.isclose(row_b["iqr"], 0.0)


def test_summarize_intervals_no_group_returns_single_row(df_intervals):
    out = summarize_intervals(df_intervals, interval_col="interval_days", group_cols=None, min_samples=2)
    # 非群組模式只應有一列
    assert len(out) == 1
    assert out.iloc[0]["n"] == 7  # 10,20,30,5,5,5,5 共 7 筆有效（忽略 0,-1,NaN）


def test_summarize_intervals_respects_min_samples(df_intervals):
    # 設定很高門檻，應該過濾到空結果
    out = summarize_intervals(df_intervals, interval_col="interval_days", group_cols=("cat",), min_samples=10)
    assert out.empty


def test_analyze_distribution_basic_shape_and_types(df_intervals):
    out = analyze_distribution(
        df_intervals,
        interval_col="interval_days",
        group_cols=("cat",),
        shapiro_max_n=5000,
        kde_grid_size=512,       # 降低格點讓測試更快
        peak_prominence=None,    # 不設門檻以避免平台差異
        min_samples=2,
    )

    # 期望欄位
    expected_cols = {
        "cat", "n", "mean", "median", "std", "iqr", "cv", "skew", "kurtosis",
        "shapiro_p", "dip_stat", "dip_p", "n_kde_peaks", "kde_peaks"
    }
    assert expected_cols.issubset(out.columns)

    # 應含兩個群組
    assert set(out["cat"]) == {"A", "B"}

    # 型別與內容合理性
    for _, row in out.iterrows():
        # n_kde_peaks 應為非負整數；kde_peaks 為 list，長度一致
        assert isinstance(row["n_kde_peaks"], (int, np.integer))
        assert row["n_kde_peaks"] >= 1  # 單峰以上（KDE 在這些分佈下至少會有 1 峰）
        assert isinstance(row["kde_peaks"], list)
        assert len(row["kde_peaks"]) == row["n_kde_peaks"]

        # shapiro_p/dip_* 可能為 NaN（樣本數、平臺實作差異），但欄位需存在且為數值型態或 NaN
        assert np.issubdtype(type(row["shapiro_p"]), (float, np.floating)) or pd.isna(row["shapiro_p"])
        assert np.issubdtype(type(row["dip_stat"]), (float, np.floating)) or pd.isna(row["dip_stat"])
        assert np.issubdtype(type(row["dip_p"]), (float, np.floating)) or pd.isna(row["dip_p"])


def test_analyze_distribution_no_group(df_intervals):
    out = analyze_distribution(
        df_intervals,
        interval_col="interval_days",
        group_cols=None,   # 整體一組
        kde_grid_size=256,
        min_samples=2,
    )
    assert len(out) == 1
    # 檢查關鍵欄位存在
    for col in ["n", "mean", "median", "shapiro_p", "n_kde_peaks", "kde_peaks"]:
        assert col in out.columns


def test_analyze_distribution_min_samples_filters_all(df_intervals):
    out = analyze_distribution(
        df_intervals,
        interval_col="interval_days",
        group_cols=("cat",),
        min_samples=10,  # 提高門檻
    )
    assert out.empty
