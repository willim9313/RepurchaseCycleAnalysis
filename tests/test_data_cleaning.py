# ======================================================
# test_data_cleaning.py
# ======================================================
# 單元測試模組: 資料清理與檢查（分類感知版）
# 對應模組: run_data_cleaning()
# ------------------------------------------------------

import pytest
import pandas as pd
import numpy as np
from repurchase_cycle.modules.data_cleaning import run_data_cleaning, _compute_outlier_mask


# ======================================================
# Fixture: 測試資料生成
# ======================================================
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """標準測試資料：包含負值、缺值、離群值"""
    np.random.seed(42)
    data = {
        "uid": [f"U{i}" for i in range(20)],
        "cat": ["A"] * 10 + ["B"] * 10,
        "interval_days": np.concatenate([
            np.random.normal(30, 5, 8),      # A群正常值
            [-5, np.nan],                     # A群負值與缺值
            np.random.normal(100, 10, 8),     # B群正常值
            [999, 1500]                       # B群極端值
        ])
    }
    return pd.DataFrame(data)


@pytest.fixture
def small_group_df() -> pd.DataFrame:
    """小群組測試資料：用於測試 min_group_size_for_stats fallback"""
    return pd.DataFrame({
        "uid": ["A1", "A2", "B1", "B2", "B3", "B4", "B5"],
        "cat": ["A", "A", "B", "B", "B", "B", "B"],
        "interval_days": [10.0, 500.0, 100.0, 105.0, 110.0, 95.0, 102.0]
    })


@pytest.fixture
def impute_test_df() -> pd.DataFrame:
    """補值策略測試資料：每個分類有明顯不同均值"""
    return pd.DataFrame({
        "uid": ["A1", "A2", "A3", "B1", "B2", "B3"],
        "cat": ["A", "A", "A", "B", "B", "B"],
        "interval_days": [10.0, 12.0, np.nan, 100.0, 110.0, np.nan]
    })


# ======================================================
# Helper: 檢查 discard_summary 結構
# ======================================================
def assert_summary_structure(summary: dict):
    """驗證 summary 字典包含所有必要欄位且類型正確"""
    expected_keys = {
        "total_rows",
        "removed_negatives",
        "removed_missing",
        "removed_outliers"
    }
    assert set(summary.keys()) == expected_keys, f"Keys mismatch: {summary.keys()}"
    for k in expected_keys:
        assert isinstance(summary[k], int), f"{k} should be int, got {type(summary[k])}"


# ======================================================
# Test 1: 基本行為檢查
# ======================================================
class TestBasicBehavior:
    """基本清理功能測試"""

    def test_returns_correct_types(self, sample_df):
        """確認回傳類型正確"""
        cleaned_df, summary = run_data_cleaning(sample_df, mode="small")
        assert isinstance(cleaned_df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert_summary_structure(summary)

    def test_negative_values_removed(self, sample_df):
        """確認負值被移除"""
        cleaned_df, summary = run_data_cleaning(sample_df, mode="small")
        assert (cleaned_df["interval_days"] < 0).sum() == 0
        assert summary["removed_negatives"] >= 1

    def test_missing_values_dropped_by_default(self, sample_df):
        """確認缺值預設被刪除"""
        cleaned_df, summary = run_data_cleaning(sample_df, mode="small")
        assert cleaned_df["interval_days"].isna().sum() == 0
        assert summary["removed_missing"] >= 1

    def test_total_rows_less_after_cleaning(self, sample_df):
        """確認清理後資料筆數減少"""
        cleaned_df, summary = run_data_cleaning(sample_df, mode="small")
        assert len(cleaned_df) < len(sample_df)
        assert summary["total_rows"] == len(sample_df)

    def test_columns_preserved(self, sample_df):
        """確認欄位結構保持不變"""
        cleaned_df, _ = run_data_cleaning(sample_df, mode="small")
        assert set(cleaned_df.columns) == set(sample_df.columns)


# ======================================================
# Test 2: 缺值處理策略
# ======================================================
class TestMissingStrategy:
    """缺值處理策略測試"""

    def test_drop_strategy(self, impute_test_df):
        """測試 drop 策略"""
        params = {"missing_strategy": "drop"}
        cleaned_df, summary = run_data_cleaning(impute_test_df, mod_params=params)
        
        assert cleaned_df["interval_days"].isna().sum() == 0
        assert summary["removed_missing"] == 2  # 兩個 NaN 被移除
        assert len(cleaned_df) == 4

    def test_impute_mean_per_category(self, impute_test_df):
        """測試 impute_mean 策略：應按分類補值（明確設定 min_group_size_for_stats）"""
        params = {"missing_strategy": "impute_mean", "min_group_size_for_stats": 2}
        cleaned_df, summary = run_data_cleaning(impute_test_df, mod_params=params)

        assert_summary_structure(summary)
        assert cleaned_df["interval_days"].isna().sum() == 0
        assert summary["removed_missing"] == 0  # 補值模式不移除

        # A 群 NaN (uid == "A3") 應被補為 (10 + 12) / 2 = 11
        a_imputed = cleaned_df.loc[cleaned_df["uid"] == "A3", "interval_days"].iloc[0]
        assert a_imputed == pytest.approx((10.0 + 12.0) / 2)

        # B 群 NaN (uid == "B3") 應被補為 (100 + 110) / 2 = 105
        b_imputed = cleaned_df.loc[cleaned_df["uid"] == "B3", "interval_days"].iloc[0]
        assert b_imputed == pytest.approx((100.0 + 110.0) / 2)

    def test_impute_fallback_to_global_mean(self):
        """測試小群組 fallback 到全域平均"""
        df = pd.DataFrame({
            "uid": ["A1", "A2", "B1", "B2", "B3", "B4"],
            "cat": ["A", "A", "B", "B", "B", "B"],
            "interval_days": [np.nan, np.nan, 100.0, 100.0, 100.0, 100.0]
        })
        params = {
            "missing_strategy": "impute_mean",
            "min_group_size_for_stats": 3  # A群只有2筆，應 fallback
        }
        cleaned_df, _ = run_data_cleaning(df, mod_params=params)
        
        # A群應使用全域平均 (100.0)
        a_vals = cleaned_df.loc[cleaned_df["cat"] == "A", "interval_days"]
        assert all(v == 100.0 for v in a_vals)


# ======================================================
# Test 3: 離群值偵測方法
# ======================================================
class TestOutlierDetection:
    """離群值偵測測試"""

    def test_iqr_per_category(self):
        """測試 IQR 方法按分類偵測"""
        df = pd.DataFrame({
            "uid": [f"U{i}" for i in range(8)],
            "cat": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "interval_days": [
                10.0, 12.0, 11.0, 120.0,   # A群：120 應該被判為離群
                100.0, 110.0, 105.0, 120.0  # B群：120 在合理範圍
            ]
        })
        params = {"outlier_method": "IQR", "outlier_threshold": 1.5}
        cleaned_df, summary = run_data_cleaning(df, mod_params=params)
        
        assert_summary_structure(summary)
        
        # A群離群值應被移除
        a_vals = cleaned_df.loc[cleaned_df["cat"] == "A", "interval_days"].tolist()
        assert 120.0 not in a_vals, "A群的 120 應被移除"
        
        # B群 120 應保留
        b_vals = cleaned_df.loc[cleaned_df["cat"] == "B", "interval_days"].tolist()
        assert 120.0 in b_vals, "B群的 120 應保留"
        
        assert summary["removed_outliers"] >= 1

    def test_mad_method(self, sample_df):
        """測試 MAD 方法"""
        params = {"outlier_method": "MAD", "outlier_threshold": 3.0}
        cleaned_df, summary = run_data_cleaning(sample_df, mod_params=params)
        
        assert_summary_structure(summary)
        assert len(cleaned_df) > 0

    def test_quantile_method(self, sample_df):
        """測試 QUANTILE 方法"""
        params = {
            "outlier_method": "QUANTILE",
            "quantile_bounds": [0.05, 0.95]
        }
        cleaned_df, summary = run_data_cleaning(sample_df, mod_params=params)
        
        assert_summary_structure(summary)
        assert len(cleaned_df) > 0

    def test_small_group_keeps_all(self, small_group_df):
        """測試小群組保留所有值（避免過度過濾）"""
        params = {"min_group_size_for_stats": 3}
        cleaned_df, summary = run_data_cleaning(small_group_df, mod_params=params)
        
        # A群只有2筆，應全部保留（包含可能的離群值）
        a_vals = cleaned_df.loc[cleaned_df["cat"] == "A", "interval_days"]
        assert len(a_vals) == 2


# ======================================================
# Test 4: 多種 outlier 方法參數化測試
# ======================================================
@pytest.mark.parametrize("method", ["IQR", "MAD", "QUANTILE"])
def test_outlier_methods_parametrized(sample_df, method):
    """參數化測試各種離群值偵測方法"""
    params = {"outlier_method": method}
    if method == "QUANTILE":
        params["quantile_bounds"] = [0.01, 0.99]
    
    cleaned_df, summary = run_data_cleaning(sample_df, mod_params=params)
    assert_summary_structure(summary)
    assert len(cleaned_df) > 0


# ======================================================
# Test 5: 模式切換測試
# ======================================================
@pytest.mark.parametrize("mode", ["small", "medium", "large"])
def test_mode_switching(sample_df, mode):
    """測試不同執行模式"""
    cleaned_df, summary = run_data_cleaning(sample_df, mode=mode)
    assert_summary_structure(summary)
    assert len(cleaned_df) > 0
    # 各模式結果應一致（或接近）
    assert (cleaned_df["interval_days"] < 0).sum() == 0


class TestLargeMode:
    """Large 模式（DuckDB）專項測試"""

    def test_large_mode_iqr(self, sample_df):
        """Large 模式 IQR 測試"""
        params = {"outlier_method": "IQR"}
        cleaned_df, summary = run_data_cleaning(sample_df, mode="large", mod_params=params)
        assert_summary_structure(summary)
        assert len(cleaned_df) > 0

    def test_large_mode_mad(self, sample_df):
        """Large 模式 MAD 測試"""
        params = {"outlier_method": "MAD", "outlier_threshold": 3.0}
        cleaned_df, summary = run_data_cleaning(sample_df, mode="large", mod_params=params)
        assert_summary_structure(summary)

    def test_large_mode_quantile(self, sample_df):
        """Large 模式 QUANTILE 測試"""
        params = {"outlier_method": "QUANTILE", "quantile_bounds": [0.05, 0.95]}
        cleaned_df, summary = run_data_cleaning(sample_df, mode="large", mod_params=params)
        assert_summary_structure(summary)

    def test_large_mode_quantile_custom_bounds(self):
        """測試 Large 模式 QUANTILE 是否正確使用自訂 quantile_bounds"""
        # 建立有明確邊界的資料
        df = pd.DataFrame({
            "uid": [f"U{i}" for i in range(10)],
            "cat": ["A"] * 10,
            "interval_days": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        
        # 使用較寬鬆的 bounds
        wide_params = {"outlier_method": "QUANTILE", "quantile_bounds": [0.0, 1.0]}
        wide_df, wide_summary = run_data_cleaning(df, mode="large", mod_params=wide_params)
        
        # 使用較嚴格的 bounds
        strict_params = {"outlier_method": "QUANTILE", "quantile_bounds": [0.2, 0.8]}
        strict_df, strict_summary = run_data_cleaning(df, mode="large", mod_params=strict_params)
        
        # 嚴格 bounds 應移除更多資料
        assert len(wide_df) >= len(strict_df)


# ======================================================
# Test 6: 參數覆蓋測試
# ======================================================
class TestParameterOverride:
    """參數覆蓋測試"""

    def test_disable_negative_removal(self):
        """測試關閉負值移除（使用確定性資料）"""
        params = {"remove_negatives": False}
        # 明確包含負值，避免依賴 fixture 變動造成的波動
        df = pd.DataFrame({
            "uid": ["U1", "U2", "U3"],
            "cat": ["A", "A", "B"],
            "interval_days": [-5.0, 30.0, 100.0]
        })

        cleaned_df, summary = run_data_cleaning(df, mod_params=params)

        assert summary["removed_negatives"] == 0
        # 確認負值仍被保留
        assert (cleaned_df["interval_days"] < 0).sum() >= 1

    def test_custom_threshold(self, sample_df):
        """測試自訂閾值"""
        # 較寬鬆的閾值應移除較少離群值
        loose_params = {"outlier_threshold": 3.0}
        strict_params = {"outlier_threshold": 1.0}
        
        loose_df, loose_summary = run_data_cleaning(sample_df, mod_params=loose_params)
        strict_df, strict_summary = run_data_cleaning(sample_df, mod_params=strict_params)
        
        assert loose_summary["removed_outliers"] <= strict_summary["removed_outliers"]

    def test_custom_quantile_bounds(self, sample_df):
        """測試自訂分位數邊界"""
        params = {
            "outlier_method": "QUANTILE",
            "quantile_bounds": [0.10, 0.90]
        }
        cleaned_df, summary = run_data_cleaning(sample_df, mod_params=params)
        assert_summary_structure(summary)


# ======================================================
# Test 7: 錯誤輸入檢查
# ======================================================
class TestErrorHandling:
    """錯誤處理測試"""

    def test_missing_required_columns(self):
        """測試缺少必要欄位"""
        df = pd.DataFrame({"uid": ["U1"], "interval_days": [10.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            run_data_cleaning(df)

    def test_invalid_missing_strategy(self, sample_df):
        """測試無效的缺值策略"""
        params = {"missing_strategy": "invalid_strategy"}
        with pytest.raises(ValueError, match="Invalid missing_strategy"):
            run_data_cleaning(sample_df, mod_params=params)

    def test_invalid_outlier_method(self, sample_df):
        """測試無效的離群值方法"""
        params = {"outlier_method": "INVALID"}
        with pytest.raises(ValueError, match="Unsupported outlier_method"):
            run_data_cleaning(sample_df, mod_params=params)

    def test_invalid_mode(self, sample_df):
        """測試無效的執行模式"""
        with pytest.raises(ValueError, match="Invalid mode"):
            run_data_cleaning(sample_df, mode="invalid_mode")

    def test_quantile_without_bounds(self, sample_df):
        """測試 QUANTILE 方法未提供 bounds"""
        # 直接測試 _compute_outlier_mask
        s = sample_df["interval_days"].dropna()
        with pytest.raises(ValueError, match="quantile_bounds must be provided"):
            _compute_outlier_mask(s, method="QUANTILE", threshold=1.5, quantile_bounds=None)


# ======================================================
# Test 8: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_empty_dataframe(self):
        """測試空資料框"""
        df = pd.DataFrame({"uid": [], "cat": [], "interval_days": []})
        cleaned_df, summary = run_data_cleaning(df)
        
        assert len(cleaned_df) == 0
        assert summary["total_rows"] == 0

    def test_all_missing_values(self):
        """測試全部缺值"""
        df = pd.DataFrame({
            "uid": ["U1", "U2", "U3"],
            "cat": ["A", "A", "A"],
            "interval_days": [np.nan, np.nan, np.nan]
        })
        cleaned_df, summary = run_data_cleaning(df, mod_params={"missing_strategy": "drop"})
        
        assert len(cleaned_df) == 0
        assert summary["removed_missing"] == 3

    def test_all_negative_values(self):
        """測試全部負值"""
        df = pd.DataFrame({
            "uid": ["U1", "U2", "U3"],
            "cat": ["A", "A", "A"],
            "interval_days": [-1.0, -2.0, -3.0]
        })
        cleaned_df, summary = run_data_cleaning(df)
        
        assert len(cleaned_df) == 0
        assert summary["removed_negatives"] == 3

    def test_single_category(self, sample_df):
        """測試單一分類"""
        df = sample_df[sample_df["cat"] == "A"].copy()
        cleaned_df, summary = run_data_cleaning(df)
        
        assert_summary_structure(summary)
        assert len(cleaned_df) > 0

    def test_identical_values_mad(self):
        """測試 MAD 方法遇到相同值（MAD=0）"""
        df = pd.DataFrame({
            "uid": ["U1", "U2", "U3", "U4"],
            "cat": ["A", "A", "A", "A"],
            "interval_days": [10.0, 10.0, 10.0, 10.0]
        })
        params = {"outlier_method": "MAD"}
        cleaned_df, summary = run_data_cleaning(df, mod_params=params)
        
        # MAD=0 時應保留所有值
        assert len(cleaned_df) == 4
        assert summary["removed_outliers"] == 0


# ======================================================
# Test 9: _compute_outlier_mask 單元測試
# ======================================================
class TestComputeOutlierMask:
    """離群值遮罩計算函數單元測試"""

    def test_iqr_mask(self):
        """測試 IQR 遮罩"""
        s = pd.Series([1, 2, 3, 4, 5, 100])
        mask = _compute_outlier_mask(s, method="IQR", threshold=1.5)
        
        assert mask.iloc[-1] == False  # 100 應被標記為離群
        assert mask.iloc[0] == True    # 1 應保留

    def test_mad_mask(self):
        """測試 MAD 遮罩"""
        s = pd.Series([1, 2, 3, 4, 5, 100])
        mask = _compute_outlier_mask(s, method="MAD", threshold=3.0)
        
        assert isinstance(mask, pd.Series)
        assert len(mask) == len(s)

    def test_quantile_mask(self):
        """測試 QUANTILE 遮罩"""
        s = pd.Series(range(100))
        mask = _compute_outlier_mask(s, method="QUANTILE", threshold=1.5, quantile_bounds=[0.05, 0.95])
        
        # 邊緣值應被過濾
        assert mask.sum() < len(s)

    def test_empty_series(self):
        """測試空序列"""
        s = pd.Series([], dtype=float)
        mask = _compute_outlier_mask(s, method="IQR", threshold=1.5)
        
        assert len(mask) == 0
