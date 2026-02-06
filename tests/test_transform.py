import pytest
import pandas as pd
import numpy as np
from scipy import stats

from repurchase_cycle.modules.transform import (
    run_transform,
    _apply_transform,
    _select_best_method,
    _inverse_transform,  # Add this import
    DEFAULT_TRANSFORM_PARAMS,
    MODE_DEFAULT_METHODS,
)


# ======================================================
# Fixtures
# ======================================================
@pytest.fixture
def df_highly_skewed() -> pd.DataFrame:
    """高度正偏態資料（指數分布）"""
    np.random.seed(42)
    data = np.random.exponential(scale=10, size=500)
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def df_symmetric() -> pd.DataFrame:
    """接近對稱的資料（常態分布）"""
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=500)
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def df_with_zeros() -> pd.DataFrame:
    """包含零值的資料"""
    np.random.seed(42)
    data = np.concatenate([
        np.zeros(50),
        np.random.exponential(scale=5, size=450)
    ])
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def df_with_negatives() -> pd.DataFrame:
    """包含負值的資料"""
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=5, size=500)
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def df_large() -> pd.DataFrame:
    """大量資料"""
    np.random.seed(42)
    data = np.random.exponential(scale=10, size=100000)
    return pd.DataFrame({"interval_days": data})


# ======================================================
# Test 1: _apply_transform 單元測試
# ======================================================
class TestApplyTransform:
    """_apply_transform 函數測試"""

    def test_log1p_transform(self):
        """log1p 轉換測試"""
        series = pd.Series([0, 1, 10, 100])
        result = _apply_transform(series, "log1p")
        expected = np.log1p(series)
        pd.testing.assert_series_equal(result, expected)

    def test_log1p_clips_negative(self):
        """log1p 應將負值裁剪為 0"""
        series = pd.Series([-5, -1, 0, 1, 10])
        result = _apply_transform(series, "log1p")
        # 負值被裁剪為 0，log1p(0) = 0
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0

    def test_yeo_johnson_transform(self):
        """Yeo-Johnson 轉換測試"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = _apply_transform(series, "yeo_johnson")
        assert len(result) == len(series)
        # Yeo-Johnson 應產生有效數值
        assert not result.isna().any()

    def test_yeo_johnson_handles_negatives(self):
        """Yeo-Johnson 應能處理負值"""
        series = pd.Series([-5, -1, 0, 1, 5])
        result = _apply_transform(series, "yeo_johnson")
        assert len(result) == len(series)
        assert not result.isna().any()

    def test_none_transform(self):
        """none 轉換應返回原始資料"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = _apply_transform(series, "none")
        pd.testing.assert_series_equal(result, series)

    def test_invalid_method_raises(self):
        """無效方法應拋出 ValueError"""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unsupported transform method"):
            _apply_transform(series, "invalid_method")


# ======================================================
# Test 2: _select_best_method 單元測試
# ======================================================
class TestSelectBestMethod:
    """_select_best_method 函數測試"""

    def test_selects_method_with_lowest_skew(self, df_highly_skewed):
        """應選擇偏態最低的方法"""
        series = df_highly_skewed["interval_days"]
        candidates = ["log1p", "yeo_johnson", "none"]
        best = _select_best_method(series, candidates)
        
        # 對於高度偏態資料，log1p 或 yeo_johnson 應優於 none
        assert best in ["log1p", "yeo_johnson"]

    def test_selects_none_for_symmetric(self, df_symmetric):
        """對稱資料可能選擇 none"""
        series = df_symmetric["interval_days"]
        candidates = ["log1p", "yeo_johnson", "none"]
        best = _select_best_method(series, candidates)
        
        # 結果應為三者之一
        assert best in candidates


# ======================================================
# Test 3: run_transform 基本功能測試
# ======================================================
class TestRunTransformBasic:
    """run_transform 基本功能測試"""

    def test_returns_tuple(self, df_highly_skewed):
        """應返回 (DataFrame, dict) tuple"""
        out_df, meta = run_transform(df_highly_skewed)
        assert isinstance(out_df, pd.DataFrame)
        assert isinstance(meta, dict)

    def test_output_contains_transformed_column(self, df_highly_skewed):
        """輸出應包含 interval_days_transformed 欄位"""
        out_df, _ = run_transform(df_highly_skewed)
        assert "interval_days_transformed" in out_df.columns

    def test_preserves_original_column(self, df_highly_skewed):
        """應保留原始 interval_days 欄位"""
        out_df, _ = run_transform(df_highly_skewed)
        assert "interval_days" in out_df.columns
        pd.testing.assert_series_equal(
            out_df["interval_days"],
            df_highly_skewed["interval_days"],
            check_names=False
        )

    def test_meta_structure(self, df_highly_skewed):
        """meta 結構驗證"""
        _, meta = run_transform(df_highly_skewed)
        expected_keys = {"method", "skewness_before", "skewness_after", "transform_params"}
        assert set(meta.keys()) == expected_keys
        assert isinstance(meta["method"], str)
        assert isinstance(meta["skewness_before"], float)
        assert isinstance(meta["skewness_after"], float)
        assert isinstance(meta["transform_params"], dict)


# ======================================================
# Test 4: 偏態降低測試
# ======================================================
class TestSkewReduction:
    """偏態降低測試"""

    def test_skew_reduction_small(self, df_highly_skewed):
        """small 模式應能降低偏態"""
        out_df, meta = run_transform(df_highly_skewed, mode="small")
        before = meta["skewness_before"]
        after = meta["skewness_after"]
        
        # 高度偏態資料轉換後偏態應降低
        assert abs(after) <= abs(before), \
            f"Skewness not reduced: before={before}, after={after}"

    def test_skew_reduction_medium(self, df_highly_skewed):
        """medium 模式應能降低偏態"""
        out_df, meta = run_transform(df_highly_skewed, mode="medium")
        before = meta["skewness_before"]
        after = meta["skewness_after"]
        
        assert abs(after) <= abs(before)

    def test_skew_reduction_large(self, df_highly_skewed):
        """large 模式應能降低偏態"""
        out_df, meta = run_transform(df_highly_skewed, mode="large")
        before = meta["skewness_before"]
        after = meta["skewness_after"]
        
        assert abs(after) <= abs(before)


# ======================================================
# Test 5: 模式切換測試
# ======================================================
class TestModeSwitch:
    """模式切換測試"""

    def test_mode_medium_uses_log1p(self, df_highly_skewed):
        """medium 模式預設使用 log1p"""
        _, meta = run_transform(df_highly_skewed, mode="medium")
        assert meta["method"] == "log1p"

    def test_mode_large_uses_yeo_johnson(self, df_highly_skewed):
        """large 模式預設使用 yeo_johnson"""
        _, meta = run_transform(df_highly_skewed, mode="large")
        assert meta["method"] == "yeo_johnson"

    def test_mode_small_auto_selects(self, df_highly_skewed):
        """small 模式自動選擇最佳方法"""
        _, meta = run_transform(df_highly_skewed, mode="small")
        assert meta["method"] in ["log1p", "yeo_johnson", "none"]

    def test_mode_defaults_defined(self):
        """模式預設值應定義正確"""
        assert MODE_DEFAULT_METHODS["medium"] == "log1p"
        assert MODE_DEFAULT_METHODS["large"] == "yeo_johnson"


# ======================================================
# Test 6: skew_threshold 測試
# ======================================================
class TestSkewThreshold:
    """skew_threshold 參數測試"""

    def test_no_transform_when_below_threshold(self, df_symmetric):
        """偏態低於閾值時不應轉換"""
        out_df, meta = run_transform(
            df_symmetric,
            mode="small",
            mod_params={"skew_threshold": 5.0}  # 設定高閾值
        )
        assert meta["method"] == "none"
        pd.testing.assert_series_equal(
            out_df["interval_days"],
            out_df["interval_days_transformed"],
            check_names=False
        )

    def test_transform_when_above_threshold(self, df_highly_skewed):
        """偏態高於閾值時應轉換"""
        out_df, meta = run_transform(
            df_highly_skewed,
            mode="small",
            mod_params={"skew_threshold": 0.1}  # 設定低閾值
        )
        assert meta["method"] in ["log1p", "yeo_johnson"]

    def test_custom_threshold(self, df_highly_skewed):
        """自訂閾值"""
        # 計算原始偏態
        original_skew = abs(stats.skew(df_highly_skewed["interval_days"]))
        
        # 設定比原始偏態更高的閾值
        _, meta = run_transform(
            df_highly_skewed,
            mode="small",
            mod_params={"skew_threshold": original_skew + 1.0}
        )
        assert meta["method"] == "none"


# ======================================================
# Test 7: 參數覆蓋測試
# ======================================================
class TestParameterOverride:
    """參數覆蓋測試"""

    def test_override_method_candidates(self, df_highly_skewed):
        """覆蓋 method_candidates"""
        _, meta = run_transform(
            df_highly_skewed,
            mode="small",
            mod_params={"method_candidates": ["none"]}
        )
        # 只有 none 可選時，應選擇 none
        assert meta["method"] == "none"

    def test_override_auto_select(self, df_highly_skewed):
        """關閉自動選擇"""
        _, meta = run_transform(
            df_highly_skewed,
            mode="small",
            mod_params={"auto_select_by_skewness": False}
        )
        # 關閉自動選擇且無預設方法時，應為 none
        assert meta["method"] == "none"


# ======================================================
# Test 8: 可逆性測試
# ======================================================
class TestReversibility:
    """轉換可逆性測試"""

    def test_log1p_reversible(self, df_highly_skewed):
        """log1p 轉換應可逆"""
        out_df, meta = run_transform(df_highly_skewed, mode="medium")
        
        if meta["method"] == "log1p":
            transformed = out_df["interval_days_transformed"]
            restored = np.expm1(transformed)
            
            # 反轉後應與原始接近
            original = df_highly_skewed["interval_days"]
            np.testing.assert_array_almost_equal(
                restored.values,
                original.values,
                decimal=10
            )

    def test_reversibility_approximate(self, df_highly_skewed):
        """轉換後反轉應與原始量級接近"""
        out_df, meta = run_transform(df_highly_skewed, mode="medium")
        method = meta["method"]
        transformed = out_df["interval_days_transformed"]
        
        if method == "log1p":
            restored = np.expm1(transformed)
        elif method == "yeo_johnson":
            # Yeo-Johnson 無封閉反函數，檢查值域合理即可
            restored = transformed
        else:
            restored = transformed
        
        # 驗證反轉後的數值分佈與原始大致相符
        original_mean = df_highly_skewed["interval_days"].mean()
        restored_mean = restored.mean()
        
        if method != "yeo_johnson":
            diff_ratio = abs(restored_mean - original_mean) / original_mean
            assert diff_ratio < 0.2, f"Reversibility check failed: diff_ratio={diff_ratio:.3f}"


# ======================================================
# Test 9: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_missing_column_raises_error(self):
        """缺少 interval_days 應報錯"""
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="interval_days"):
            run_transform(df)

    def test_with_zeros(self, df_with_zeros):
        """含零值資料應能處理"""
        out_df, meta = run_transform(df_with_zeros)
        assert "interval_days_transformed" in out_df.columns
        assert not out_df["interval_days_transformed"].isna().any()

    def test_with_negatives_yeo_johnson(self, df_with_negatives):
        """含負值資料使用 yeo_johnson 應能處理"""
        # 設定低閾值以強制進行轉換
        out_df, meta = run_transform(
            df_with_negatives, 
            mode="large",
            mod_params={"skew_threshold": 0.0}
        )
        assert meta["method"] == "yeo_johnson"
        assert not out_df["interval_days_transformed"].isna().any()

    def test_with_nan_values(self):
        """含 NaN 值應能處理"""
        df = pd.DataFrame({
            "interval_days": [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        out_df, meta = run_transform(df)
        # NaN 應在轉換時被適當處理
        assert "interval_days_transformed" in out_df.columns

    def test_single_value(self):
        """單一值應能處理"""
        df = pd.DataFrame({"interval_days": [10.0]})
        out_df, meta = run_transform(df)
        assert "interval_days_transformed" in out_df.columns

    def test_constant_values(self):
        """常數值應能處理"""
        df = pd.DataFrame({"interval_days": [5.0] * 100})
        out_df, meta = run_transform(df)
        # 常數值偏態為 0，應跳過轉換
        assert meta["method"] == "none"

    def test_empty_dataframe(self):
        """空資料框應能處理"""
        df = pd.DataFrame({"interval_days": []})
        out_df, meta = run_transform(df)
        assert len(out_df) == 0


# ======================================================
# Test 10: 大資料測試
# ======================================================
class TestLargeDataset:
    """大資料測試"""

    def test_large_dataset_performance(self, df_large):
        """大資料集應能正常處理"""
        out_df, meta = run_transform(df_large, mode="large")
        assert meta["method"] == "yeo_johnson"
        assert len(out_df) == len(df_large)

    def test_large_dataset_preserves_data(self, df_large):
        """大資料集應保留所有資料"""
        out_df, meta = run_transform(df_large, mode="large")
        pd.testing.assert_series_equal(
            out_df["interval_days"],
            df_large["interval_days"],
            check_names=False
        )


# ======================================================
# Test 11: 預設參數測試
# ======================================================
class TestDefaultParams:
    """預設參數測試"""

    def test_default_params_exist(self):
        """預設參數應存在"""
        assert "method_candidates" in DEFAULT_TRANSFORM_PARAMS
        assert "auto_select_by_skewness" in DEFAULT_TRANSFORM_PARAMS
        assert "skew_threshold" in DEFAULT_TRANSFORM_PARAMS

    def test_default_method_candidates(self):
        """預設方法候選應正確"""
        assert "log1p" in DEFAULT_TRANSFORM_PARAMS["method_candidates"]
        assert "yeo_johnson" in DEFAULT_TRANSFORM_PARAMS["method_candidates"]
        assert "none" in DEFAULT_TRANSFORM_PARAMS["method_candidates"]


# ======================================================
# Test 12: Inverse Transform 測試
# ======================================================
class TestInverseTransform:
    """_inverse_transform 函數測試"""

    def test_inverse_log1p(self):
        """log1p 逆轉換測試"""
        original = pd.Series([0.0, 1.0, 10.0, 100.0])  # Use float to match output dtype
        transformed = _apply_transform(original, "log1p")
        restored = _inverse_transform(transformed, "log1p")
        pd.testing.assert_series_equal(restored, original, check_names=False)

    def test_inverse_none(self):
        """none 逆轉換應返回原始資料"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = _inverse_transform(series, "none")
        pd.testing.assert_series_equal(result, series)

    def test_inverse_yeo_johnson_with_lambda(self, df_highly_skewed):
        """Yeo-Johnson 逆轉換測試（含 lambda 參數）"""
        out_df, meta = run_transform(df_highly_skewed, mode="large")
        
        if meta["method"] == "yeo_johnson" and "lmbda" in meta["transform_params"]:
            transformed = out_df["interval_days_transformed"]
            restored = _inverse_transform(
                transformed,
                "yeo_johnson",
                meta["transform_params"]
            )
            original = df_highly_skewed["interval_days"]
            np.testing.assert_array_almost_equal(
                restored.values,
                original.values,
                decimal=5
            )

    def test_inverse_yeo_johnson_without_lambda(self):
        """Yeo-Johnson 無 lambda 參數應返回原值並警告"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = _inverse_transform(series, "yeo_johnson")
        pd.testing.assert_series_equal(result, series)

    def test_inverse_invalid_method_raises(self):
        """無效方法應拋出 ValueError"""
        series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unsupported transform method"):
            _inverse_transform(series, "invalid_method")

    def test_inverse_empty_series(self):
        """空 series 應能處理"""
        series = pd.Series([], dtype=float)
        result = _inverse_transform(series, "log1p")
        assert len(result) == 0
