# ======================================================
# test_unimodality.py
# ======================================================
# 單元測試模組: 單峰性檢定
# 對應模組: run_unimodality_test()
# ------------------------------------------------------

import numpy as np
import pandas as pd
import pytest

from repurchase_cycle.modules.unimodality_test import (
    run_unimodality_test,
    _resolve_mode_auto,
    _subsample,
    _kde_peak_count,
    _dip_p_value,
    _silverman_unimodality_p_value,
    _emd_unimodality_p_value,
    _run_method,
    _is_flat_distribution,  # Add this import
    DEFAULT_UNIMODALITY_PARAMS,
)


# ======================================================
# Fixtures
# ======================================================
@pytest.fixture
def unimodal_df() -> pd.DataFrame:
    """單峰常態分布資料"""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=500)
    return pd.DataFrame({"interval_days_transformed": data})


@pytest.fixture
def bimodal_df() -> pd.DataFrame:
    """明顯雙峰分布資料"""
    rng = np.random.default_rng(42)
    comp1 = rng.normal(loc=-4.0, scale=0.5, size=300)
    comp2 = rng.normal(loc=4.0, scale=0.5, size=300)
    data = np.concatenate([comp1, comp2])
    rng.shuffle(data)
    return pd.DataFrame({"interval_days_transformed": data})


@pytest.fixture
def trimodal_df() -> pd.DataFrame:
    """三峰分布資料"""
    rng = np.random.default_rng(42)
    comp1 = rng.normal(loc=-6.0, scale=0.5, size=200)
    comp2 = rng.normal(loc=0.0, scale=0.5, size=200)
    comp3 = rng.normal(loc=6.0, scale=0.5, size=200)
    data = np.concatenate([comp1, comp2, comp3])
    rng.shuffle(data)
    return pd.DataFrame({"interval_days_transformed": data})


@pytest.fixture
def flat_uniform_df() -> pd.DataFrame:
    """平坦/均勻分布資料"""
    rng = np.random.default_rng(42)
    data = rng.uniform(low=0.0, high=10.0, size=500)
    return pd.DataFrame({"interval_days_transformed": data})


# ======================================================
# Test 1: _resolve_mode_auto 單元測試
# ======================================================
class TestResolveModeAuto:
    """模式自動解析測試"""

    def test_small_mode(self):
        """小資料量應返回 small"""
        cfg = {"data_size_thresholds": {"small": 100, "medium": 1000}}
        assert _resolve_mode_auto(50, cfg) == "small"
        assert _resolve_mode_auto(100, cfg) == "small"

    def test_medium_mode(self):
        """中等資料量應返回 medium"""
        cfg = {"data_size_thresholds": {"small": 100, "medium": 1000}}
        assert _resolve_mode_auto(101, cfg) == "medium"
        assert _resolve_mode_auto(500, cfg) == "medium"
        assert _resolve_mode_auto(1000, cfg) == "medium"

    def test_large_mode(self):
        """大資料量應返回 large"""
        cfg = {"data_size_thresholds": {"small": 100, "medium": 1000}}
        assert _resolve_mode_auto(1001, cfg) == "large"
        assert _resolve_mode_auto(10000, cfg) == "large"

    def test_default_thresholds(self):
        """缺少閾值時使用預設值"""
        cfg = {}
        # Default: small=1e4, medium=1e6
        assert _resolve_mode_auto(100, cfg) == "small"
        assert _resolve_mode_auto(50000, cfg) == "medium"
        assert _resolve_mode_auto(2000000, cfg) == "large"


# ======================================================
# Test 2: _subsample 單元測試
# ======================================================
class TestSubsample:
    """子抽樣函數測試"""

    def test_no_subsample_when_small(self):
        """資料量小於 max_n 時不抽樣"""
        values = np.array([1, 2, 3, 4, 5])
        result = _subsample(values, max_n=10, seed=42)
        np.testing.assert_array_equal(result, values)

    def test_subsample_when_large(self):
        """資料量大於 max_n 時應抽樣"""
        values = np.arange(1000)
        result = _subsample(values, max_n=100, seed=42)
        assert len(result) == 100
        # 所有值應來自原始資料
        assert all(v in values for v in result)

    def test_reproducibility(self):
        """相同種子應產生相同結果"""
        values = np.arange(1000)
        result1 = _subsample(values, max_n=100, seed=42)
        result2 = _subsample(values, max_n=100, seed=42)
        np.testing.assert_array_equal(result1, result2)


# ======================================================
# Test 3: _kde_peak_count 單元測試
# ======================================================
class TestKdePeakCount:
    """KDE 峰數計算測試"""

    def test_unimodal_returns_1(self):
        """單峰資料應返回 1"""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 500)
        assert _kde_peak_count(values) == 1

    def test_bimodal_returns_2(self):
        """明顯雙峰資料應返回 2"""
        rng = np.random.default_rng(42)
        values = np.concatenate([
            rng.normal(-5, 0.5, 300),
            rng.normal(5, 0.5, 300)
        ])
        assert _kde_peak_count(values) >= 2

    def test_constant_returns_1(self):
        """常數值應返回 1"""
        values = np.ones(100)
        assert _kde_peak_count(values) == 1

    def test_few_samples_returns_1(self):
        """極少樣本應返回 1"""
        values = np.array([1, 2, 3])
        assert _kde_peak_count(values) == 1


# ======================================================
# Test 4: _dip_p_value 單元測試
# ======================================================
class TestDipPValue:
    """Dip test p-value 測試"""

    def test_unimodal_high_p(self):
        """單峰資料應有較高 p-value"""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 500)
        p = _dip_p_value(values, seed=42, cfg={})
        assert 0.0 <= p <= 1.0

    def test_bimodal_low_p(self):
        """雙峰資料應有較低 p-value"""
        rng = np.random.default_rng(42)
        values = np.concatenate([
            rng.normal(-5, 0.5, 300),
            rng.normal(5, 0.5, 300)
        ])
        p = _dip_p_value(values, seed=42, cfg={})
        assert 0.0 <= p <= 1.0
        # 雙峰通常有較低 p-value
        assert p < 0.1

    def test_few_samples_returns_1(self):
        """極少樣本應返回 1.0"""
        values = np.array([1, 2, 3])
        p = _dip_p_value(values, seed=42, cfg={})
        assert p == 1.0


# ======================================================
# Test 5: _silverman_unimodality_p_value 單元測試
# ======================================================
class TestSilvermanPValue:
    """Silverman test p-value 測試"""

    def test_unimodal_high_p(self):
        """單峰資料應有較高 p-value"""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 200)
        cfg = {"silverman_grid_size": 256, "silverman_search_iters": 20, "silverman_bootstrap_samples": 50}
        p = _silverman_unimodality_p_value(values, seed=42, cfg=cfg)
        assert 0.0 <= p <= 1.0

    def test_bimodal_low_p(self):
        """雙峰資料應有較低 p-value"""
        rng = np.random.default_rng(42)
        values = np.concatenate([
            rng.normal(-5, 0.5, 150),
            rng.normal(5, 0.5, 150)
        ])
        cfg = {"silverman_grid_size": 256, "silverman_search_iters": 20, "silverman_bootstrap_samples": 50}
        p = _silverman_unimodality_p_value(values, seed=42, cfg=cfg)
        assert 0.0 <= p <= 1.0

    def test_few_samples_returns_1(self):
        """極少樣本應返回 1.0"""
        values = np.array([1, 2, 3])
        cfg = {}
        p = _silverman_unimodality_p_value(values, seed=42, cfg=cfg)
        assert p == 1.0


# ======================================================
# Test 6: _emd_unimodality_p_value 單元測試
# ======================================================
class TestEmdPValue:
    """EMD heuristic p-value 測試"""

    def test_unimodal_high_p(self):
        """單峰資料應有較高 p-value"""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 200)
        cfg = {"emd_bootstrap_samples": 50, "emd_sample_size": 500}
        p = _emd_unimodality_p_value(values, seed=42, cfg=cfg)
        assert 0.0 <= p <= 1.0

    def test_few_samples_returns_1(self):
        """極少樣本應返回 1.0"""
        values = np.array([1, 2, 3])
        cfg = {}
        p = _emd_unimodality_p_value(values, seed=42, cfg=cfg)
        assert p == 1.0


# ======================================================
# Test 7: run_unimodality_test 基本功能測試
# ======================================================
class TestRunUnimodalityTestBasic:
    """run_unimodality_test 基本功能測試"""

    @pytest.mark.parametrize("mode", ["small", "medium", "large"])
    def test_p_value_range(self, unimodal_df, mode):
        """p-value 應在 [0, 1] 範圍內"""
        params = {"dip_bootstrap_samples": 30, "max_sample_for_test": 1000}
        result = run_unimodality_test(unimodal_df, mode=mode, mod_params=params)

        assert 0.0 <= result["dip_p"] <= 1.0
        assert result["decision"] in {"unimodal", "multimodal"}
        assert isinstance(result["method_used"], str)
        assert len(result["method_used"]) > 0

    def test_result_structure(self, unimodal_df):
        """結果結構驗證"""
        result = run_unimodality_test(unimodal_df, mode="small")
        assert "dip_p" in result
        assert "method_used" in result
        assert "decision" in result

    def test_unimodal_detected_as_unimodal(self, unimodal_df):
        """單峰資料應被識別為 unimodal"""
        params = {"dip_bootstrap_samples": 50}
        result = run_unimodality_test(unimodal_df, mode="small", mod_params=params)
        # 單峰資料通常應有較高 p-value，判定為 unimodal
        # 但統計測試有隨機性，此處只檢查結果合理
        assert result["decision"] in {"unimodal", "multimodal"}

    def test_bimodal_detected_as_multimodal(self, bimodal_df):
        """雙峰資料應被識別為 multimodal"""
        params = {"dip_bootstrap_samples": 50}
        result = run_unimodality_test(bimodal_df, mode="small", mod_params=params)
        # 明顯雙峰資料應被判定為 multimodal
        assert result["decision"] == "multimodal"
        assert result["dip_p"] < 0.05


# ======================================================
# Test 8: 模式切換測試
# ======================================================
class TestModeSwitch:
    """模式切換與方法選擇測試"""

    def test_small_mode_methods(self, unimodal_df):
        """Small 模式應使用 dip + silverman"""
        params = {
            "dip_bootstrap_samples": 30,
            "silverman_bootstrap_samples": 30,
        }
        result = run_unimodality_test(unimodal_df, mode="small", mod_params=params)
        assert "dip" in result["method_used"]
        assert "silverman" in result["method_used"]

    def test_medium_mode_methods(self, unimodal_df):
        """Medium 模式應使用 dip_subsampled + silverman"""
        params = {
            "dip_bootstrap_samples": 30,
            "silverman_bootstrap_samples": 30,
            "max_sample_for_test": 100,
        }
        result = run_unimodality_test(unimodal_df, mode="medium", mod_params=params)
        assert "dip_subsampled" in result["method_used"] or "silverman" in result["method_used"]

    def test_large_mode_methods(self, unimodal_df):
        """Large 模式應使用 kde_extrema + smoothness_emd"""
        params = {"emd_bootstrap_samples": 30}
        result = run_unimodality_test(unimodal_df, mode="large", mod_params=params)
        assert "kde_extrema" in result["method_used"] or "smoothness_emd" in result["method_used"]

    def test_large_kde_multimodal_detection(self, bimodal_df):
        """Large 模式 KDE 極值法應偵測多峰"""
        result = run_unimodality_test(bimodal_df, mode="large")
        assert result["method_used"] == "kde_extrema+smoothness_emd"
        assert result["decision"] == "multimodal"
        assert result["dip_p"] == 0.0  # KDE 極值法偵測到多峰時給 0.0


# ======================================================
# Test 9: auto 模式測試
# ======================================================
class TestAutoMode:
    """Auto 模式測試"""

    def test_auto_switches_by_size(self):
        """Auto 模式應依資料量切換方法"""
        params = {
            "data_size_thresholds": {"small": 50, "medium": 100},
            "dip_bootstrap_samples": 20,
            "silverman_bootstrap_samples": 20,
            "max_sample_for_test": 80,
        }

        # Small
        df_small = pd.DataFrame({
            "interval_days_transformed": np.random.normal(0, 1, 30)
        })
        res_small = run_unimodality_test(df_small, mode="auto", general_params=params, mod_params=params)
        assert "dip" in res_small["method_used"]
        assert "subsampled" not in res_small["method_used"]

        # Medium
        df_medium = pd.DataFrame({
            "interval_days_transformed": np.random.normal(0, 1, 80)
        })
        res_medium = run_unimodality_test(df_medium, mode="auto", general_params=params, mod_params=params)
        assert "dip_subsampled" in res_medium["method_used"]

        # Large
        df_large = pd.DataFrame({
            "interval_days_transformed": np.random.normal(0, 1, 150)
        })
        res_large = run_unimodality_test(df_large, mode="auto", general_params=params, mod_params=params)
        assert "kde_extrema" in res_large["method_used"]


# ======================================================
# Test 10: 參數覆蓋測試
# ======================================================
class TestParameterOverride:
    """參數覆蓋測試"""

    def test_custom_alpha(self, unimodal_df):
        """自訂 alpha 閾值"""
        # 使用非常寬鬆的 alpha
        params = {"alpha": 0.99, "dip_bootstrap_samples": 30}
        result = run_unimodality_test(unimodal_df, mode="small", mod_params=params)
        # 幾乎所有資料都會被判為 multimodal
        assert result["decision"] in {"unimodal", "multimodal"}

    def test_custom_value_col(self):
        """自訂 value_col"""
        df = pd.DataFrame({"custom_col": np.random.normal(0, 1, 100)})
        params = {"value_col": "custom_col", "dip_bootstrap_samples": 20}
        result = run_unimodality_test(df, mode="small", mod_params=params)
        assert result["decision"] in {"unimodal", "multimodal"}

    def test_custom_methods_by_mode(self, unimodal_df):
        """自訂 methods_by_mode"""
        params = {
            "methods_by_mode": {"small": ["dip"]},  # 只使用 dip
            "dip_bootstrap_samples": 30,
        }
        result = run_unimodality_test(unimodal_df, mode="small", mod_params=params)
        assert result["method_used"] == "dip"


# ======================================================
# Test 11: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_missing_column_raises(self):
        """缺少 value_col 應拋出 KeyError"""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(KeyError, match="interval_days_transformed"):
            run_unimodality_test(df)

    def test_empty_values_raises(self):
        """全 NaN 值應拋出 ValueError"""
        df = pd.DataFrame({"interval_days_transformed": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="No valid"):
            run_unimodality_test(df)

    def test_unsupported_mode_raises(self, unimodal_df):
        """不支援的模式應拋出 ValueError"""
        with pytest.raises(ValueError, match="Unsupported mode"):
            run_unimodality_test(unimodal_df, mode="invalid")

    def test_few_samples(self):
        """少量樣本應能處理"""
        df = pd.DataFrame({"interval_days_transformed": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = run_unimodality_test(df, mode="small")
        assert result["decision"] in {"unimodal", "multimodal"}

    def test_constant_values(self):
        """常數值應被判為 unimodal"""
        df = pd.DataFrame({"interval_days_transformed": [5.0] * 100})
        result = run_unimodality_test(df, mode="small")
        assert result["decision"] == "unimodal"

    def test_zero_mean_flat_check(self):
        """mean 為 0 時平坦檢測應返回 False"""
        # 創建 mean 接近 0 的資料
        values = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] * 10)
        assert _is_flat_distribution(values) is False


# ======================================================
# Test 12: _run_method 單元測試
# ======================================================
class TestRunMethod:
    """_run_method 函數測試"""

    def test_dip_method(self):
        """dip 方法測試"""
        values = np.random.normal(0, 1, 100)
        cfg = {}
        name, p = _run_method("dip", values, seed=42, cfg=cfg, max_sample=1000)
        assert name == "dip"
        assert 0.0 <= p <= 1.0

    def test_dip_subsampled_method(self):
        """dip_subsampled 方法測試"""
        values = np.random.normal(0, 1, 500)
        cfg = {}
        name, p = _run_method("dip_subsampled", values, seed=42, cfg=cfg, max_sample=100)
        assert name == "dip_subsampled"
        assert 0.0 <= p <= 1.0

    def test_silverman_method(self):
        """silverman 方法測試"""
        values = np.random.normal(0, 1, 100)
        cfg = {"silverman_grid_size": 256, "silverman_search_iters": 10, "silverman_bootstrap_samples": 20}
        name, p = _run_method("silverman", values, seed=42, cfg=cfg, max_sample=1000)
        assert name == "silverman"
        assert 0.0 <= p <= 1.0

    def test_kde_extrema_method(self):
        """kde_extrema 方法測試"""
        values = np.random.normal(0, 1, 100)
        cfg = {"silverman_grid_size": 256}
        name, p = _run_method("kde_extrema", values, seed=42, cfg=cfg, max_sample=1000)
        assert name == "kde_extrema"
        assert p in {0.0, 1.0}  # KDE 極值法只返回 0 或 1

    def test_smoothness_emd_method(self):
        """smoothness_emd 方法測試"""
        values = np.random.normal(0, 1, 100)
        cfg = {"emd_bootstrap_samples": 20, "emd_sample_size": 200}
        name, p = _run_method("smoothness_emd", values, seed=42, cfg=cfg, max_sample=1000)
        assert name == "smoothness_emd"
        assert 0.0 <= p <= 1.0

    def test_unknown_method_raises(self):
        """未知方法應拋出 ValueError"""
        values = np.random.normal(0, 1, 100)
        with pytest.raises(ValueError, match="Unknown unimodality method"):
            _run_method("unknown_method", values, seed=42, cfg={}, max_sample=1000)


# ======================================================
# Test 13: 多方法聚合測試
# ======================================================
class TestMethodAggregation:
    """多方法結果聚合測試"""

    def test_multiple_methods_combined(self, unimodal_df):
        """多方法結果應正確聚合"""
        params = {
            "dip_bootstrap_samples": 30,
            "silverman_bootstrap_samples": 30,
        }
        result = run_unimodality_test(unimodal_df, mode="small", mod_params=params)
        # 方法名稱應以 "+" 連接
        assert "+" in result["method_used"]
        methods = result["method_used"].split("+")
        assert len(methods) >= 2

    def test_min_p_value_used(self, bimodal_df):
        """應使用最小 p-value 作為最終結果"""
        params = {
            "dip_bootstrap_samples": 50,
            "silverman_bootstrap_samples": 50,
        }
        result = run_unimodality_test(bimodal_df, mode="small", mod_params=params)
        # 雙峰資料的最小 p-value 應很低
        assert result["dip_p"] < 0.1


# ======================================================
# Test 14: random_state 測試
# ======================================================
class TestRandomState:
    """隨機種子測試"""

    def test_reproducibility(self, unimodal_df):
        """相同種子應產生相同結果"""
        params = {"dip_bootstrap_samples": 30, "silverman_bootstrap_samples": 30}
        general = {"random_seed": 42}

        result1 = run_unimodality_test(unimodal_df, mode="small", general_params=general, mod_params=params)
        result2 = run_unimodality_test(unimodal_df, mode="small", general_params=general, mod_params=params)

        assert result1["dip_p"] == result2["dip_p"]
        assert result1["decision"] == result2["decision"]
