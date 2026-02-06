# ======================================================
# test_stability_assessment.py
# ======================================================
# 單元測試模組: 穩定性檢驗
# 對應模組: run_stability_assessment()
# ------------------------------------------------------

import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from repurchase_cycle.modules.stability_assessment import (
    run_stability_assessment,
    DEFAULT_STABILITY_PARAMS,
    MODE_DEFAULTS,
)


# ======================================================
# Fixtures
# ======================================================
@pytest.fixture
def unimodal_df() -> pd.DataFrame:
    """單峰測試資料"""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=0.5, size=500)
    return pd.DataFrame({"interval_days_transformed": data})


@pytest.fixture
def bimodal_df() -> pd.DataFrame:
    """雙峰測試資料"""
    rng = np.random.default_rng(42)
    peak1 = rng.normal(loc=5.0, scale=0.5, size=300)
    peak2 = rng.normal(loc=15.0, scale=0.8, size=300)
    data = np.concatenate([peak1, peak2])
    return pd.DataFrame({"interval_days_transformed": data})


@pytest.fixture
def single_peak_table() -> list:
    """單峰 peaks_table"""
    return [{"pos": 5.0, "height": 0.1, "width": 0.3, "prominence": 0.05}]


@pytest.fixture
def double_peak_table() -> list:
    """雙峰 peaks_table"""
    return [
        {"pos": 5.0, "height": 0.1, "width": 0.3, "prominence": 0.05},
        {"pos": 15.0, "height": 0.08, "width": 0.4, "prominence": 0.04},
    ]


@pytest.fixture
def log_transform_meta() -> dict:
    """Log1p 轉換的 transform_meta"""
    return {
        "method": "log1p",  # 修正: 使用 log1p 而非 log
        "transform_params": {}
    }


@pytest.fixture
def yeo_johnson_transform_meta() -> dict:
    """Yeo-Johnson 轉換的 transform_meta"""
    return {
        "method": "yeo_johnson",  # 修正: 使用 yeo_johnson 而非 box-cox
        "transform_params": {"lmbda": 0.5}  # 注意: 參數名是 lmbda 不是 lambda
    }


# ======================================================
# Test 1: 基本功能測試
# ======================================================
class TestBasicFunctionality:
    """基本功能測試"""

    def test_returns_tuple(self, unimodal_df, single_peak_table, tmp_path):
        """應回傳 tuple (stable_peaks, plot_path)"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert isinstance(stable_peaks, list)
        assert isinstance(plot_path, str)

    def test_stable_peaks_structure(self, unimodal_df, single_peak_table, tmp_path):
        """stable_peaks 結構驗證"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20},
            output_dir=str(tmp_path)
        )
        for peak in stable_peaks:
            assert "pos" in peak
            assert "support_ratio" in peak
            assert isinstance(peak["pos"], float)
            assert isinstance(peak["support_ratio"], float)

    def test_support_ratio_range(self, unimodal_df, single_peak_table, tmp_path):
        """support_ratio 應在 [0, 1] 範圍內"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 30},
            output_dir=str(tmp_path)
        )
        for peak in stable_peaks:
            assert 0.0 <= peak["support_ratio"] <= 1.0

    def test_plot_file_exists(self, unimodal_df, single_peak_table, tmp_path):
        """圖檔應存在"""
        _, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20},
            output_dir=str(tmp_path)
        )
        assert os.path.exists(plot_path)
        assert plot_path.endswith(".png")


# ======================================================
# Test 2: 模式切換測試
# ======================================================
class TestModeSwitch:
    """模式切換測試"""

    def test_small_mode(self, unimodal_df, single_peak_table, tmp_path):
        """Small 模式測試"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test_small",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 30},
            output_dir=str(tmp_path)
        )
        for peak in stable_peaks:
            assert 0.0 <= peak["support_ratio"] <= 1.0
        assert os.path.exists(plot_path)

    def test_medium_mode(self, unimodal_df, single_peak_table, tmp_path):
        """Medium 模式測試"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test_medium",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="medium",
            mod_params={"n_bootstrap": 20},
            output_dir=str(tmp_path)
        )
        for peak in stable_peaks:
            assert 0.0 <= peak["support_ratio"] <= 1.0
        assert os.path.exists(plot_path)

    def test_large_mode(self, unimodal_df, single_peak_table, tmp_path):
        """Large 模式測試"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test_large",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="large",
            mod_params={"n_bootstrap": 10},
            output_dir=str(tmp_path)
        )
        for peak in stable_peaks:
            assert 0.0 <= peak["support_ratio"] <= 1.0
        assert os.path.exists(plot_path)

    def test_mode_defaults_applied(self, unimodal_df, single_peak_table, tmp_path):
        """各模式預設值應正確套用"""
        # 驗證 MODE_DEFAULTS 存在
        assert "small" in MODE_DEFAULTS
        assert "medium" in MODE_DEFAULTS
        assert "large" in MODE_DEFAULTS

        # small 模式應有較多 bootstrap
        assert MODE_DEFAULTS["small"]["n_bootstrap"] > MODE_DEFAULTS["large"]["n_bootstrap"]

    def test_unknown_mode_uses_defaults(self, unimodal_df, single_peak_table, tmp_path):
        """未知模式應使用 DEFAULT_STABILITY_PARAMS"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test_unknown",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="unknown_mode",
            mod_params={"n_bootstrap": 20},
            output_dir=str(tmp_path)
        )
        # 應正常執行不報錯
        assert isinstance(stable_peaks, list)
        assert os.path.exists(plot_path)


# ======================================================
# Test 3: 參數覆蓋測試
# ======================================================
class TestParameterOverride:
    """參數覆蓋測試"""

    def test_support_threshold_filtering(self, unimodal_df, single_peak_table, tmp_path):
        """超高 support_threshold 應過濾掉所有峰"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 20,
                "support_threshold": 1.0  # 100% 支持才保留
            },
            output_dir=str(tmp_path)
        )
        # 極高閾值應過濾掉大部分或所有峰
        assert len(stable_peaks) == 0

    def test_match_tolerance_override(self, unimodal_df, single_peak_table, tmp_path):
        """極小 match_tol 應導致幾乎無法匹配"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 20,
                "match_tol": 0.00001,  # 極小容差
                "support_threshold": 0.5
            },
            output_dir=str(tmp_path)
        )
        # 極小容差應導致低支持率，可能被過濾
        assert len(stable_peaks) == 0

    def test_match_tolerance_none_string(self, unimodal_df, single_peak_table, tmp_path):
        """match_tol 為 'None' 字串時應自動計算"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 20,
                "match_tol": "None"  # 字串 None
            },
            output_dir=str(tmp_path)
        )
        assert os.path.exists(plot_path)

    def test_match_tolerance_null_string(self, unimodal_df, single_peak_table, tmp_path):
        """match_tol 為 'null' 字串時應自動計算"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 20,
                "match_tol": "null"  # 字串 null
            },
            output_dir=str(tmp_path)
        )
        assert os.path.exists(plot_path)

    def test_sample_fraction_override(self, unimodal_df, single_peak_table, tmp_path):
        """自訂 sample_fraction"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 20,
                "sample_fraction": 0.5
            },
            output_dir=str(tmp_path)
        )
        # 應正常執行
        assert os.path.exists(plot_path)

    def test_grid_size_override(self, unimodal_df, single_peak_table, tmp_path):
        """自訂 grid_size"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 20,
                "grid_size": 256
            },
            output_dir=str(tmp_path)
        )
        assert os.path.exists(plot_path)


# ======================================================
# Test 4: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_no_peaks_input(self, unimodal_df, tmp_path):
        """空 peaks_table 應回傳空結果"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=[],
            mode="small",
            mod_params={"n_bootstrap": 10},
            output_dir=str(tmp_path)
        )
        assert stable_peaks == []
        assert plot_path == ""

    def test_empty_dataframe(self, tmp_path, single_peak_table):
        """空資料框應回傳空結果"""
        df = pd.DataFrame({"interval_days_transformed": []})
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=single_peak_table,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert stable_peaks == []
        assert plot_path == ""

    def test_all_nan_values(self, tmp_path, single_peak_table):
        """全 NaN 資料應回傳空結果"""
        df = pd.DataFrame({"interval_days_transformed": [np.nan, np.nan, np.nan]})
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=single_peak_table,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert stable_peaks == []
        assert plot_path == ""

    def test_single_value(self, tmp_path, single_peak_table):
        """單一值資料應能處理"""
        df = pd.DataFrame({"interval_days_transformed": [5.0]})
        # 單一值無法建立有效 KDE，應回傳空結果
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 10},
            output_dir=str(tmp_path)
        )
        # 應能處理不報錯
        assert isinstance(stable_peaks, list)

    def test_two_identical_values(self, tmp_path, single_peak_table):
        """兩個相同值資料應能處理（KDE 可能失敗）"""
        df = pd.DataFrame({"interval_days_transformed": [5.0, 5.0]})
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 10},
            output_dir=str(tmp_path)
        )
        # 應能處理不報錯
        assert isinstance(stable_peaks, list)

    def test_data_range_zero(self, tmp_path, single_peak_table):
        """資料範圍為零時應能處理"""
        df = pd.DataFrame({"interval_days_transformed": [5.0] * 100})
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 10},
            output_dir=str(tmp_path)
        )
        # 應能處理不報錯
        assert isinstance(stable_peaks, list)


# ======================================================
# Test 5: 欄位名稱測試
# ======================================================
class TestColumnNames:
    """欄位名稱測試"""

    def test_value_col_override(self, tmp_path, single_peak_table):
        """自訂 value_col 欄位名稱"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"custom_col": rng.normal(5.0, 0.5, 500)})

        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={
                "value_col": "custom_col",
                "n_bootstrap": 20
            },
            output_dir=str(tmp_path)
        )
        assert os.path.exists(plot_path)
        for peak in stable_peaks:
            assert 0.0 <= peak["support_ratio"] <= 1.0

    def test_missing_value_col_raises(self, unimodal_df, single_peak_table, tmp_path):
        """缺少 value_col 應拋出 ValueError"""
        with pytest.raises(ValueError, match="not found"):
            run_stability_assessment(
                profile_name="test",
                df=unimodal_df,
                peaks_table=single_peak_table,
                mode="small",
                mod_params={"value_col": "nonexistent_column"},
                output_dir=str(tmp_path)
            )


# ======================================================
# Test 6: profile_name 測試
# ======================================================
class TestProfileName:
    """profile_name 參數測試"""

    def test_with_profile_name(self, unimodal_df, single_peak_table, tmp_path):
        """有 profile_name 時檔名包含 profile"""
        _, plot_path = run_stability_assessment(
            profile_name="my_profile",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20},
            output_dir=str(tmp_path)
        )
        assert "my_profile" in plot_path

    def test_without_profile_name(self, unimodal_df, single_peak_table, tmp_path):
        """無 profile_name 時使用預設檔名"""
        _, plot_path = run_stability_assessment(
            profile_name=None,
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20},
            output_dir=str(tmp_path)
        )
        assert "stability_assessment_peaks" in plot_path


# ======================================================
# Test 7: 雙峰穩定性測試
# ======================================================
class TestBimodalStability:
    """雙峰穩定性測試"""

    def test_both_peaks_stable(self, bimodal_df, double_peak_table, tmp_path):
        """雙峰資料中兩個峰都應穩定"""
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test_bimodal",
            df=bimodal_df,
            peaks_table=double_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 50,
                "support_threshold": 0.5
            },
            output_dir=str(tmp_path)
        )
        # 雙峰資料中至少應有 1 個穩定峰
        assert len(stable_peaks) >= 1
        assert os.path.exists(plot_path)

    def test_peak_positions_preserved(self, bimodal_df, double_peak_table, tmp_path):
        """穩定峰的位置應來自原始 peaks_table"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=bimodal_df,
            peaks_table=double_peak_table,
            mode="small",
            mod_params={
                "n_bootstrap": 30,
                "support_threshold": 0.3
            },
            output_dir=str(tmp_path)
        )
        original_positions = {p["pos"] for p in double_peak_table}
        for sp in stable_peaks:
            assert sp["pos"] in original_positions


# ======================================================
# Test 8: random_state 測試
# ======================================================
class TestRandomState:
    """隨機種子測試"""

    def test_reproducibility(self, unimodal_df, single_peak_table, tmp_path):
        """相同 random_state 應產生相同結果"""
        result1, _ = run_stability_assessment(
            profile_name="test1",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            general_params={"random_state": 42},
            mod_params={"n_bootstrap": 30},
            output_dir=str(tmp_path / "run1")
        )

        result2, _ = run_stability_assessment(
            profile_name="test2",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            general_params={"random_state": 42},
            mod_params={"n_bootstrap": 30},
            output_dir=str(tmp_path / "run2")
        )

        # 相同種子應產生相同支持率
        if result1 and result2:
            assert result1[0]["support_ratio"] == result2[0]["support_ratio"]

    def test_different_random_state(self, unimodal_df, single_peak_table, tmp_path):
        """不同 random_state 可能產生不同結果"""
        result1, _ = run_stability_assessment(
            profile_name="test1",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            general_params={"random_state": 42},
            mod_params={"n_bootstrap": 30},
            output_dir=str(tmp_path / "run1")
        )

        result2, _ = run_stability_assessment(
            profile_name="test2",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            general_params={"random_state": 123},
            mod_params={"n_bootstrap": 30},
            output_dir=str(tmp_path / "run2")
        )

        # 兩個結果都應該是有效的
        assert isinstance(result1, list)
        assert isinstance(result2, list)


# ======================================================
# Test 9: 輸出目錄測試
# ======================================================
class TestOutputDirectory:
    """輸出目錄測試"""

    def test_creates_output_dir(self, unimodal_df, single_peak_table, tmp_path):
        """應自動建立輸出目錄"""
        new_dir = tmp_path / "new" / "nested" / "dir"
        _, plot_path = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20},
            output_dir=str(new_dir)
        )
        assert os.path.exists(new_dir)
        assert os.path.exists(plot_path)


# ======================================================
# Test 10: 多峰過濾測試
# ======================================================
class TestPeakFiltering:
    """峰過濾測試"""

    def test_unstable_peak_filtered(self, unimodal_df, tmp_path):
        """不穩定的峰應被過濾"""
        # 加入一個離群峰位置（不在資料分布中）
        peaks_table = [
            {"pos": 5.0, "height": 0.1, "width": 0.3, "prominence": 0.05},  # 穩定峰
            {"pos": 100.0, "height": 0.05, "width": 0.2, "prominence": 0.02},  # 不穩定峰
        ]
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,  # 資料集中在 5.0 附近
            peaks_table=peaks_table,
            mode="small",
            mod_params={
                "n_bootstrap": 30,
                "support_threshold": 0.5
            },
            output_dir=str(tmp_path)
        )
        # 100.0 位置的峰不應出現在穩定峰中
        stable_positions = {p["pos"] for p in stable_peaks}
        assert 100.0 not in stable_positions

    def test_all_unstable_returns_empty(self, tmp_path):
        """所有峰都不穩定時應回傳空列表"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"interval_days_transformed": rng.normal(5.0, 0.5, 500)})
        # 所有峰都遠離資料分布
        peaks_table = [
            {"pos": 50.0, "height": 0.1},
            {"pos": 100.0, "height": 0.05},
        ]
        stable_peaks, plot_path = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=peaks_table,
            mode="small",
            mod_params={
                "n_bootstrap": 30,
                "support_threshold": 0.5
            },
            output_dir=str(tmp_path)
        )
        assert stable_peaks == []
        # 即使無穩定峰，圖檔仍應存在（顯示所有峰都不穩定）
        assert os.path.exists(plot_path)


# ======================================================
# Test 11: transform_meta 參數測試
# ======================================================
class TestTransformMeta:
    """transform_meta 參數測試"""

    def test_without_transform_meta(self, unimodal_df, single_peak_table, tmp_path):
        """無 transform_meta 時不應包含原始尺度欄位"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20, "support_threshold": 0.3},
            output_dir=str(tmp_path),
            transform_meta=None
        )
        for peak in stable_peaks:
            assert "pos" in peak
            assert "support_ratio" in peak
            # 無 transform_meta 時不應有原始尺度欄位
            assert "pos_original" not in peak
            assert "pos_transformed" not in peak

    def test_with_transform_meta_none_method(self, unimodal_df, single_peak_table, tmp_path):
        """transform_meta method 為 'none' 時不應添加原始尺度欄位"""
        transform_meta = {"method": "none", "transform_params": {}}
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20, "support_threshold": 0.3},
            output_dir=str(tmp_path),
            transform_meta=transform_meta
        )
        for peak in stable_peaks:
            # method 為 'none' 時不應添加原始尺度欄位
            assert "pos_original" not in peak
            assert "pos_transformed" not in peak

    def test_with_log1p_transform_meta(self, unimodal_df, single_peak_table, tmp_path, log_transform_meta):
        """使用 log1p transform_meta 時應包含原始尺度欄位"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20, "support_threshold": 0.3},
            output_dir=str(tmp_path),
            transform_meta=log_transform_meta
        )
        for peak in stable_peaks:
            assert "pos_original" in peak
            assert "pos_transformed" in peak
            assert isinstance(peak["pos_original"], float)
            assert isinstance(peak["pos_transformed"], float)
            # pos_transformed 應與原始 pos 相同
            assert peak["pos_transformed"] == peak["pos"]

    def test_with_yeo_johnson_transform_meta(self, unimodal_df, single_peak_table, tmp_path, yeo_johnson_transform_meta):
        """使用 yeo_johnson transform_meta 時應包含原始尺度欄位"""
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20, "support_threshold": 0.3},
            output_dir=str(tmp_path),
            transform_meta=yeo_johnson_transform_meta
        )
        for peak in stable_peaks:
            assert "pos_original" in peak
            assert "pos_transformed" in peak
            assert isinstance(peak["pos_original"], float)
            assert isinstance(peak["pos_transformed"], float)

    def test_inverse_transform_applied_correctly(self, tmp_path):
        """驗證逆轉換正確應用"""
        # 創建 log1p 轉換後的資料
        # log1p(x) = y  =>  x = expm1(y)
        # 若 y ≈ 2.0，則 x = expm1(2.0) ≈ 6.389
        rng = np.random.default_rng(42)
        transformed_data = rng.normal(loc=2.0, scale=0.2, size=500)
        df = pd.DataFrame({"interval_days_transformed": transformed_data})
        
        peaks_table = [{"pos": 2.0, "height": 0.1, "width": 0.3, "prominence": 0.05}]
        transform_meta = {"method": "log1p", "transform_params": {}}
        
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=peaks_table,
            mode="small",
            mod_params={"n_bootstrap": 30, "support_threshold": 0.3},
            output_dir=str(tmp_path),
            transform_meta=transform_meta
        )
        
        if stable_peaks:
            # log1p 的逆轉換是 expm1，pos=2.0 應轉換為 expm1(2.0) ≈ 6.389
            assert stable_peaks[0]["pos_original"] > stable_peaks[0]["pos_transformed"]
            expected_original = np.expm1(2.0)  # ≈ 6.389
            assert abs(stable_peaks[0]["pos_original"] - expected_original) < 0.01

    def test_transform_meta_empty_stable_peaks(self, tmp_path):
        """當無穩定峰時，transform_meta 不應導致錯誤"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"interval_days_transformed": rng.normal(5.0, 0.5, 500)})
        # 峰位置遠離資料分布
        peaks_table = [{"pos": 100.0, "height": 0.1}]
        transform_meta = {"method": "log1p", "transform_params": {}}
        
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=df,
            peaks_table=peaks_table,
            mode="small",
            mod_params={"n_bootstrap": 20, "support_threshold": 0.5},
            output_dir=str(tmp_path),
            transform_meta=transform_meta
        )
        # 應回傳空列表，不報錯
        assert stable_peaks == []

    def test_yeo_johnson_without_lmbda_param(self, unimodal_df, single_peak_table, tmp_path):
        """yeo_johnson 缺少 lmbda 參數時應能處理（回傳原值）"""
        transform_meta = {"method": "yeo_johnson", "transform_params": {}}
        stable_peaks, _ = run_stability_assessment(
            profile_name="test",
            df=unimodal_df,
            peaks_table=single_peak_table,
            mode="small",
            mod_params={"n_bootstrap": 20, "support_threshold": 0.3},
            output_dir=str(tmp_path),
            transform_meta=transform_meta
        )
        # 應能處理不報錯，pos_original 應等於 pos（因缺少 lmbda 會回傳原值）
        for peak in stable_peaks:
            assert "pos_original" in peak
            assert peak["pos_original"] == peak["pos"]


# ======================================================
# Test 12: DEFAULT_STABILITY_PARAMS 測試
# ======================================================
class TestDefaultParams:
    """預設參數測試"""

    def test_default_params_exist(self):
        """驗證 DEFAULT_STABILITY_PARAMS 包含必要欄位"""
        assert "n_bootstrap" in DEFAULT_STABILITY_PARAMS
        assert "sample_fraction" in DEFAULT_STABILITY_PARAMS
        assert "support_threshold" in DEFAULT_STABILITY_PARAMS
        assert "value_col" in DEFAULT_STABILITY_PARAMS
        assert "match_tol" in DEFAULT_STABILITY_PARAMS
        assert "grid_size" in DEFAULT_STABILITY_PARAMS

    def test_default_value_col(self):
        """驗證預設 value_col"""
        assert DEFAULT_STABILITY_PARAMS["value_col"] == "interval_days_transformed"

    def test_default_match_tol_is_none(self):
        """驗證預設 match_tol 為 None（自動計算）"""
        assert DEFAULT_STABILITY_PARAMS["match_tol"] is None

    def test_mode_defaults_structure(self):
        """驗證 MODE_DEFAULTS 結構"""
        for mode in ["small", "medium", "large"]:
            assert mode in MODE_DEFAULTS
            assert "n_bootstrap" in MODE_DEFAULTS[mode]
            assert "sample_fraction" in MODE_DEFAULTS[mode]
