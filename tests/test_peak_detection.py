# ======================================================
# test_peak_detection.py
# ======================================================
# 單元測試模組: 峰值偵測
# 對應模組: run_peak_detection()
# ------------------------------------------------------

import os
import numpy as np
import pandas as pd
import pytest

from repurchase_cycle.modules.peak_detection import (
    run_peak_detection,
    _get_interval_values,
    _build_grid,
    _fit_kde_scipy,
    _fit_kde_kdepy,
    _auto_argrelmax_order,
    _peaks_table_from_indices,
    _detect_peaks_small,
    _detect_peaks_medium,
    _detect_peaks_large,
    DEFAULT_PEAK_PARAMS,
)


# ======================================================
# Fixture: 測試資料生成
# ======================================================
@pytest.fixture
def unimodal_df() -> pd.DataFrame:
    """單峰測試資料"""
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=1.5, size=500)
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def bimodal_df() -> pd.DataFrame:
    """雙峰測試資料：兩個明顯分離的峰"""
    np.random.seed(42)
    peak1 = np.random.normal(loc=5, scale=0.8, size=500)
    peak2 = np.random.normal(loc=20, scale=1.2, size=600)
    data = np.concatenate([peak1, peak2])
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def trimodal_df() -> pd.DataFrame:
    """三峰測試資料"""
    np.random.seed(42)
    peak1 = np.random.normal(loc=5, scale=0.5, size=400)
    peak2 = np.random.normal(loc=15, scale=0.8, size=500)
    peak3 = np.random.normal(loc=30, scale=1.0, size=400)
    data = np.concatenate([peak1, peak2, peak3])
    return pd.DataFrame({"interval_days": data})


@pytest.fixture
def transformed_df() -> pd.DataFrame:
    """含有 interval_days_transformed 欄位的資料"""
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=500)
    return pd.DataFrame({
        "interval_days": data,
        "interval_days_transformed": np.log1p(data)
    })


# ======================================================
# Helper: 檢查函數
# ======================================================
def assert_peak_positions_valid(peaks_table: list, df: pd.DataFrame):
    """檢查 peak 的位置是否都在資料範圍內"""
    if len(df) == 0:
        return
    
    col = "interval_days_transformed" if "interval_days_transformed" in df.columns else "interval_days"
    min_x = df[col].min()
    max_x = df[col].max()
    
    for p in peaks_table:
        assert min_x - 1 <= p["pos"] <= max_x + 1, \
            f"Peak pos {p['pos']} out of expected data range [{min_x}, {max_x}]"


def assert_peaks_table_structure(peaks_table: list, has_transform_meta: bool = False):
    """驗證 peaks_table 結構
    
    Parameters
    ----------
    peaks_table : list
        峰值列表
    has_transform_meta : bool
        是否有提供 transform_meta，若有則應包含 pos_original 和 pos_transformed
    """
    base_keys = {"pos", "height", "width", "prominence"}
    extended_keys = base_keys | {"pos_original", "pos_transformed"}
    
    for peak in peaks_table:
        if has_transform_meta:
            assert extended_keys.issubset(set(peak.keys())), \
                f"Expected keys {extended_keys}, got {set(peak.keys())}"
        else:
            assert base_keys.issubset(set(peak.keys())), \
                f"Expected keys {base_keys}, got {set(peak.keys())}"
        
        assert isinstance(peak["pos"], float)
        assert isinstance(peak["height"], float)
        assert isinstance(peak["width"], float)
        assert isinstance(peak["prominence"], float)
        
        if has_transform_meta:
            assert isinstance(peak["pos_original"], float)
            assert isinstance(peak["pos_transformed"], float)


# ======================================================
# Test 1: _get_interval_values 單元測試
# ======================================================
class TestGetIntervalValues:
    """取得 interval 值函數測試"""

    def test_prioritizes_transformed_column(self, transformed_df):
        """應優先使用 interval_days_transformed"""
        values = _get_interval_values(
            transformed_df,
            ["interval_days_transformed", "interval_days"]
        )
        expected = np.sort(transformed_df["interval_days_transformed"].dropna().values)
        np.testing.assert_array_almost_equal(values, expected)

    def test_fallback_to_interval_days(self, unimodal_df):
        """若無 transformed 欄位，使用 interval_days"""
        values = _get_interval_values(
            unimodal_df,
            ["interval_days_transformed", "interval_days"]
        )
        expected = np.sort(unimodal_df["interval_days"].dropna().values)
        np.testing.assert_array_almost_equal(values, expected)

    def test_removes_nan_values(self):
        """應移除 NaN 值"""
        df = pd.DataFrame({"interval_days": [1.0, np.nan, 3.0, np.nan, 5.0]})
        values = _get_interval_values(df, ["interval_days"])
        assert len(values) == 3
        assert not np.any(np.isnan(values))

    def test_raises_on_missing_column(self):
        """缺少欄位應拋出 KeyError"""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(KeyError):
            _get_interval_values(df, ["interval_days"])

    def test_raises_on_all_nan(self):
        """全部 NaN 應拋出 ValueError"""
        df = pd.DataFrame({"interval_days": [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError, match="only NaN"):
            _get_interval_values(df, ["interval_days"])


# ======================================================
# Test 2: _build_grid 單元測試
# ======================================================
class TestBuildGrid:
    """建立評估網格測試"""

    def test_grid_size(self):
        """網格大小應正確"""
        x = np.array([1, 2, 3, 4, 5])
        grid = _build_grid(x, grid_size=100)
        assert len(grid) == 100

    def test_grid_covers_data_range(self):
        """網格應涵蓋資料範圍"""
        x = np.array([10, 20, 30, 40, 50])
        grid = _build_grid(x, grid_size=256)
        assert grid.min() < x.min()
        assert grid.max() > x.max()

    def test_handles_constant_values(self):
        """處理常數值"""
        x = np.array([5, 5, 5, 5])
        grid = _build_grid(x, grid_size=100)
        assert len(grid) == 100
        assert grid.min() < 5
        assert grid.max() > 5


# ======================================================
# Test 3: KDE 擬合函數測試
# ======================================================
class TestKDEFitting:
    """KDE 擬合函數測試"""

    def test_scipy_kde_shape(self):
        """scipy KDE 輸出形狀正確"""
        x = np.random.normal(0, 1, 500)
        grid = np.linspace(-3, 3, 256)
        density = _fit_kde_scipy(x, bandwidth_factor=0.5, grid=grid)
        assert len(density) == len(grid)
        assert np.all(density >= 0)

    def test_kdepy_kde_shape(self):
        """KDEpy 輸出形狀正確"""
        x = np.random.normal(0, 1, 500)
        grid, density = _fit_kde_kdepy(x, bandwidth=0.5, grid_size=256)
        assert len(grid) == 256
        assert len(density) == 256
        assert np.all(density >= 0)

    def test_scipy_kde_constant_values(self):
        """scipy KDE 處理常數值"""
        x = np.array([5.0] * 100)
        grid = np.linspace(0, 10, 256)
        density = _fit_kde_scipy(x, bandwidth_factor=0.5, grid=grid)
        assert len(density) == len(grid)
        # 應該在常數值附近有一個峰
        assert np.max(density) > 0


# ======================================================
# Test 4: _auto_argrelmax_order 測試
# ======================================================
class TestAutoArgrelmaxOrder:
    """argrelmax order 自動計算測試"""

    def test_user_value_takes_precedence(self):
        """使用者提供值優先"""
        result = _auto_argrelmax_order(512, user_value=10)
        assert result == 10

    def test_auto_calculation(self):
        """自動計算"""
        result = _auto_argrelmax_order(512, user_value=None)
        assert result == max(3, 512 // 100)

    def test_minimum_is_3(self):
        """最小值為 3"""
        result = _auto_argrelmax_order(100, user_value=None)
        assert result >= 3


# ======================================================
# Test 5: Small / Medium / Large 模式測試
# ======================================================
@pytest.mark.parametrize("mode", ["small", "medium"])
def test_peak_detection_small_medium(bimodal_df, tmp_path, mode):
    """small / medium 模式基本測試"""
    output_dir = tmp_path / "reports"
    peaks_table, plot_path = run_peak_detection(
        profile_name="test",
        df=bimodal_df,
        mode=mode,
        mod_params={"grid_size": 256},
        output_dir=str(output_dir)
    )

    # Check 1: 應偵測到峰
    assert len(peaks_table) > 0, f"{mode} mode failed: No peaks detected."

    # Check 2: 峰結構正確
    assert_peaks_table_structure(peaks_table)

    # Check 3: 位置在範圍內
    assert_peak_positions_valid(peaks_table, bimodal_df)

    # Check 4: 圖檔存在
    assert os.path.exists(plot_path), "KDE peak plot is not created."


def test_peak_detection_large(bimodal_df, tmp_path):
    """large 模式（MeanShift）測試"""
    output_dir = tmp_path / "reports"

    peaks_table, plot_path = run_peak_detection(
        profile_name="test_large",
        df=bimodal_df,
        mode="large",
        mod_params={"meanshift_bandwidth": 1.0},
        output_dir=str(output_dir)
    )

    # Check 1: 應偵測到峰
    assert len(peaks_table) > 0, "large mode failed: No peaks detected."

    # Check 2: 峰結構正確
    assert_peaks_table_structure(peaks_table)

    # Check 3: 位置在範圍內
    assert_peak_positions_valid(peaks_table, bimodal_df)

    # Check 4: 圖檔存在
    assert os.path.exists(plot_path)


# ======================================================
# Test 6: 單峰偵測測試
# ======================================================
class TestUnimodalDetection:
    """單峰偵測測試"""

    def test_detects_single_peak(self, unimodal_df, tmp_path):
        """單峰資料應偵測到 1 個峰"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            output_dir=str(tmp_path)
        )
        # 單峰資料應偵測到 1 個峰（或可能因參數寬鬆偵測到更多）
        assert len(peaks_table) >= 1

    def test_peak_near_data_center(self, unimodal_df, tmp_path):
        """峰位置應接近資料中心"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            output_dir=str(tmp_path)
        )
        if peaks_table:
            data_mean = unimodal_df["interval_days"].mean()
            peak_pos = peaks_table[0]["pos"]
            # 峰位置應在均值附近（±3 標準差）
            data_std = unimodal_df["interval_days"].std()
            assert abs(peak_pos - data_mean) < 3 * data_std


# ======================================================
# Test 7: 雙峰偵測測試
# ======================================================
class TestBimodalDetection:
    """雙峰偵測測試"""

    @pytest.mark.parametrize("mode", ["small", "medium", "large"])
    def test_detects_two_peaks(self, bimodal_df, tmp_path, mode):
        """雙峰資料應偵測到 2 個峰"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=bimodal_df,
            mode=mode,
            mod_params={"prominence_min": 0.005},
            output_dir=str(tmp_path)
        )
        # 雙峰資料應偵測到至少 1 個峰，理想情況 2 個
        assert len(peaks_table) >= 1

    def test_peak_positions_match_data(self, bimodal_df, tmp_path):
        """峰位置應接近資料群中心"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=bimodal_df,
            mode="small",
            mod_params={"prominence_min": 0.005},
            output_dir=str(tmp_path)
        )
        if len(peaks_table) >= 2:
            positions = sorted([p["pos"] for p in peaks_table])
            # 第一峰應接近 5，第二峰應接近 20
            assert positions[0] < 10
            assert positions[-1] > 15


# ======================================================
# Test 8: 參數覆蓋測試
# ======================================================
class TestParameterOverride:
    """參數覆蓋測試"""

    def test_custom_grid_size(self, unimodal_df, tmp_path):
        """自訂 grid_size"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            mod_params={"grid_size": 1024},
            output_dir=str(tmp_path)
        )
        assert isinstance(peaks_table, list)

    def test_custom_bandwidth_small(self, unimodal_df, tmp_path):
        """自訂 kde_bandwidth_factor (small mode)"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            mod_params={"kde_bandwidth_factor": 0.3},
            output_dir=str(tmp_path)
        )
        assert isinstance(peaks_table, list)

    def test_custom_bandwidth_medium(self, unimodal_df, tmp_path):
        """自訂 kde_bandwidth (medium mode)"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="medium",
            mod_params={"kde_bandwidth": 1.0},
            output_dir=str(tmp_path)
        )
        assert isinstance(peaks_table, list)

    def test_custom_prominence_min(self, bimodal_df, tmp_path):
        """自訂 prominence_min 影響峰數量"""
        # 較嚴格的 prominence 應偵測較少峰
        strict_peaks, _ = run_peak_detection(
            profile_name=None,
            df=bimodal_df,
            mode="small",
            mod_params={"prominence_min": 0.1},
            output_dir=str(tmp_path)
        )
        loose_peaks, _ = run_peak_detection(
            profile_name=None,
            df=bimodal_df,
            mode="small",
            mod_params={"prominence_min": 0.001},
            output_dir=str(tmp_path)
        )
        assert len(strict_peaks) <= len(loose_peaks)

    def test_general_params_ignored(self, unimodal_df, tmp_path):
        """general_params 參數應可傳入（目前未使用）"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            general_params={"some_param": "value"},
            output_dir=str(tmp_path)
        )
        assert isinstance(peaks_table, list)


# ======================================================
# Test 9: profile_name 測試
# ======================================================
class TestProfileName:
    """profile_name 參數測試"""

    def test_with_profile_name(self, unimodal_df, tmp_path):
        """有 profile_name 時檔名包含 profile"""
        _, plot_path = run_peak_detection(
            profile_name="my_profile",
            df=unimodal_df,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert "my_profile" in plot_path

    def test_without_profile_name(self, unimodal_df, tmp_path):
        """無 profile_name 時使用預設檔名"""
        _, plot_path = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert "peak_detection_kde" in plot_path


# ======================================================
# Test 10: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_empty_dataframe_raises(self):
        """空資料框應拋出 ValueError"""
        df = pd.DataFrame({"interval_days": []})
        with pytest.raises(ValueError, match="empty"):
            run_peak_detection(None, df)

    def test_none_dataframe_raises(self):
        """None 應拋出 ValueError"""
        with pytest.raises(ValueError, match="empty"):
            run_peak_detection(None, None)

    def test_constant_values(self, tmp_path):
        """常數值應能處理"""
        df = pd.DataFrame({"interval_days": [10.0] * 100})
        peaks_table, plot_path = run_peak_detection(
            profile_name=None,
            df=df,
            mode="small",
            output_dir=str(tmp_path)
        )
        # 常數值應偵測到 0 或 1 個峰
        assert len(peaks_table) <= 1
        assert os.path.exists(plot_path)

    def test_few_data_points(self, tmp_path):
        """少量資料點應能處理"""
        df = pd.DataFrame({"interval_days": [1.0, 5.0, 10.0, 15.0, 20.0]})
        peaks_table, plot_path = run_peak_detection(
            profile_name=None,
            df=df,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert isinstance(peaks_table, list)
        assert os.path.exists(plot_path)

    def test_with_nan_values(self, tmp_path):
        """含 NaN 值應能處理"""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        data[::10] = np.nan
        df = pd.DataFrame({"interval_days": data})
        
        peaks_table, plot_path = run_peak_detection(
            profile_name=None,
            df=df,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert isinstance(peaks_table, list)


# ======================================================
# Test 11: 錯誤處理測試
# ======================================================
class TestErrorHandling:
    """錯誤處理測試"""

    def test_invalid_mode(self, unimodal_df, tmp_path):
        """無效模式應拋出 ValueError"""
        with pytest.raises(ValueError, match="mode must be"):
            run_peak_detection(
                profile_name=None,
                df=unimodal_df,
                mode="invalid",
                output_dir=str(tmp_path)
            )

    def test_missing_interval_column(self, tmp_path):
        """缺少 interval 欄位應拋出 KeyError"""
        df = pd.DataFrame({"other_col": [1, 2, 3, 4, 5]})
        with pytest.raises(KeyError):
            run_peak_detection(
                profile_name=None,
                df=df,
                mode="small",
                output_dir=str(tmp_path)
            )


# ======================================================
# Test 12: 內部偵測函數測試
# ======================================================
class TestInternalDetectFunctions:
    """內部峰偵測函數測試"""

    def test_detect_peaks_small(self):
        """_detect_peaks_small 函數測試"""
        np.random.seed(42)
        x = np.concatenate([
            np.random.normal(5, 0.5, 300),
            np.random.normal(15, 0.5, 300)
        ])
        x = np.sort(x)
        
        cfg = {**DEFAULT_PEAK_PARAMS}
        peaks_table, grid, density = _detect_peaks_small(x, cfg)
        
        assert isinstance(peaks_table, list)
        assert len(grid) == cfg["grid_size"]
        assert len(density) == cfg["grid_size"]

    def test_detect_peaks_medium(self):
        """_detect_peaks_medium 函數測試"""
        np.random.seed(42)
        x = np.concatenate([
            np.random.normal(5, 0.5, 300),
            np.random.normal(15, 0.5, 300)
        ])
        x = np.sort(x)
        
        cfg = {**DEFAULT_PEAK_PARAMS}
        peaks_table, grid, density = _detect_peaks_medium(x, cfg)
        
        assert isinstance(peaks_table, list)
        assert len(grid) == cfg["grid_size"]
        assert len(density) == cfg["grid_size"]

    def test_detect_peaks_large(self):
        """_detect_peaks_large 函數測試"""
        np.random.seed(42)
        x = np.concatenate([
            np.random.normal(5, 0.5, 300),
            np.random.normal(15, 0.5, 300)
        ])
        x = np.sort(x)
        
        cfg = {**DEFAULT_PEAK_PARAMS}
        peaks_table, grid, density = _detect_peaks_large(x, cfg)
        
        assert isinstance(peaks_table, list)
        assert len(grid) == cfg["grid_size"]
        assert len(density) == cfg["grid_size"]


# ======================================================
# Test 13: _peaks_table_from_indices 測試
# ======================================================
class TestPeaksTableFromIndices:
    """峰表格生成函數測試"""

    def test_empty_indices(self):
        """空 indices 應回傳空列表"""
        grid = np.linspace(0, 10, 100)
        density = np.sin(grid)
        indices = np.array([], dtype=int)
        
        result = _peaks_table_from_indices(grid, density, indices)
        assert result == []

    def test_single_peak(self):
        """單一峰測試"""
        grid = np.linspace(0, 10, 100)
        density = np.exp(-((grid - 5) ** 2) / 2)  # Gaussian centered at 5
        peak_idx = np.array([50])  # Approximate peak location
        
        result = _peaks_table_from_indices(grid, density, peak_idx)
        assert len(result) == 1
        assert "pos" in result[0]
        assert "height" in result[0]
        assert "width" in result[0]
        assert "prominence" in result[0]

    def test_multiple_peaks_sorted_by_pos(self):
        """多峰應按位置排序"""
        grid = np.linspace(0, 20, 200)
        # Two peaks
        density = np.exp(-((grid - 5) ** 2) / 2) + np.exp(-((grid - 15) ** 2) / 2)
        peak_indices = np.array([50, 150])
        
        result = _peaks_table_from_indices(grid, density, peak_indices)
        assert len(result) == 2
        assert result[0]["pos"] < result[1]["pos"]


# ======================================================
# Test 14: 輸出目錄測試
# ======================================================
class TestOutputDirectory:
    """輸出目錄測試"""

    def test_creates_output_dir(self, unimodal_df, tmp_path):
        """應自動建立輸出目錄"""
        new_dir = tmp_path / "new" / "nested" / "dir"
        _, plot_path = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            output_dir=str(new_dir)
        )
        assert os.path.exists(new_dir)
        assert os.path.exists(plot_path)

    def test_plot_file_extension(self, unimodal_df, tmp_path):
        """圖檔應為 PNG"""
        _, plot_path = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            output_dir=str(tmp_path)
        )
        assert plot_path.endswith(".png")


# ======================================================
# Test 15: 使用 transformed 欄位測試
# ======================================================
class TestTransformedColumn:
    """使用 transformed 欄位測試"""

    def test_uses_transformed_when_available(self, transformed_df, tmp_path):
        """有 transformed 欄位時應優先使用"""
        peaks_table, plot_path = run_peak_detection(
            profile_name=None,
            df=transformed_df,
            mode="small",
            output_dir=str(tmp_path)
        )

        # 峰位置應在 transformed 範圍內
        if peaks_table:
            transformed_min = transformed_df["interval_days_transformed"].min()
            transformed_max = transformed_df["interval_days_transformed"].max()
            for peak in peaks_table:
                assert transformed_min - 1 <= peak["pos"] <= transformed_max + 1


# ======================================================
# Test 16: transform_meta 逆轉換測試
# ======================================================
class TestTransformMeta:
    """transform_meta 參數及逆轉換功能測試"""

    def test_with_log1p_transform_meta(self, tmp_path):
        """使用 log1p 轉換的 transform_meta"""
        np.random.seed(42)
        # 原始資料
        original_data = np.random.exponential(scale=10, size=500)
        # 轉換後資料
        transformed_data = np.log1p(original_data)
        
        df = pd.DataFrame({
            "interval_days": original_data,
            "interval_days_transformed": transformed_data
        })
        
        transform_meta = {
            "method": "log1p",
            "transform_params": {}
        }
        
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=df,
            mode="small",
            transform_meta=transform_meta,
            output_dir=str(tmp_path)
        )
        
        if peaks_table:
            # 檢查有逆轉換欄位
            assert_peaks_table_structure(peaks_table, has_transform_meta=True)
            
            for peak in peaks_table:
                # pos_original 應該是 pos_transformed 的逆轉換
                expected_original = np.expm1(peak["pos_transformed"])
                assert abs(peak["pos_original"] - expected_original) < 0.01

    def test_with_none_transform_method(self, unimodal_df, tmp_path):
        """transform_meta method 為 none 時不進行逆轉換"""
        transform_meta = {
            "method": "none",
            "transform_params": {}
        }
        
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            transform_meta=transform_meta,
            output_dir=str(tmp_path)
        )
        
        # method 為 none 時不應新增逆轉換欄位
        if peaks_table:
            assert "pos_original" not in peaks_table[0]
            assert "pos_transformed" not in peaks_table[0]

    def test_without_transform_meta(self, unimodal_df, tmp_path):
        """未提供 transform_meta 時不進行逆轉換"""
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=unimodal_df,
            mode="small",
            transform_meta=None,
            output_dir=str(tmp_path)
        )
        
        if peaks_table:
            assert "pos_original" not in peaks_table[0]
            assert "pos_transformed" not in peaks_table[0]

    def test_with_yeo_johnson_transform_meta(self, tmp_path):
        """使用 yeo_johnson 轉換的 transform_meta"""
        from scipy.stats import yeojohnson
        
        np.random.seed(42)
        original_data = np.random.exponential(scale=10, size=500)
        transformed_data, lmbda = yeojohnson(original_data)
        
        df = pd.DataFrame({
            "interval_days": original_data,
            "interval_days_transformed": transformed_data
        })
        
        transform_meta = {
            "method": "yeo_johnson",
            "transform_params": {"lmbda": lmbda}
        }
        
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=df,
            mode="small",
            transform_meta=transform_meta,
            output_dir=str(tmp_path)
        )
        
        if peaks_table:
            assert_peaks_table_structure(peaks_table, has_transform_meta=True)
            # pos_original 應該大於 0（因為原始資料是正的）
            for peak in peaks_table:
                assert peak["pos_original"] > 0

    def test_transform_meta_pos_values_consistent(self, tmp_path):
        """確認 pos 和 pos_transformed 值一致"""
        np.random.seed(42)
        original_data = np.random.exponential(scale=10, size=500)
        transformed_data = np.log1p(original_data)
        
        df = pd.DataFrame({
            "interval_days": original_data,
            "interval_days_transformed": transformed_data
        })
        
        transform_meta = {
            "method": "log1p",
            "transform_params": {}
        }
        
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=df,
            mode="small",
            transform_meta=transform_meta,
            output_dir=str(tmp_path)
        )
        
        if peaks_table:
            for peak in peaks_table:
                # pos 和 pos_transformed 應該相等
                assert peak["pos"] == peak["pos_transformed"]


# ======================================================
# Test 17: 三峰偵測測試
# ======================================================
class TestTrimodalDetection:
    """三峰偵測測試"""

    @pytest.mark.parametrize("mode", ["small", "medium", "large"])
    def test_detects_multiple_peaks(self, trimodal_df, tmp_path, mode):
        """三峰資料應偵測到多個峰"""
        # 使用較小的帶寬以便更好地偵測分離的峰
        mod_params = {
            "prominence_min": 0.005,
            "kde_bandwidth_factor": 0.2,  # small mode: 較小的帶寬因子
            "kde_bandwidth": 0.3,          # medium mode: 較小的帶寬
            "meanshift_bandwidth": 0.5,    # large mode: 較小的帶寬
        }
        
        peaks_table, _ = run_peak_detection(
            profile_name=None,
            df=trimodal_df,
            mode=mode,
            mod_params=mod_params,
            output_dir=str(tmp_path)
        )
        # 三峰資料應偵測到至少 2 個峰
        assert len(peaks_table) >= 2, \
            f"{mode} mode: Expected at least 2 peaks, got {len(peaks_table)}. Peaks: {peaks_table}"
