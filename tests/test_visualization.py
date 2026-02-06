# tests/test_visualization.py
# ======================================================
# Module: 03_visualization - pytest
# Spec 對齊：
# - Input : transformed_df (含 interval_days_transformed)
# - Output: plots_dir (資料夾路徑), summary_stats (n / mean / median / std / skew)
# - 支援 mode: small / medium / large / auto
# - raincloud plot 需要 cat 欄位
# ======================================================

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from repurchase_cycle.modules.visualization import (
    run_visualization,
    _auto_select_mode,
    _get_working_series,
    _build_approx_series_large,
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def default_general_params() -> Dict[str, Any]:
    """
    預設的 general_params，包含資料量門檻與隨機種子。
    """
    return {
        "data_size_thresholds": {
            "small": int(1e4),
            "medium": int(1e6)
        },
        "random_seed": 42
    }


@pytest.fixture
def small_df() -> pd.DataFrame:
    """
    建立一個簡單、可計算「確切統計量」的小型 DataFrame。
    interval_days_transformed: 1, 2, 3, 4, 5
    """
    return pd.DataFrame({
        "interval_days_transformed": [1.0, 2.0, 3.0, 4.0, 5.0]
    })


@pytest.fixture
def random_df() -> pd.DataFrame:
    """
    建立較貼近真實情境的隨機 DataFrame。
    """
    rng = np.random.default_rng(42)
    data = rng.gamma(shape=2.0, scale=5.0, size=1000)
    return pd.DataFrame({
        "interval_days_transformed": data
    })


@pytest.fixture
def raincloud_df() -> pd.DataFrame:
    """
    建立適合 raincloud plot 的 DataFrame，包含 cat 分組資料。
    """
    rng = np.random.default_rng(42)
    n_per_group = 200
    
    group_a = rng.gamma(shape=2.0, scale=5.0, size=n_per_group)
    group_b = rng.gamma(shape=3.0, scale=7.0, size=n_per_group)
    
    df = pd.DataFrame({
        "interval_days_transformed": np.concatenate([group_a, group_b]),
        "cat": ["Group A"] * n_per_group + ["Group B"] * n_per_group
    })
    
    return df


# -----------------------------
# Unit Tests for Helper Functions
# -----------------------------
class TestAutoSelectMode:
    """測試 _auto_select_mode 函數"""
    
    def test_explicit_small_mode(self):
        """明確指定 small mode 時應直接使用"""
        result = _auto_select_mode(n_rows=100000, explicit_mode="small")
        assert result == "small"
    
    def test_explicit_medium_mode(self):
        """明確指定 medium mode 時應直接使用"""
        result = _auto_select_mode(n_rows=100, explicit_mode="medium")
        assert result == "medium"
    
    def test_explicit_large_mode(self):
        """明確指定 large mode 時應直接使用"""
        result = _auto_select_mode(n_rows=100, explicit_mode="large")
        assert result == "large"
    
    def test_auto_mode_selects_small(self):
        """auto mode 在小資料量時選擇 small"""
        result = _auto_select_mode(n_rows=5000, explicit_mode="auto")
        assert result == "small"
    
    def test_auto_mode_selects_medium(self):
        """auto mode 在中等資料量時選擇 medium"""
        result = _auto_select_mode(n_rows=500000, explicit_mode="auto")
        assert result == "medium"
    
    def test_auto_mode_selects_large(self):
        """auto mode 在大資料量時選擇 large"""
        result = _auto_select_mode(n_rows=5000000, explicit_mode="auto")
        assert result == "large"
    
    def test_custom_thresholds(self):
        """使用自訂門檻值"""
        result = _auto_select_mode(
            n_rows=500,
            explicit_mode="auto",
            small_threshold=100,
            medium_threshold=1000
        )
        assert result == "medium"
    
    def test_unknown_mode_fallback_to_auto(self):
        """未知的 mode 應 fallback 到 auto 邏輯"""
        result = _auto_select_mode(n_rows=5000, explicit_mode="unknown_mode")
        assert result == "small"  # 5000 <= 1e4 -> small
    
    def test_empty_string_mode_fallback_to_auto(self):
        """空字串 mode 應 fallback 到 auto 邏輯"""
        result = _auto_select_mode(n_rows=500000, explicit_mode="")
        assert result == "medium"
    
    def test_none_mode_fallback_to_auto(self):
        """None mode 應 fallback 到 auto 邏輯"""
        result = _auto_select_mode(n_rows=5000000, explicit_mode=None)
        assert result == "large"


class TestBuildApproxSeriesLarge:
    """測試 _build_approx_series_large 函數"""
    
    def test_returns_series(self):
        """應回傳 pd.Series"""
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 10000))
        result = _build_approx_series_large(s)
        assert isinstance(result, pd.Series)
    
    def test_respects_max_points(self):
        """輸出大小應接近 max_points"""
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 100000))
        result = _build_approx_series_large(s, max_points=1000)
        # 允許一定誤差
        assert len(result) <= 2000
    
    def test_empty_series(self):
        """空 Series 應回傳空 Series"""
        s = pd.Series([], dtype=float)
        result = _build_approx_series_large(s)
        assert len(result) == 0
    
    def test_handles_nan(self):
        """應正確處理 NaN 值"""
        s = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        result = _build_approx_series_large(s)
        assert not result.isna().any()
    
    def test_preserves_name(self):
        """應保留 Series 的 name 屬性"""
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 10000), name="test_col")
        result = _build_approx_series_large(s)
        assert result.name == "test_col"
    
    def test_custom_bins(self):
        """可使用自訂的 min_bins 和 max_bins"""
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 10000))
        result = _build_approx_series_large(s, min_bins=50, max_bins=50)
        assert isinstance(result, pd.Series)
        assert len(result) > 0


class TestGetWorkingSeries:
    """測試 _get_working_series 函數"""
    
    def test_small_mode_returns_full_data(self):
        """small mode 應回傳完整資料"""
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = _get_working_series(df, "col", "small", 0.5, 42)
        assert len(result) == 5
    
    def test_medium_mode_samples_data(self):
        """medium mode 應進行抽樣"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.normal(0, 1, 1000)})
        result = _get_working_series(df, "col", "medium", 0.1, 42)
        assert len(result) == 100  # 1000 * 0.1
    
    def test_large_mode_approximates(self):
        """large mode 應進行近似處理"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.normal(0, 1, 100000)})
        result = _get_working_series(df, "col", "large", 0.1, 42)
        # large mode 使用 histogram approximation，預設 max_points=5000
        assert len(result) <= 10000
    
    def test_missing_column_raises(self):
        """欄位不存在時應 raise ValueError"""
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            _get_working_series(df, "col", "small", 0.5, 42)
    
    def test_drops_nan(self):
        """應移除 NaN 值"""
        df = pd.DataFrame({"col": [1.0, np.nan, 3.0]})
        result = _get_working_series(df, "col", "small", 0.5, 42)
        assert len(result) == 2
        assert not result.isna().any()
    
    def test_medium_mode_zero_sample_ratio_fallback(self):
        """medium mode 當 sample_ratio <= 0 時應 fallback 到 0.05"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.normal(0, 1, 1000)})
        result = _get_working_series(df, "col", "medium", 0.0, 42)
        assert len(result) == 50  # 1000 * 0.05
    
    def test_medium_mode_clamps_sample_ratio(self):
        """medium mode 應將 sample_ratio 限制在 [0, 1]"""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"col": rng.normal(0, 1, 100)})
        result = _get_working_series(df, "col", "medium", 1.5, 42)
        assert len(result) == 100  # clipped to 1.0
    
    def test_unknown_mode_fallback_to_small(self):
        """未知的 mode 應 fallback 到 small"""
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = _get_working_series(df, "col", "unknown", 0.5, 42)
        assert len(result) == 5
    
    def test_empty_series_after_dropna(self):
        """全為 NaN 時應回傳空 Series"""
        df = pd.DataFrame({"col": [np.nan, np.nan]})
        result = _get_working_series(df, "col", "small", 0.5, 42)
        assert len(result) == 0


# -----------------------------
# Basic Functionality Tests
# -----------------------------
def test_run_visualization_basic_stats(tmp_path, small_df, default_general_params):
    """
    測試：
    - 在 small mode 下可正常執行
    - 回傳的 summary_stats 含 n / mean / median / std / skew
    - 統計量與預期相符
    """
    output_dir = tmp_path / "reports"

    plots_dir, summary_stats = run_visualization(
        df=small_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde", "violin", "box", "cdf"]},
        output_dir=str(output_dir),
    )

    # 1) plots_dir 要是一個存在的資料夾
    assert isinstance(plots_dir, str)
    assert os.path.isdir(plots_dir)

    # 2) summary_stats 結構與 key
    expected_keys = {"n", "mean", "median", "std", "skew"}
    assert set(summary_stats.keys()) == expected_keys

    # 3) 數值檢查
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    expected_n = 5.0
    expected_mean = values.mean()          # 3.0
    expected_median = np.median(values)    # 3.0
    expected_std = values.std(ddof=1)      # ~1.5811

    assert summary_stats["n"] == pytest.approx(expected_n, rel=1e-6)
    assert summary_stats["mean"] == pytest.approx(expected_mean, rel=1e-6)
    assert summary_stats["median"] == pytest.approx(expected_median, rel=1e-6)
    assert summary_stats["std"] == pytest.approx(expected_std, rel=1e-6)
    assert summary_stats["skew"] == pytest.approx(0.0, abs=1e-6)


def test_run_visualization_creates_plot_files(tmp_path, random_df, default_general_params):
    """
    測試：
    - 呼叫後指定的 plots_dir 中，至少產生一個圖檔
    - 確認繪圖流程有實際輸出
    """
    output_dir = tmp_path / "reports"

    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde", "violin", "box", "cdf"]},
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)

    image_files = [
        f
        for f in os.listdir(plots_dir)
        if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
    ]
    assert len(image_files) >= 1

    # summary_stats 型別檢查
    assert isinstance(summary_stats["n"], float)
    assert isinstance(summary_stats["mean"], float)
    assert isinstance(summary_stats["median"], float)
    assert isinstance(summary_stats["std"], float)
    assert isinstance(summary_stats["skew"], float)


def test_run_visualization_respects_custom_plot_types(tmp_path, random_df, default_general_params):
    """
    測試：
    - 當 plot_types 指定較少種類時，輸出的圖檔數量應較少
    """
    output_dir_full = tmp_path / "reports_full"
    plots_dir_full, _ = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde", "violin", "box", "cdf"]},
        output_dir=str(output_dir_full),
    )
    full_files = [
        f
        for f in os.listdir(plots_dir_full)
        if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
    ]

    output_dir_partial = tmp_path / "reports_partial"
    plots_dir_partial, _ = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde", "box"]},
        output_dir=str(output_dir_partial),
    )
    partial_files = [
        f
        for f in os.listdir(plots_dir_partial)
        if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
    ]

    assert len(partial_files) >= 1
    assert len(partial_files) <= len(full_files)


# -----------------------------
# Mode Selection Tests
# -----------------------------
def test_run_visualization_supports_auto_mode(tmp_path, random_df, default_general_params):
    """
    測試：
    - mode="auto" 時函數可正常執行，並回傳合法的輸出
    """
    output_dir = tmp_path / "reports_auto"
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="auto",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde"]},
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    assert set(summary_stats.keys()) == {"n", "mean", "median", "std", "skew"}


def test_run_visualization_medium_mode_sampling(tmp_path, default_general_params):
    """
    測試：
    - medium mode 會進行抽樣
    """
    rng = np.random.default_rng(42)
    large_data = rng.gamma(shape=2.0, scale=5.0, size=50000)
    df = pd.DataFrame({"interval_days_transformed": large_data})
    
    output_dir = tmp_path / "reports_medium"
    plots_dir, summary_stats = run_visualization(
        df=df,
        mode="medium",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["hist_kde"],
            "sample_ratio": 0.1
        },
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    image_files = [
        f for f in os.listdir(plots_dir)
        if f.lower().endswith(".png")
    ]
    assert len(image_files) >= 1
    
    # summary_stats 應該基於全量資料
    assert summary_stats["n"] == 50000.0


def test_run_visualization_large_mode(tmp_path, default_general_params):
    """
    測試：
    - large mode 使用 histogram approximation
    """
    rng = np.random.default_rng(42)
    large_data = rng.gamma(shape=2.0, scale=5.0, size=100000)
    df = pd.DataFrame({"interval_days_transformed": large_data})
    
    output_dir = tmp_path / "reports_large"
    plots_dir, summary_stats = run_visualization(
        df=df,
        mode="large",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde"]},
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    # summary_stats 應該基於全量資料
    assert summary_stats["n"] == 100000.0


# -----------------------------
# Prefix Tests
# -----------------------------
def test_run_visualization_with_prefix(tmp_path, random_df, default_general_params):
    """
    測試：
    - viz_prefix 參數可正確加入檔名前綴
    """
    output_dir = tmp_path / "reports_prefix"
    prefix = "test_category"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        viz_prefix=prefix,
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde", "violin"]},
        output_dir=str(output_dir),
    )

    image_files = os.listdir(plots_dir)
    prefixed_files = [f for f in image_files if f.startswith(f"{prefix}_")]
    
    assert len(prefixed_files) >= 1


def test_run_visualization_without_prefix(tmp_path, random_df, default_general_params):
    """
    測試：
    - 不設定 viz_prefix 時，檔名不應有額外前綴
    """
    output_dir = tmp_path / "reports_no_prefix"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde"]},
        output_dir=str(output_dir),
    )

    image_files = os.listdir(plots_dir)
    assert "interval_hist_kde.png" in image_files


# -----------------------------
# Error Handling Tests
# -----------------------------
def test_run_visualization_raises_when_column_missing(tmp_path, default_general_params):
    """
    測試：
    - 當 DataFrame 中沒有 interval_days_transformed 欄位時，應明確 raise 錯誤
    """
    df = pd.DataFrame({"other_col": [1, 2, 3]})

    with pytest.raises((KeyError, ValueError)):
        run_visualization(
            df=df,
            mode="small",
            general_params=default_general_params,
            mod_params={"plot_types": ["hist_kde"]},
            output_dir=str(tmp_path / "reports"),
        )


def test_run_visualization_raises_when_all_nan(tmp_path, default_general_params):
    """
    測試：
    - interval_days_transformed 全為 NaN 時，函數應明確拒絕並 raise error
    """
    df = pd.DataFrame({
        "interval_days_transformed": [np.nan, np.nan, np.nan]
    })

    with pytest.raises(ValueError, match="No non-null values"):
        run_visualization(
            df=df,
            mode="small",
            general_params=default_general_params,
            mod_params={"plot_types": ["hist_kde"]},
            output_dir=str(tmp_path / "reports"),
        )


# -----------------------------
# Raincloud Plot Tests
# -----------------------------
def test_run_visualization_raincloud_creates_file(tmp_path, raincloud_df, default_general_params):
    """
    測試：
    - 包含 raincloud 在 plot_types 時，應產生 interval_raincloud.png
    - 需要 cat 欄位才會繪製
    """
    output_dir = tmp_path / "reports_raincloud"
    
    plots_dir, summary_stats = run_visualization(
        df=raincloud_df,
        mode="small",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud"],
        },
        output_dir=str(output_dir),
    )

    raincloud_path = os.path.join(plots_dir, "interval_raincloud.png")
    assert os.path.isfile(raincloud_path), "Raincloud plot file not created"
    assert os.path.getsize(raincloud_path) > 0, "Raincloud plot file is empty"


def test_run_visualization_raincloud_skipped_without_cat(tmp_path, random_df, default_general_params):
    """
    測試：
    - 沒有 cat 欄位時，raincloud plot 應被跳過（不報錯）
    """
    output_dir = tmp_path / "reports_raincloud_no_cat"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,  # 沒有 cat 欄位
        mode="small",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud", "hist_kde"],
        },
        output_dir=str(output_dir),
    )

    # raincloud 應該不存在
    raincloud_path = os.path.join(plots_dir, "interval_raincloud.png")
    assert not os.path.isfile(raincloud_path)
    
    # 但其他圖應該存在
    hist_path = os.path.join(plots_dir, "interval_hist_kde.png")
    assert os.path.isfile(hist_path)


def test_run_visualization_raincloud_skipped_without_interval_days_transformed(tmp_path, default_general_params):
    """
    測試：
    - 有 cat 但沒有 interval_days_transformed 欄位時，應在前面就報錯
    - 這個情況會在 s_all = df["interval_days_transformed"].dropna() 時就失敗
    """
    df = pd.DataFrame({
        "cat": ["A", "B", "C"],
        "other_col": [1, 2, 3]
    })
    
    with pytest.raises((KeyError, ValueError)):
        run_visualization(
            df=df,
            mode="small",
            general_params=default_general_params,
            mod_params={"plot_types": ["raincloud"]},
            output_dir=str(tmp_path / "reports"),
        )


def test_run_visualization_raincloud_with_custom_params(tmp_path, raincloud_df, default_general_params):
    """
    測試：
    - raincloud plot 可接受自定義的 palette, sigma, orient 等參數
    """
    output_dir = tmp_path / "reports_raincloud_custom"
    
    plots_dir, summary_stats = run_visualization(
        df=raincloud_df,
        mode="small",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud"],
            "palette": "viridis",
            "sigma": 0.3,
            "orient": "v",
        },
        output_dir=str(output_dir),
    )

    raincloud_path = os.path.join(plots_dir, "interval_raincloud.png")
    assert os.path.isfile(raincloud_path)
    assert os.path.getsize(raincloud_path) > 0


def test_run_visualization_raincloud_with_hue(tmp_path, default_general_params):
    """
    測試：
    - raincloud plot 可使用 data_hue 參數進行分組著色
    """
    rng = np.random.default_rng(42)
    n_per_group = 100
    
    df = pd.DataFrame({
        "interval_days_transformed": np.concatenate([
            rng.gamma(2, 5, n_per_group),
            rng.gamma(3, 7, n_per_group)
        ]),
        "cat": ["A"] * n_per_group + ["B"] * n_per_group,
        "subtype": (["Type1", "Type2"] * n_per_group)
    })
    
    output_dir = tmp_path / "reports_raincloud_hue"
    
    plots_dir, summary_stats = run_visualization(
        df=df,
        mode="small",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud"],
            "data_hue": "subtype",
        },
        output_dir=str(output_dir),
    )

    raincloud_path = os.path.join(plots_dir, "interval_raincloud.png")
    assert os.path.isfile(raincloud_path)


def test_run_visualization_raincloud_with_other_plots(tmp_path, raincloud_df, default_general_params):
    """
    測試：
    - raincloud 可與其他圖型同時使用
    """
    output_dir = tmp_path / "reports_mixed"
    
    plots_dir, summary_stats = run_visualization(
        df=raincloud_df,
        mode="small",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud", "hist_kde", "violin"],
        },
        output_dir=str(output_dir),
    )

    image_files = [
        f
        for f in os.listdir(plots_dir)
        if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
    ]
    
    assert len(image_files) >= 3
    assert "interval_raincloud.png" in image_files
    assert "interval_hist_kde.png" in image_files
    assert "interval_violin.png" in image_files


def test_run_visualization_raincloud_with_prefix(tmp_path, raincloud_df, default_general_params):
    """
    測試：
    - raincloud plot 檔名也應遵循 viz_prefix 設定
    """
    output_dir = tmp_path / "reports_raincloud_prefix"
    prefix = "category_A"
    
    plots_dir, summary_stats = run_visualization(
        df=raincloud_df,
        mode="small",
        viz_prefix=prefix,
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud"],
        },
        output_dir=str(output_dir),
    )

    expected_filename = f"{prefix}_interval_raincloud.png"
    raincloud_path = os.path.join(plots_dir, expected_filename)
    assert os.path.isfile(raincloud_path)


def test_run_visualization_raincloud_only(tmp_path, raincloud_df, default_general_params):
    """
    測試：
    - 只繪製 raincloud plot 時的行為
    """
    output_dir = tmp_path / "reports_raincloud_only"
    
    plots_dir, summary_stats = run_visualization(
        df=raincloud_df,
        mode="small",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud"],
        },
        output_dir=str(output_dir),
    )

    image_files = [
        f
        for f in os.listdir(plots_dir)
        if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
    ]
    
    assert len(image_files) == 1
    assert image_files[0] == "interval_raincloud.png"
    
    assert set(summary_stats.keys()) == {"n", "mean", "median", "std", "skew"}


def test_run_visualization_raincloud_uses_full_dataframe(tmp_path, default_general_params):
    """
    測試：
    - raincloud plot 使用完整的 DataFrame（不受 mode 影響）
    - 即使在 medium/large mode 下，raincloud 仍應正常運作
    """
    rng = np.random.default_rng(42)
    n_per_group = 5000
    
    df = pd.DataFrame({
        "interval_days_transformed": np.concatenate([
            rng.gamma(2, 5, n_per_group),
            rng.gamma(3, 7, n_per_group)
        ]),
        "cat": ["A"] * n_per_group + ["B"] * n_per_group,
    })
    
    output_dir = tmp_path / "reports_raincloud_medium"
    
    plots_dir, summary_stats = run_visualization(
        df=df,
        mode="medium",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["raincloud"],
            "sample_ratio": 0.1,
        },
        output_dir=str(output_dir),
    )

    raincloud_path = os.path.join(plots_dir, "interval_raincloud.png")
    assert os.path.isfile(raincloud_path)
    # summary_stats 應基於全量資料
    assert summary_stats["n"] == 10000.0


# -----------------------------
# Integration Tests
# -----------------------------
def test_run_visualization_full_workflow(tmp_path, default_general_params):
    """
    整合測試：
    - 使用完整的參數組合
    - 驗證所有圖型都能正常產生
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "interval_days_transformed": rng.gamma(2, 5, 500),
        "cat": (["A", "B"] * 250)
    })
    
    output_dir = tmp_path / "reports_full"
    
    plots_dir, summary_stats = run_visualization(
        df=df,
        mode="small",
        viz_prefix="full_test",
        general_params=default_general_params,
        mod_params={
            "plot_types": ["hist_kde", "violin", "box", "cdf", "raincloud"],
            "kde_bandwidths": [0.3, 0.6, 1.0],
            "palette": "Set2",
            "sigma": 0.2,
        },
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    
    expected_files = [
        "full_test_interval_hist_kde.png",
        "full_test_interval_violin.png",
        "full_test_interval_box.png",
        "full_test_interval_cdf.png",
        "full_test_interval_raincloud.png"
    ]

    actual_files = os.listdir(plots_dir)
    for expected in expected_files:
        assert expected in actual_files, f"Missing expected file: {expected}"

    assert set(summary_stats.keys()) == {"n", "mean", "median", "std", "skew"}
    assert all(isinstance(v, float) for v in summary_stats.values())


def test_run_visualization_default_params(tmp_path, raincloud_df):
    """
    測試：
    - 使用預設參數（不傳入 general_params 和 mod_params）
    - 預設 plot_types 包含 ["hist_kde", "violin", "box", "cdf", "raincloud"]
    - raincloud_df 有 cat 欄位，所以 raincloud 也會產生
    """
    output_dir = tmp_path / "reports_default"
    
    plots_dir, summary_stats = run_visualization(
        df=raincloud_df,
        mode="small",
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    
    # 預設應該產生所有 5 種圖型（raincloud_df 有 cat 欄位）
    expected_files = [
        "interval_hist_kde.png",
        "interval_violin.png",
        "interval_box.png",
        "interval_cdf.png",
        "interval_raincloud.png"
    ]
    
    actual_files = os.listdir(plots_dir)
    for expected in expected_files:
        assert expected in actual_files, f"Missing expected file: {expected}"
    
    assert len(actual_files) == 5


def test_run_visualization_default_params_without_cat(tmp_path, random_df):
    """
    測試：
    - 使用預設參數但沒有 cat 欄位
    - 應產生 4 種圖型（排除 raincloud）
    """
    output_dir = tmp_path / "reports_default_no_cat"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    
    # 沒有 cat 欄位，raincloud 會被跳過
    expected_files = [
        "interval_hist_kde.png",
        "interval_violin.png",
        "interval_box.png",
        "interval_cdf.png",
    ]
    
    actual_files = os.listdir(plots_dir)
    for expected in expected_files:
        assert expected in actual_files, f"Missing expected file: {expected}"
    
    # raincloud 不應存在
    assert "interval_raincloud.png" not in actual_files
    assert len(actual_files) == 4


def test_run_visualization_output_dir_created(tmp_path, random_df, default_general_params):
    """
    測試：
    - 若 output_dir 不存在，應自動建立
    """
    output_dir = tmp_path / "new_dir" / "nested" / "reports"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde"]},
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    assert "interval_hist_kde.png" in os.listdir(plots_dir)


def test_run_visualization_empty_plot_types(tmp_path, random_df, default_general_params):
    """
    測試：
    - 空的 plot_types 列表時，不產生任何圖檔
    """
    output_dir = tmp_path / "reports_empty"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": []},
        output_dir=str(output_dir),
    )

    assert os.path.isdir(plots_dir)
    image_files = [
        f for f in os.listdir(plots_dir)
        if f.lower().endswith((".png", ".pdf", ".jpg", ".jpeg"))
    ]
    assert len(image_files) == 0
    
    # summary_stats 仍應正常計算
    assert set(summary_stats.keys()) == {"n", "mean", "median", "std", "skew"}


def test_run_visualization_visualization_subdir(tmp_path, random_df, default_general_params):
    """
    測試：
    - plots_dir 應該是 output_dir/visualization
    """
    output_dir = tmp_path / "reports"
    
    plots_dir, summary_stats = run_visualization(
        df=random_df,
        mode="small",
        general_params=default_general_params,
        mod_params={"plot_types": ["hist_kde"]},
        output_dir=str(output_dir),
    )

    expected_plots_dir = os.path.join(str(output_dir), "visualization")
    assert plots_dir == expected_plots_dir
    assert os.path.isdir(expected_plots_dir)
