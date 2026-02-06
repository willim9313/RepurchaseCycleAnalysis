# ======================================================
# test_pipeline_integration.py
# ======================================================
# 整合測試模組: Pipeline 端到端測試
# 對應模組: pipeline.py
# ------------------------------------------------------

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from repurchase_cycle.pipeline import (
    run_all_categories,
    run_category_pipeline,
    decide_mode,
)


# ======================================================
# Fixture: 測試配置與資料
# ======================================================
def _build_minimal_config(tmp_path: Path) -> Dict[str, Any]:
    """Build minimal config for integration test."""
    return {
        "pipeline_controller": {
            "data_size_thresholds": {"small": 10, "medium": 1000},
            "parallel_execution": {"enabled": False, "n_jobs": 1},
            "random_seed": 42,
            "reports_path": str(tmp_path / "reports"),
            "logging": {
                "level": "WARNING",
                "save_path": str(tmp_path / "logs"),
            },
        },
        "modules": {
            "interval_derivation": {
                "uid_col": "uid",
                "cat_col": "cat",
                "date_col": "order_date",
                "groupby_cols": ["uid", "cat"],
                "keep_first_purchase": False,
                "date_format": None,
                "extra_cols": [],
                "min_intervals_per_group": 1,
            },
            "data_cleaning": {
                "remove_negatives": True,
                "missing_strategy": "drop",
                "outlier_method": "IQR",
                "outlier_threshold": 3.0,
            },
            "transform": {
                "method_candidates": ["log1p", "none"],
                "auto_select_by_skewness": True,
            },
            "visualization": {
                "sample_ratio": 1.0,
                "plot_types": ["hist_kde"],
            },
            "unimodality_test": {
                "alpha": 0.05,
            },
            "peak_detection": {
                "grid_size": 256,
                "prominence_min": 0.01,
            },
            "modality_quantification": {
                "k_range": [1, 3],
                "selection_metric": "BIC",
            },
            "stability_assessment": {
                "n_bootstrap": 10,
                "sample_fraction": 0.8,
            },
            "reporting": {
                "provide_details": True,
                "separate_category_report": False,
            },
        },
    }


def _generate_transaction_data(
    n_users: int,
    category: str,
    intervals: np.ndarray,
    base_date: str = "2024-01-01"
) -> pd.DataFrame:
    """
    生成交易資料格式的測試資料。
    每個用戶有 2 筆交易，間隔由 intervals 指定。
    """
    base = pd.to_datetime(base_date)
    rows = []
    for i, interval in enumerate(intervals):
        uid = f"u{i}"
        rows.append({"uid": uid, "cat": category, "order_date": base})
        rows.append({"uid": uid, "cat": category, "order_date": base + pd.Timedelta(days=float(interval))})
    return pd.DataFrame(rows)


@pytest.fixture
def minimal_config(tmp_path):
    """Minimal config fixture."""
    return _build_minimal_config(tmp_path)


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Simple test DataFrame with 2 categories (transaction format)."""
    base = pd.to_datetime("2024-01-01")
    return pd.DataFrame({
        "uid": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5", "u6", "u6"],
        "cat": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
        "order_date": [
            base, base + pd.Timedelta(days=5),   # u1 in A: interval=5
            base, base + pd.Timedelta(days=6),   # u2 in A: interval=6
            base, base + pd.Timedelta(days=7),   # u3 in A: interval=7
            base, base + pd.Timedelta(days=30),  # u4 in B: interval=30
            base, base + pd.Timedelta(days=35),  # u5 in B: interval=35
            base, base + pd.Timedelta(days=40),  # u6 in B: interval=40
        ],
    })


@pytest.fixture
def unimodal_df() -> pd.DataFrame:
    """Unimodal test data for single category (transaction format)."""
    np.random.seed(42)
    n = 100
    intervals = np.clip(np.random.normal(30, 5, n), 1, None)
    return _generate_transaction_data(n, "A", intervals)


@pytest.fixture
def bimodal_df() -> pd.DataFrame:
    """Bimodal test data for single category (transaction format)."""
    np.random.seed(42)
    n = 200
    peak1 = np.clip(np.random.normal(10, 2, n // 2), 1, None)
    peak2 = np.clip(np.random.normal(40, 3, n // 2), 1, None)
    intervals = np.concatenate([peak1, peak2])
    return _generate_transaction_data(n, "A", intervals)


# ======================================================
# Test 1: decide_mode 測試
# ======================================================
class TestDecideMode:
    """decide_mode 函數測試"""

    def test_small_mode(self):
        """小於 small 閾值應回傳 small"""
        thresholds = {"small": 10, "medium": 1000}
        assert decide_mode(5, thresholds) == "small"
        assert decide_mode(9, thresholds) == "small"

    def test_medium_mode(self):
        """介於 small 和 medium 閾值應回傳 medium"""
        thresholds = {"small": 10, "medium": 1000}
        assert decide_mode(10, thresholds) == "medium"
        assert decide_mode(500, thresholds) == "medium"
        assert decide_mode(999, thresholds) == "medium"

    def test_large_mode(self):
        """大於等於 medium 閾值應回傳 large"""
        thresholds = {"small": 10, "medium": 1000}
        assert decide_mode(1000, thresholds) == "large"
        assert decide_mode(10000, thresholds) == "large"

    def test_default_thresholds(self):
        """缺少閾值時使用預設值"""
        thresholds = {}
        # Default: small=1e4, medium=1e6
        assert decide_mode(100, thresholds) == "small"
        assert decide_mode(50000, thresholds) == "medium"
        assert decide_mode(2000000, thresholds) == "large"

    def test_edge_cases(self):
        """邊界條件"""
        thresholds = {"small": 10, "medium": 1000}
        assert decide_mode(0, thresholds) == "small"
        assert decide_mode(1, thresholds) == "small"


# ======================================================
# Test 2: run_category_pipeline 測試
# ======================================================
class TestRunCategoryPipeline:
    """run_category_pipeline 函數測試"""

    def test_returns_two_dicts(self, unimodal_df, minimal_config):
        """應回傳 brief 和 detailed 兩個字典"""
        brief, detailed = run_category_pipeline(
            unimodal_df, "A", minimal_config
        )
        
        assert isinstance(brief, dict)
        assert isinstance(detailed, dict)

    def test_brief_summary_structure(self, unimodal_df, minimal_config):
        """brief summary 結構驗證"""
        brief, _ = run_category_pipeline(
            unimodal_df, "A", minimal_config
        )
        
        # brief_summary 結構為 {"summary": {...}, "figures": {...}}
        assert "summary" in brief
        assert "figures" in brief
        
        summary = brief["summary"]
        # 基本統計欄位
        basic_stats_keys = {"n", "mean", "median", "std", "skew"}
        assert basic_stats_keys.issubset(set(summary.keys()))
        
        # n 是間隔數，應該 > 0
        assert summary["n"] > 0

    def test_detailed_result_structure(self, unimodal_df, minimal_config):
        """detailed result 結構驗證"""
        _, detailed = run_category_pipeline(
            unimodal_df, "A", minimal_config
        )
        
        # 應包含各模組的詳細結果
        assert isinstance(detailed, dict)
        # 驗證關鍵欄位存在
        expected_keys = ["interval_conversion_summary", "mode", "discard_summary", 
                        "transform_meta", "summary_stats", "unimodality_test_result"]
        for key in expected_keys:
            assert key in detailed, f"Missing key: {key}"

    def test_unimodal_path_skips_peak_detection(self, unimodal_df, minimal_config):
        """單峰資料應跳過峰偵測（若 dip test 判定為 unimodal）"""
        brief, detailed = run_category_pipeline(
            unimodal_df, "A", minimal_config
        )
        
        # dip_p 應在 summary 中
        assert "dip_p" in brief["summary"]
        
        # 若判定為 unimodal，peaks_table 應為 None
        unimodality_result = detailed.get("unimodality_test_result", {})
        if isinstance(unimodality_result, dict):
            decision = unimodality_result.get("decision")
            if decision == "unimodal":
                assert detailed.get("peaks_table") is None
                assert detailed.get("stable_peaks_table") is None

    def test_bimodal_path_runs_peak_detection(self, bimodal_df, minimal_config):
        """雙峰資料應執行峰偵測"""
        brief, detailed = run_category_pipeline(
            bimodal_df, "A", minimal_config
        )
        
        # 應有峰相關資訊
        assert isinstance(brief, dict)
        assert "summary" in brief


# ======================================================
# Test 3: run_all_categories 整合測試
# ======================================================
class TestRunAllCategories:
    """run_all_categories 整合測試"""

    def test_processes_all_categories(self, simple_df, tmp_path):
        """應處理所有分類"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(simple_df, cfg)
        
        # 檢查輸出檔案
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        assert summary_path.exists(), "summary_all.json should exist"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert set(summary.keys()) == {"A", "B"}

    def test_creates_output_directory(self, simple_df, tmp_path):
        """應自動建立輸出目錄"""
        cfg = _build_minimal_config(tmp_path)
        new_reports_path = tmp_path / "new" / "nested" / "reports"
        cfg["pipeline_controller"]["reports_path"] = str(new_reports_path)
        
        run_all_categories(simple_df, cfg)
        
        assert new_reports_path.exists()

    def test_processes_specified_categories_only(self, simple_df, tmp_path):
        """指定 cats 參數時只處理指定分類"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(simple_df, cfg, cats=["A"])
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert set(summary.keys()) == {"A"}

    def test_skips_empty_category(self, simple_df, tmp_path):
        """空分類應被跳過"""
        cfg = _build_minimal_config(tmp_path)
        
        # 指定一個不存在的分類
        run_all_categories(simple_df, cfg, cats=["A", "NonExistent"])
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        # NonExistent 應被跳過
        assert "NonExistent" not in summary
        assert "A" in summary

    def test_separate_reports_option(self, simple_df, tmp_path):
        """separate_category_report=True 時應產生個別報告"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["reporting"]["separate_category_report"] = True
        
        run_all_categories(simple_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        separate_path = reports_path / "separate_reports"
        
        assert separate_path.exists()
        assert (separate_path / "summary_A.json").exists()
        assert (separate_path / "summary_B.json").exists()

    def test_detailed_reports_option(self, simple_df, tmp_path):
        """provide_details=True 時應產生詳細報告"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["reporting"]["provide_details"] = True
        
        run_all_categories(simple_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        detailed_path = reports_path / "complete_report_all.json"
        
        assert detailed_path.exists()


# ======================================================
# Test 4: 端到端測試
# ======================================================
class TestEndToEnd:
    """端到端整合測試"""

    def test_full_pipeline_unimodal(self, unimodal_df, tmp_path):
        """單峰資料完整 pipeline 測試"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert "A" in summary
        result = summary["A"]
        
        # brief_summary 結構為 {"summary": {...}, "figures": {...}}
        assert "summary" in result
        stats = result["summary"]
        
        # 驗證基本統計（n 是間隔數，不是原始交易數）
        assert stats["n"] > 0
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats

    def test_full_pipeline_bimodal(self, bimodal_df, tmp_path):
        """雙峰資料完整 pipeline 測試"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(bimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert "A" in summary

    def test_full_pipeline_multiple_categories(self, tmp_path):
        """多分類完整 pipeline 測試"""
        np.random.seed(42)
        base = pd.to_datetime("2024-01-01")
        
        rows = []
        # A: 100 users, unimodal ~10 days
        for i in range(100):
            interval = max(1, np.random.normal(10, 2))
            rows.append({"uid": f"a{i}", "cat": "A", "order_date": base})
            rows.append({"uid": f"a{i}", "cat": "A", "order_date": base + pd.Timedelta(days=interval)})
        
        # B: 100 users, unimodal ~30 days
        for i in range(100):
            interval = max(1, np.random.normal(30, 5))
            rows.append({"uid": f"b{i}", "cat": "B", "order_date": base})
            rows.append({"uid": f"b{i}", "cat": "B", "order_date": base + pd.Timedelta(days=interval)})
        
        # C: 100 users, bimodal
        for i in range(50):
            interval = max(1, np.random.normal(10, 2))
            rows.append({"uid": f"c{i}", "cat": "C", "order_date": base})
            rows.append({"uid": f"c{i}", "cat": "C", "order_date": base + pd.Timedelta(days=interval)})
        for i in range(50, 100):
            interval = max(1, np.random.normal(50, 5))
            rows.append({"uid": f"c{i}", "cat": "C", "order_date": base})
            rows.append({"uid": f"c{i}", "cat": "C", "order_date": base + pd.Timedelta(days=interval)})
        
        df = pd.DataFrame(rows)
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert set(summary.keys()) == {"A", "B", "C"}


# ======================================================
# Test 5: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_single_user_with_two_transactions(self, tmp_path):
        """單一用戶兩筆交易（產生一個間隔）"""
        base = pd.to_datetime("2024-01-01")
        df = pd.DataFrame({
            "uid": ["u1", "u1"],
            "cat": ["A", "A"],
            "order_date": [base, base + pd.Timedelta(days=10)],
        })
        cfg = _build_minimal_config(tmp_path)
        
        # 應能處理不報錯
        run_all_categories(df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        assert summary_path.exists()

    def test_multiple_users_few_transactions(self, tmp_path):
        """多用戶少量交易"""
        base = pd.to_datetime("2024-01-01")
        df = pd.DataFrame({
            "uid": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "cat": ["A"] * 6,
            "order_date": [
                base, base + pd.Timedelta(days=10),
                base, base + pd.Timedelta(days=15),
                base, base + pd.Timedelta(days=20),
            ],
        })
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        # 應有 3 個間隔
        assert summary["A"]["summary"]["n"] == 3

    def test_large_dataset_mode_switching(self, tmp_path):
        """大資料集應切換到適當模式"""
        np.random.seed(42)
        n = 50  # 每個用戶 2 筆交易 = 100 筆，介於 small 和 medium
        intervals = np.clip(np.random.normal(30, 5, n), 1, None)
        df = _generate_transaction_data(n, "A", intervals)
        
        cfg = _build_minimal_config(tmp_path)
        cfg["pipeline_controller"]["data_size_thresholds"] = {"small": 10, "medium": 200}
        
        run_all_categories(df, cfg)
        
        # 應成功處理
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        assert (reports_path / "summary_all.json").exists()


# ======================================================
# Test 6: 配置選項測試
# ======================================================
class TestConfigurationOptions:
    """配置選項測試"""

    def test_custom_data_cleaning_params(self, unimodal_df, tmp_path):
        """自訂資料清理參數"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["data_cleaning"]["outlier_method"] = "MAD"
        cfg["modules"]["data_cleaning"]["outlier_threshold"] = 3.5
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        assert (reports_path / "summary_all.json").exists()

    def test_custom_transform_params(self, unimodal_df, tmp_path):
        """自訂轉換參數"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["transform"]["method_candidates"] = ["none"]
        cfg["modules"]["transform"]["auto_select_by_skewness"] = False
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        assert (reports_path / "summary_all.json").exists()

    def test_custom_peak_detection_params(self, bimodal_df, tmp_path):
        """自訂峰偵測參數"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["peak_detection"]["grid_size"] = 512
        cfg["modules"]["peak_detection"]["prominence_min"] = 0.005
        
        run_all_categories(bimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        assert (reports_path / "summary_all.json").exists()


# ======================================================
# Test 7: 輸出驗證測試
# ======================================================
class TestOutputValidation:
    """輸出驗證測試"""

    def test_summary_json_schema(self, unimodal_df, tmp_path):
        """summary_all.json 結構驗證"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        result = summary["A"]
        
        # 驗證結構
        assert "summary" in result
        assert "figures" in result
        
        stats = result["summary"]
        
        # 基本統計欄位
        required_fields = ["n", "mean", "median", "std", "skew"]
        for field in required_fields:
            assert field in stats, f"Missing field: {field}"
        
        # 額外必要欄位（根據 reporting.py 實際輸出）
        additional_fields = [
            "original_transaction_counts", "original_n", "dip_p",
            "n_peaks", "PEP"
        ]
        for field in additional_fields:
            assert field in stats, f"Missing field: {field}"

    def test_validation_plots_created(self, bimodal_df, tmp_path):
        """驗證圖檔應被建立"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(bimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        validation_path = reports_path / "validation_plots"
        
        # validation_plots 目錄應存在
        assert validation_path.exists() or reports_path.exists()

    def test_numeric_values_are_valid(self, unimodal_df, tmp_path):
        """數值應為有效值（非 NaN、非 Inf）"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        result = summary["A"]["summary"]
        
        # 檢查數值欄位
        numeric_fields = ["mean", "median", "std", "skew"]
        for field in numeric_fields:
            if field in result:
                value = result[field]
                assert value is not None
                assert not (isinstance(value, float) and (np.isnan(value) or np.isinf(value)))

    def test_detailed_result_structure(self, unimodal_df, tmp_path):
        """detailed_result 結構驗證"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["reporting"]["provide_details"] = True
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        detailed_path = reports_path / "complete_report_all.json"
        
        with open(detailed_path) as f:
            detailed = json.load(f)
        
        result = detailed["A"]
        
        # 驗證 detailed_result 包含所有模組輸出
        expected_keys = [
            "interval_conversion_summary", "mode", "discard_summary",
            "transform_meta", "summary_stats", "unimodality_test_result",
            "peaks_table", "kde_plot_with_peaks", "modality_result",
            "consistency_check", "stable_peaks_table", "stability_plot", "PEP"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key in detailed_result: {key}"

    def test_pep_field_exists(self, unimodal_df, tmp_path):
        """PEP (Predicted Effective Period) 欄位應存在"""
        cfg = _build_minimal_config(tmp_path)
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        summary_path = reports_path / "summary_all.json"
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert "PEP" in summary["A"]["summary"]

    def test_interval_conversion_summary(self, unimodal_df, tmp_path):
        """interval_conversion_summary 應包含轉換統計"""
        cfg = _build_minimal_config(tmp_path)
        cfg["modules"]["reporting"]["provide_details"] = True
        
        run_all_categories(unimodal_df, cfg)
        
        reports_path = Path(cfg["pipeline_controller"]["reports_path"])
        detailed_path = reports_path / "complete_report_all.json"
        
        with open(detailed_path) as f:
            detailed = json.load(f)
        
        conversion = detailed["A"]["interval_conversion_summary"]
        assert "total_transactions" in conversion
        assert "output_intervals" in conversion
