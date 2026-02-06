# ======================================================
# test_reporting.py
# ======================================================
# 單元測試模組: 結果整合與匯出
# 對應模組: run_reporting()
# ------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pytest

from repurchase_cycle.modules.reporting import run_reporting, _decide_pep


# ======================================================
# Fixtures
# ======================================================
@pytest.fixture
def base_conversion_summary() -> Dict[str, Any]:
    """交易轉換摘要"""
    return {
        "total_transactions": 500,
        "unique_customers": 100,
        "unique_products": 50,
    }


@pytest.fixture
def base_discard_summary() -> Dict[str, Any]:
    """資料清理摘要"""
    return {
        "total_rows": 120,
        "removed_negatives": 5,
        "removed_missing": 10,
        "removed_outliers": 5,
    }


@pytest.fixture
def base_transform_meta() -> Dict[str, Any]:
    """轉換元資料"""
    return {
        "method": "log1p",
        "skewness_before": 2.5,
        "skewness_after": 0.8,
    }


@pytest.fixture
def base_summary_stats() -> Dict[str, float]:
    """分布統計量"""
    return {
        "n": 100,
        "mean": 20.0,
        "median": 18.0,
        "std": 5.0,
        "skew": 1.2,
    }


@pytest.fixture
def base_unimodality_test_result() -> Dict[str, Any]:
    """單峰檢定結果"""
    return {
        "dip_p": 0.5,
        "decision": "unimodal",
        "alpha": 0.05,
        "method_used": "dip_test",
        "n": 100,
    }


@pytest.fixture
def base_peaks_table() -> List[Dict[str, float]]:
    """峰偵測結果"""
    return [
        {"pos": 10.0, "height": 0.1, "width": 2.0, "prominence": 0.05},
        {"pos": 30.0, "height": 0.08, "width": 3.0, "prominence": 0.04},
    ]


@pytest.fixture
def base_modality_result() -> Dict[str, Any]:
    """GMM 模態量化結果"""
    return {
        "best_n_components": 2,
        "aic_scores": [100.0, 90.0, 95.0],
        "bic_scores": [110.0, 95.0, 100.0],
    }


@pytest.fixture
def base_consistency_check() -> Dict[str, Any]:
    """一致性檢查結果"""
    return {
        "kde_n_peaks": 2,
        "gmm_n_components": 2,
        "status": "consistent",
    }


@pytest.fixture
def base_stable_peaks_table() -> List[Dict[str, float]]:
    """穩定峰列表"""
    return [
        {"pos": 10.0, "support_ratio": 0.85},
        {"pos": 30.0, "support_ratio": 0.80},
    ]


# ======================================================
# Test 1: _decide_pep 單元測試
# ======================================================
class TestDecidePep:
    """PEP 決策邏輯測試"""

    def test_single_cycle_with_median(self):
        """單峰 → 回傳單一週期訊息並顯示 median"""
        pep = _decide_pep(
            dip_p=0.6,
            unimodality_test_decision="unimodal",
            alpha=0.05,
            n_peaks=1,
            n_stable_peaks=1,
            best_n_components=1,
            consistency="consistent",
            median=18.0,
        )
        assert "Single repurchase cycle detected" in pep
        assert "18.0" in pep

    def test_single_cycle_without_median(self):
        """單峰但無 median → 回傳單一週期訊息"""
        pep = _decide_pep(
            dip_p=0.6,
            unimodality_test_decision="unimodal",
            alpha=0.05,
            n_peaks=0,
            n_stable_peaks=0,
            best_n_components=1,
            consistency="consistent",
            median=None,
        )
        assert pep == "Single repurchase cycle detected"

    def test_flat_distribution(self):
        """平坦分佈 → 無明顯購買週期"""
        pep = _decide_pep(
            dip_p=0.8,
            unimodality_test_decision="unimodal",
            alpha=0.05,
            n_peaks=0,
            n_stable_peaks=0,
            best_n_components=None,
            consistency="",
            is_flat_distribution=True,
        )
        assert pep == "No clear repurchase cycle detected (uniform-like distribution)"

    def test_multi_cycle_consistent(self):
        """多峰顯著 + 穩定峰 >= 2 + 一致 → 回傳多週期訊息"""
        pep = _decide_pep(
            dip_p=0.001,
            unimodality_test_decision="multimodal",
            alpha=0.05,
            n_peaks=3,
            n_stable_peaks=2,
            best_n_components=3,
            consistency="consistent",
            stable_peak_positions=[10.0, 30.0],
        )
        assert "2 repurchase cycles detected" in pep
        assert "~10.0" in pep
        assert "~30.0" in pep
        assert "inconsistent" not in pep.lower()

    def test_multi_cycle_inconsistent(self):
        """多峰 + 穩定峰 >= 2 + 不一致 → 回傳多週期訊息並提示手動驗證"""
        pep = _decide_pep(
            dip_p=0.01,
            unimodality_test_decision="multimodal",
            alpha=0.05,
            n_peaks=2,
            n_stable_peaks=2,
            best_n_components=1,
            consistency="inconsistent",
            stable_peak_positions=[10.0, 30.0],
        )
        assert "2 repurchase cycles detected" in pep
        assert "inconsistent" in pep.lower()
        assert "verify manually" in pep.lower()

    def test_unstable_peaks(self):
        """多峰但穩定峰不足 → 建議更多數據或人工審查"""
        pep = _decide_pep(
            dip_p=0.01,
            unimodality_test_decision="multimodal",
            alpha=0.05,
            n_peaks=3,
            n_stable_peaks=1,
            best_n_components=3,
            consistency="consistent",
        )
        assert "unstable" in pep.lower()
        assert "1 stable" in pep
        assert "manual review" in pep.lower()

    def test_unimodal_with_none_components(self):
        """單峰 + best_n_components=None → 回傳單一週期訊息"""
        pep = _decide_pep(
            dip_p=0.8,
            unimodality_test_decision="unimodal",
            alpha=0.05,
            n_peaks=0,
            n_stable_peaks=0,
            best_n_components=None,
            consistency="",
            median=25.0,
        )
        assert "Single repurchase cycle detected" in pep
        assert "25.0" in pep

    def test_consistency_match_status(self):
        """一致性狀態為 'match' 應視為一致"""
        pep = _decide_pep(
            dip_p=0.001,
            unimodality_test_decision="multimodal",
            alpha=0.05,
            n_peaks=2,
            n_stable_peaks=2,
            best_n_components=2,
            consistency="match",
            stable_peak_positions=[15.0, 45.0],
        )
        assert "2 repurchase cycles detected" in pep
        assert "inconsistent" not in pep.lower()

    def test_inconclusive_fallback(self):
        """結果不明確時 → 建議人工審查"""
        pep = _decide_pep(
            dip_p=0.5,
            unimodality_test_decision="unknown",
            alpha=0.05,
            n_peaks=1,
            n_stable_peaks=0,
            best_n_components=2,
            consistency="unknown",
        )
        assert "inconclusive" in pep.lower()
        assert "manual review" in pep.lower()


# ======================================================
# Test 2: run_reporting 基本結構測試
# ======================================================
class TestRunReportingStructure:
    """run_reporting 回傳結構測試"""

    def test_returns_two_dicts(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_peaks_table,
        base_modality_result,
        base_consistency_check,
        base_stable_peaks_table,
    ):
        """應回傳 brief_summary 和 detailed_result 兩個字典"""
        brief, detailed = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks="/path/to/kde_plot.png",
            modality_result=base_modality_result,
            consistency_check=base_consistency_check,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot="/path/to/stability_plot.png",
            mode="small",
        )

        assert isinstance(brief, dict)
        assert isinstance(detailed, dict)

    def test_brief_summary_structure(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_peaks_table,
        base_modality_result,
        base_consistency_check,
        base_stable_peaks_table,
    ):
        """brief_summary 結構驗證"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks=None,
            modality_result=base_modality_result,
            consistency_check=base_consistency_check,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot=None,
        )

        assert "summary" in brief
        assert "figures" in brief

        summary_json = brief["summary"]
        expected_keys = {
            "original_transaction_counts", "original_n", "n", "mean", "median", "std", "skew",
            "dip_p", "is_flat_distribution", "n_peaks", "peaks", "stable_peaks",
            "best_n_components", "consistency", "PEP", "meta"
        }
        assert expected_keys.issubset(set(summary_json.keys()))

    def test_detailed_result_structure(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_peaks_table,
        base_modality_result,
        base_consistency_check,
        base_stable_peaks_table,
    ):
        """detailed_result 結構驗證"""
        _, detailed = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks="/path/to/kde.png",
            modality_result=base_modality_result,
            consistency_check=base_consistency_check,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot="/path/to/stability.png",
        )

        expected_keys = {
            "interval_conversion_summary", "mode", "discard_summary", "transform_meta", "summary_stats",
            "unimodality_test_result", "peaks_table", "kde_plot_with_peaks",
            "modality_result", "consistency_check", "stable_peaks_table",
            "stability_plot", "PEP"
        }
        assert expected_keys.issubset(set(detailed.keys()))


# ======================================================
# Test 3: PEP 分支邏輯整合測試
# ======================================================
class TestPepBranchLogic:
    """PEP 分支邏輯整合測試"""

    def test_pep_single_cycle(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
    ):
        """情境一：單峰 → 單一購買週期"""
        unimodality_result = {
            "dip_p": 0.6,
            "decision": "unimodal",
            "alpha": 0.05,
            "method_used": "dip_test",
        }
        peaks_table = [{"pos": 10.0, "height": 0.1, "width": 2.0, "prominence": 0.05}]
        stable_peaks = [{"pos": 10.0, "support_ratio": 0.9}]
        modality_result = {"best_n_components": 1}
        consistency_check = {"status": "consistent"}

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=unimodality_result,
            peaks_table=peaks_table,
            kde_plot_with_peaks=None,
            modality_result=modality_result,
            consistency_check=consistency_check,
            stable_peaks_table=stable_peaks,
            stability_plot=None,
        )

        pep = brief["summary"]["PEP"]
        assert "Single repurchase cycle detected" in pep

    def test_pep_multi_cycle(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
    ):
        """情境二：多峰顯著 + 穩定峰 >= 2 → 多購買週期"""
        unimodality_result = {
            "dip_p": 0.001,
            "decision": "multimodal",
            "alpha": 0.05,
            "method_used": "dip_test",
        }
        peaks_table = [
            {"pos": 10.0, "height": 0.1, "width": 2.0, "prominence": 0.05},
            {"pos": 30.0, "height": 0.08, "width": 3.0, "prominence": 0.04},
        ]
        stable_peaks = [
            {"pos": 10.0, "support_ratio": 0.85},
            {"pos": 30.0, "support_ratio": 0.80},
        ]
        modality_result = {"best_n_components": 3}
        consistency_check = {"status": "consistent"}

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=unimodality_result,
            peaks_table=peaks_table,
            kde_plot_with_peaks=None,
            modality_result=modality_result,
            consistency_check=consistency_check,
            stable_peaks_table=stable_peaks,
            stability_plot=None,
        )

        pep = brief["summary"]["PEP"]
        assert "2 repurchase cycles detected" in pep

    def test_pep_inconsistent(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
    ):
        """情境三：結果矛盾 → 提示驗證"""
        unimodality_result = {
            "dip_p": 0.01,
            "decision": "multimodal",
            "alpha": 0.05,
            "method_used": "dip_test",
        }
        peaks_table = [
            {"pos": 10.0, "height": 0.1, "width": 2.0, "prominence": 0.05},
            {"pos": 30.0, "height": 0.08, "width": 3.0, "prominence": 0.04},
        ]
        stable_peaks = [
            {"pos": 10.0, "support_ratio": 0.85},
            {"pos": 30.0, "support_ratio": 0.80},
        ]
        modality_result = {"best_n_components": 1}
        consistency_check = {"status": "inconsistent"}

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=unimodality_result,
            peaks_table=peaks_table,
            kde_plot_with_peaks=None,
            modality_result=modality_result,
            consistency_check=consistency_check,
            stable_peaks_table=stable_peaks,
            stability_plot=None,
        )

        pep = brief["summary"]["PEP"]
        assert "inconsistent" in pep.lower()
        assert "verify manually" in pep.lower()

    def test_pep_flat_distribution(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
    ):
        """情境四：平坦分佈 → 無明顯週期"""
        unimodality_result = {
            "dip_p": 0.8,
            "decision": "unimodal",
            "alpha": 0.05,
            "method_used": "flat_distribution_detected",
        }

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=unimodality_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        pep = brief["summary"]["PEP"]
        assert "No clear repurchase cycle detected" in pep
        assert brief["summary"]["is_flat_distribution"] is True


# ======================================================
# Test 4: None 值處理測試
# ======================================================
class TestNoneValueHandling:
    """None 值輸入處理測試"""

    def test_none_peaks_table_unimodal(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
    ):
        """peaks_table 為 None 且 unimodal 時應推斷單峰"""
        unimodality_result = {
            "dip_p": 0.5,
            "decision": "unimodal",
            "alpha": 0.05,
            "method_used": "dip_test",
            "n": 100,
        }
        brief, detailed = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=unimodality_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        # Unimodal without peaks_table should infer single peak from median
        assert brief["summary"]["n_peaks"] == 1
        assert len(brief["summary"]["peaks"]) == 1
        assert brief["summary"]["peaks"][0]["source"] == "inferred_from_unimodal_median"

    def test_none_modality_result(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_peaks_table,
        base_stable_peaks_table,
    ):
        """modality_result 為 None 時應正常處理"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot=None,
        )

        assert brief["summary"]["best_n_components"] is None

    def test_none_consistency_check(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_peaks_table,
        base_modality_result,
        base_stable_peaks_table,
    ):
        """consistency_check 為 None 時應正常處理"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks=None,
            modality_result=base_modality_result,
            consistency_check=None,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot=None,
        )

        assert brief["summary"]["consistency"] == "UNKNOWN"


# ======================================================
# Test 5: 統計值提取測試
# ======================================================
class TestStatisticsExtraction:
    """統計值正確提取測試"""

    def test_extracts_statistics_correctly(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_unimodality_test_result,
        base_peaks_table,
        base_modality_result,
        base_consistency_check,
        base_stable_peaks_table,
    ):
        """應正確提取統計值"""
        summary_stats = {
            "n": 150,
            "mean": 25.5,
            "median": 22.0,
            "std": 8.3,
            "skew": 0.7,
        }

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks=None,
            modality_result=base_modality_result,
            consistency_check=base_consistency_check,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot=None,
        )

        s = brief["summary"]
        assert s["n"] == 150
        assert s["mean"] == 25.5
        assert s["median"] == 22.0
        assert s["std"] == 8.3
        assert s["skew"] == 0.7

    def test_extracts_original_n_from_discard_summary(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """應從 discard_summary 提取 original_n"""
        discard_summary = {
            "total_rows": 200,
            "removed_negatives": 10,
            "removed_missing": 20,
            "removed_outliers": 15,
        }

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        assert brief["summary"]["original_n"] == 200.0

    def test_extracts_original_transaction_counts(
        self,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """應從 conversion_summary 提取 original_transaction_counts"""
        conversion_summary = {
            "total_transactions": 1000,
            "unique_customers": 200,
        }

        brief, _ = run_reporting(
            conversion_summary=conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        assert brief["summary"]["original_transaction_counts"] == 1000


# ======================================================
# Test 6: 圖檔路徑處理測試
# ======================================================
class TestFigurePaths:
    """圖檔路徑處理測試"""

    def test_figure_paths_included(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_peaks_table,
        base_modality_result,
        base_consistency_check,
        base_stable_peaks_table,
    ):
        """圖檔路徑應包含在 figures 中"""
        kde_path = "/reports/kde_plot.png"
        stability_path = "/reports/stability_plot.png"

        brief, detailed = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=base_peaks_table,
            kde_plot_with_peaks=kde_path,
            modality_result=base_modality_result,
            consistency_check=base_consistency_check,
            stable_peaks_table=base_stable_peaks_table,
            stability_plot=stability_path,
        )

        assert brief["figures"]["distribution_plot"] == kde_path
        assert brief["figures"]["stability_plot"] == stability_path
        assert detailed["kde_plot_with_peaks"] == kde_path
        assert detailed["stability_plot"] == stability_path

    def test_none_figure_paths(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """None 圖檔路徑應正常處理"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        assert brief["figures"]["distribution_plot"] is None
        assert brief["figures"]["stability_plot"] is None


# ======================================================
# Test 7: 模式參數測試
# ======================================================
class TestModeParameter:
    """模式參數測試"""

    @pytest.mark.parametrize("mode", ["small", "medium", "large"])
    def test_mode_included_in_result(
        self,
        mode,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """模式應包含在結果中"""
        brief, detailed = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
            mode=mode,
        )

        assert detailed["mode"] == mode
        assert brief["summary"]["meta"]["mode"] == mode


# ======================================================
# Test 8: PEP 非空字串測試
# ======================================================
class TestPepNonEmpty:
    """PEP 必須為非空字串"""

    def test_pep_always_non_empty_string(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """PEP 應為非空字串"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        pep = brief["summary"]["PEP"]
        assert isinstance(pep, str)
        assert len(pep.strip()) > 0

    def test_pep_in_detailed_result(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """PEP 應同時出現在 detailed_result 中"""
        _, detailed = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        assert "PEP" in detailed
        assert isinstance(detailed["PEP"], str)
        assert len(detailed["PEP"].strip()) > 0


# ======================================================
# Test 9: Meta 資訊測試
# ======================================================
class TestMetaInformation:
    """Meta 資訊測試"""

    def test_meta_contains_unimodality_result(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_modality_result,
    ):
        """meta 應包含 unimodality_test_result"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=base_modality_result,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        meta = brief["summary"]["meta"]
        assert "unimodality_test_result" in meta
        assert meta["unimodality_test_result"] == base_unimodality_test_result

    def test_meta_contains_gmm_result(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
        base_modality_result,
    ):
        """meta 應包含 gmm_result"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=base_modality_result,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        meta = brief["summary"]["meta"]
        assert "gmm_result" in meta
        assert meta["gmm_result"] == base_modality_result

    def test_meta_contains_alpha_used(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """meta 應包含 alpha_used"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        meta = brief["summary"]["meta"]
        assert "alpha_used" in meta
        assert meta["alpha_used"] == 0.05


# ======================================================
# Test 10: is_flat_distribution 欄位測試
# ======================================================
class TestFlatDistributionFlag:
    """is_flat_distribution 欄位測試"""

    def test_flat_distribution_true(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
    ):
        """method_used 包含 flat_distribution 時，is_flat_distribution 應為 True"""
        unimodality_result = {
            "dip_p": 0.9,
            "decision": "unimodal",
            "alpha": 0.05,
            "method_used": "flat_distribution_detected",
        }

        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=unimodality_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        assert brief["summary"]["is_flat_distribution"] is True

    def test_flat_distribution_false(
        self,
        base_conversion_summary,
        base_discard_summary,
        base_transform_meta,
        base_summary_stats,
        base_unimodality_test_result,
    ):
        """一般情況下，is_flat_distribution 應為 False"""
        brief, _ = run_reporting(
            conversion_summary=base_conversion_summary,
            discard_summary=base_discard_summary,
            transform_meta=base_transform_meta,
            summary_stats=base_summary_stats,
            unimodality_test_result=base_unimodality_test_result,
            peaks_table=None,
            kde_plot_with_peaks=None,
            modality_result=None,
            consistency_check=None,
            stable_peaks_table=None,
            stability_plot=None,
        )

        assert brief["summary"]["is_flat_distribution"] is False
