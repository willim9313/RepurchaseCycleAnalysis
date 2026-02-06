from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DEFAULT_REPORTING_PARAMS: Dict[str, Any] = {
    "export_formats": ["json", "pdf", "png"],
    "provide_details": True,
    "separate_category_report": True,
    "reports_path": "./reports"
}


def _decide_pep(
    dip_p: Optional[float],
    unimodality_test_decision: Optional[str],
    alpha: float,
    n_peaks: int,
    n_stable_peaks: int,
    best_n_components: Optional[int],
    consistency: str,
    is_flat_distribution: bool = False,
    stable_peak_positions: Optional[List[float]] = None,
    median: Optional[float] = None,
) -> str:
    """
    Peak Existence Probability(PEP) decision - provide actionable insights.

    Returns a human-readable summary of detected repurchase cycles.
    """
    # 平坦分佈：沒有明顯的購買週期
    if is_flat_distribution:
        return "No clear repurchase cycle detected (uniform-like distribution)"

    decision = (unimodality_test_decision or "").lower()
    consistency_norm = (consistency or "").lower()
    bnc = best_n_components if isinstance(best_n_components, (int, float)) else None
    is_unimodal = decision == "unimodal"

    # === Unimodal case ===
    if is_unimodal and n_peaks <= 1:
        if median is not None:
            return f"Single repurchase cycle detected at ~{median:.1f} days"
        return "Single repurchase cycle detected"

    # === Multimodal case with stable peaks ===
    if n_stable_peaks >= 2 and stable_peak_positions:
        # 檢查一致性
        is_consistent = consistency_norm in ("consistent", "match", "ok")
        peaks_str = ", ".join([f"~{p:.1f}" for p in sorted(stable_peak_positions)])
        
        if is_consistent:
            return f"{n_stable_peaks} repurchase cycles detected at {peaks_str} days"
        else:
            return f"{n_stable_peaks} repurchase cycles detected at {peaks_str} days (GMM/KDE inconsistent, verify manually)"

    # === Multimodal but unstable or few stable peaks ===
    if n_peaks >= 2 and n_stable_peaks < 2:
        return f"Potential multiple cycles detected but unstable (only {n_stable_peaks} stable peaks), recommend more data or manual review"

    # === Fallback: 結果矛盾或不明確 ===
    return "Results inconclusive, recommend manual review"


def run_reporting(
    conversion_summary: Dict[str, Any],
    discard_summary: Dict[str, Any],
    transform_meta: Dict[str, Any],
    summary_stats: Dict[str, float],
    unimodality_test_result: Dict[str, Any],
    peaks_table: Optional[List[Dict[str, float]]],
    kde_plot_with_peaks: Optional[str],
    modality_result: Optional[Dict[str, Any]],
    consistency_check: Optional[Dict[str, Any]],
    stable_peaks_table: Optional[List[Dict[str, float]]],
    stability_plot: Optional[str],
    mode: str = "small",
    mod_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Summarize analysis results, then output reports and recommendations (PEP).

    Parameters
    ----------
    discard_summary : dict
        Summary of discarded data.
    transform_meta : dict
        Data transformation metadata.
    summary_stats : dict
        Distribution statistics.
    unimodality_test_result : dict
        single-peak test results.
    peaks_table : list[dict]
        peak detection results, each element represents a single peak's features.
    kde_plot_with_peaks : str
        File path to the KDE plot with detected peaks annotated.
    modality_result : dict
        GMM modality quantification results.
    consistency_check : dict
        Consistency check results between GMM and KDE peak counts.
    stable_peaks_table : list[dict]
        List of peaks that remain supported after stability testing.
    stability_plot : str
        File path to the peak stability plot.
    mode : {"small", "medium", "large"}
        Different data scale modes.
    mod_params : dict, optional
        Other supplementary parameters. Check `DEFAULT_REPORTING_PARAMS` for possible keys.

    Returns
    -------
    brief_summary : dict
        Summary report including key indicators and PEP recommendation.
    detailed_result : dict
        Complete report details including all sub-module results.
    """
    mod_params = mod_params or {}
    cfg = {**DEFAULT_REPORTING_PARAMS, **mod_params}
    logger.info("=== Reporting Config ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Using config: {cfg}")

    original_n = float(discard_summary.get("total_rows", float("nan")))
    n = (
        summary_stats.get("n")
        or summary_stats.get("count")
        or unimodality_test_result.get("n")
    )

    # if n still None, try fallback from summary_stats
    if n is None:
        n = int(summary_stats.get("n_samples", 0))

    # Retrieve statistics and test indicators
    mean = float(summary_stats.get("mean", float("nan")))
    median = float(summary_stats.get("median", float("nan")))
    std = float(summary_stats.get("std", float("nan")))
    skew = float(summary_stats.get("skew", float("nan")))

    dip_p = unimodality_test_result.get("dip_p")
    alpha = unimodality_test_result.get("alpha", mod_params.get("alpha", 0.05))
    decision = unimodality_test_result.get("decision")
    method_used = unimodality_test_result.get("method_used", "")

    # === 關鍵：區分 unimodal 的兩種來源 ===
    is_unimodal = (decision or "").lower() == "unimodal"
    is_flat_distribution = "flat_distribution" in method_used.lower()

    # === 根據不同情況設定 peaks 相關數值 ===
    if is_flat_distribution:
        # Uniform-like：沒有明顯的峰
        n_peaks = 0
        n_stable_peaks = 0
        peaks_for_report = []
        stable_peaks_for_report = []
        best_n_components = None  # GMM 也沒跑
        consistency = "N/A (flat distribution)"
    elif is_unimodal and peaks_table is None:
        # 真正的單峰，但沒有跑 peak detection
        n_peaks = 1
        n_stable_peaks = 1
        peaks_for_report = [{
            "pos": median,
            "pos_transformed": None,
            "height": None,
            "width": None,
            "prominence": None,
            "source": "inferred_from_unimodal_median"
        }]
        stable_peaks_for_report = [{
            "pos": median,
            "pos_transformed": None,
            "support_ratio": 1.0,
            "source": "inferred_from_unimodal_median"
        }]
        best_n_components = 1
        consistency = "consistent"
    else:
        # Multimodal case 或有實際 peak detection 結果
        n_peaks = len(peaks_table) if peaks_table is not None else 0
        n_stable_peaks = len(stable_peaks_table) if stable_peaks_table is not None else 0
        
        peaks_for_report = []
        if peaks_table:
            for p in peaks_table:
                peaks_for_report.append({
                    "pos": p.get("pos_original", p["pos"]),
                    "pos_transformed": p.get("pos_transformed", p["pos"]),
                    "height": p["height"],
                    "width": p["width"],
                    "prominence": p["prominence"],
                })

        stable_peaks_for_report = []
        if stable_peaks_table:
            for p in stable_peaks_table:
                stable_peaks_for_report.append({
                    "pos": p.get("pos_original", p["pos"]),
                    "pos_transformed": p.get("pos_transformed", p["pos"]),
                    "support_ratio": p["support_ratio"],
                })

        best_n_components = modality_result.get("best_n_components") if modality_result else None
        consistency = consistency_check.get("status") if consistency_check else "UNKNOWN"

    # 收集 stable peaks 的位置（原始尺度）
    stable_peak_positions = None
    if stable_peaks_for_report:
        stable_peak_positions = [p["pos"] for p in stable_peaks_for_report if p.get("pos") is not None]

    # decide PEP
    pep = _decide_pep(
        dip_p=dip_p,
        unimodality_test_decision=decision,
        alpha=alpha,
        n_peaks=n_peaks,
        n_stable_peaks=n_stable_peaks,
        best_n_components=best_n_components,
        consistency=str(consistency),
        is_flat_distribution=is_flat_distribution,
        stable_peak_positions=stable_peak_positions,
        median=median,
    )

    summary_json: Dict[str, Any] = {
        "original_transaction_counts": int(conversion_summary.get("total_transactions", 0)),
        "original_n": original_n,
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "skew": skew,
        "dip_p": float(dip_p) if dip_p is not None else None,
        "is_flat_distribution": is_flat_distribution,  # 新增欄位
        "n_peaks": int(n_peaks),
        "peaks": peaks_for_report,
        "stable_peaks": stable_peaks_for_report,
        "best_n_components": best_n_components,
        "consistency": consistency,
        "PEP": pep,
        "meta": {
            "unimodality_test_result": unimodality_test_result,
            "gmm_result": modality_result,
            "mode": mode,
            "alpha_used": alpha,
        },
    }

    # Handle figure paths (produced by previous modules)
    figures = {
        "distribution_plot": kde_plot_with_peaks if kde_plot_with_peaks else None,
        "stability_plot": stability_plot if stability_plot else None,
    }

    # integrate output infos
    brief_summary = {
        "summary": summary_json,
        "figures": figures,
    }

    detailed_result = {
        "interval_conversion_summary": conversion_summary,
        "mode": mode,
        "discard_summary": discard_summary,
        "transform_meta": transform_meta,
        "summary_stats": summary_stats,
        "unimodality_test_result": unimodality_test_result,
        "peaks_table": peaks_table,
        "kde_plot_with_peaks": kde_plot_with_peaks,
        "modality_result": modality_result,
        "consistency_check": consistency_check,
        "stable_peaks_table": stable_peaks_table,
        "stability_plot": stability_plot,
        "PEP": pep
    }

    return brief_summary, detailed_result
