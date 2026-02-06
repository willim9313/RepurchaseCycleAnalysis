"""
Repurchase Cycle Analysis Pipeline
----------------------------------
Automated pipeline to analyze repurchase cycles across categories.
"""
# repurchase_cycle/pipeline.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple
import os
from pathlib import Path
import pandas as pd
import json
from repurchase_cycle.modules.interval_derivation import run_interval_calculation
from repurchase_cycle.modules.data_cleaning import run_data_cleaning
from repurchase_cycle.modules.transform import run_transform
from repurchase_cycle.modules.visualization import run_visualization
from repurchase_cycle.modules.unimodality_test import run_unimodality_test
from repurchase_cycle.modules.peak_detection import run_peak_detection
from repurchase_cycle.modules.modality_quantification import run_modality_quantification
from repurchase_cycle.modules.stability_assessment import run_stability_assessment
from repurchase_cycle.modules.reporting import run_reporting
from repurchase_cycle.logging_utils import setup_logging

Mode = Literal["small", "medium", "large"]


def decide_mode(
    n_rows: int,
    thresholds: Mapping[str, float | int],
) -> Mode:
    """
    Decide processing mode ("small" | "medium" | "large") based on data size.
    Threshold depends on provided configuration(data_size_thresholds).

    Parameters
    ----------
    n_rows : int
        Number of rows for the current category.
    thresholds : Mapping[str, float | int]
        Thresholds configuration, expected keys:
        - "small", "medium"; defaults to 1e4 and 1e6 if not provided.

    Returns
    -------
    Mode : str
        One of "small", "medium", or "large".
    """
    small_th = int(float(thresholds.get("small", 1e4)))
    medium_th = int(float(thresholds.get("medium", 1e6)))

    if n_rows < small_th:
        return "small"
    if n_rows < medium_th:
        return "medium"
    return "large"


def run_category_pipeline(
    df_cat: pd.DataFrame,
    cat: str,
    cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full analysis pipeline for one single category.

    Parameters
    ----------
    df_cat : pd.DataFrame
        DataFrame filtered to a single category, containing at least
        ["uid", "cat", "interval_days"] columns.
    cat : str
        The category name.
    cfg : Dict[str, Any]
        Configuration dict(including pipeline and module-level configs).

    Returns
    -------
    result : Dict[str, Any]
        A dictionary containing all results and metadata for the category.
    """

    controller_cfg = cfg.get("pipeline_controller", {})
    modules_cfg = cfg.get("modules", {})

    thresholds = controller_cfg.get("data_size_thresholds", {})
    n_rows = len(df_cat)
    mode: Mode = decide_mode(n_rows, thresholds)
    output_path = Path(controller_cfg.get("reports_path", "./reports"))
    validation_graph_path = output_path / "validation_plots"

    # --- 00. Interval derivation ---
    id_cfg = modules_cfg.get("interval_derivation", {})
    interval_df, conversion_summary = run_interval_calculation(
        df_cat,
        mode,
        controller_cfg,
        id_cfg
    )

    # --- 01. Data cleaning ---
    dc_cfg = modules_cfg.get("data_cleaning", {})
    cleaned_df, discard_summary = run_data_cleaning(
        interval_df,
        mode,
        controller_cfg,
        dc_cfg
    )

    # --- 02. Transform ---
    tf_cfg = modules_cfg.get("transform", {})
    transformed_df, transform_meta = run_transform(
        cleaned_df,
        mode,
        controller_cfg,
        tf_cfg
    )

    # --- 03. Visualization ---
    vis_cfg = modules_cfg.get("visualization", {})
    plots_dir, summary_stats = run_visualization(
        transformed_df,
        mode,
        cat,
        controller_cfg,
        vis_cfg,
        controller_cfg.get("reports_path", "./reports")
    )

    # --- 04. unimodality_test test ---
    uni_cfg = modules_cfg.get("unimodality_test", {})
    unimodality_test_result = run_unimodality_test(
        transformed_df,
        mode,
        controller_cfg,
        uni_cfg
    )

    decision = None
    if isinstance(unimodality_test_result, dict):
        decision = unimodality_test_result.get("decision")
    else:
        decision = getattr(unimodality_test_result, "decision", None)

    # initialize outputs for optional modules
    peaks_table = None
    kde_plot_with_peaks = None
    modality_result = None
    consistency_check = None
    stable_peaks_table = None
    stability_plot = None

    if decision is None or decision != "unimodal":
        # --- 05. Peak detection ---
        pd_cfg = modules_cfg.get("peak_detection", {})
        peaks_table, kde_plot_with_peaks = run_peak_detection(
            cat,
            transformed_df,
            mode,
            controller_cfg,
            pd_cfg,
            validation_graph_path.as_posix(),
            transform_meta=transform_meta
        )
        # --- 06. Modality quantification ---
        mq_cfg = modules_cfg.get("modality_quantification", {})
        modality_result, consistency_check = run_modality_quantification(
            transformed_df,
            mode,
            controller_cfg,
            mq_cfg,
        )
        # --- 07. Stability assessment ---
        sa_cfg = modules_cfg.get("stability_assessment", {})
        stable_peaks_table, stability_plot = run_stability_assessment(
            cat,
            transformed_df,
            peaks_table,
            mode,
            controller_cfg,
            sa_cfg,
            validation_graph_path.as_posix(),
            transform_meta=transform_meta
        )

    # --- 08. Reporting ---
    rpt_cfg = modules_cfg.get("reporting", {})
    brief_summary, detailed_result = run_reporting(
        conversion_summary,
        discard_summary,
        transform_meta,
        summary_stats,
        unimodality_test_result,
        peaks_table,
        kde_plot_with_peaks,
        modality_result,
        consistency_check,
        stable_peaks_table,
        stability_plot,
        mode,
        rpt_cfg
    )

    return brief_summary, detailed_result


def run_all_categories(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    cats: Optional[Iterable[str]] = None
) -> None:
    """
    Run the analysis pipeline for all specified categories.
    Each category is processed independently based on its row counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least columns ["uid", "cat", "interval_days"].
    cfg : Dict[str, Any]
        Global configuration dict, typically from `load_config`.
    cats : Optional[Iterable[str]], optional
        Categories to process. If None, use df["cat"].unique().

    Returns
    -------
    None
        Results are saved to files as specified in the configuration.
    """
    # Setup
    controller_cfg = cfg.get("pipeline_controller", {})
    logger = setup_logging(controller_cfg, "repurchase_cycle")
    logger.info("Starting repurchase cycle analysis pipeline")

    report_cfg = cfg.get("modules", {}).get("reporting", {})
    provide_details = report_cfg.get("provide_details", False)
    save_separate_reports = report_cfg.get("separate_category_report", False)
    cat_col = cfg.get("modules", {})\
                 .get("interval_derivation", {})\
                 .get("cat_col", "cat")

    # folder to save reports
    root_path = Path(controller_cfg.get("reports_path", "./reports"))
    os.makedirs(root_path, exist_ok=True)
    if save_separate_reports:
        os.makedirs(root_path / "separate_reports", exist_ok=True)

    if cats is None:
        cats_list = df[cat_col].unique().tolist()
    else:
        cats_list = list(cats)

    brief_results: Dict[str, Dict[str, Any]] = {}
    detailed_results: Dict[str, Dict[str, Any]] = {}

    for cat in cats_list:
        logger.info(f"Processing category: {cat}")
        df_cat = df[df[cat_col] == cat].copy()
        if df_cat.empty:
            logger.info(f"Category '{cat}' has no data, skipping.")
            continue

        brief, detailed = run_category_pipeline(df_cat, str(cat), cfg)
        brief_results[str(cat)] = brief
        detailed_results[str(cat)] = detailed

        if save_separate_reports:
            output_path = root_path / "separate_reports" / f"summary_{cat}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(brief, f, ensure_ascii=False, indent=2)

            if provide_details:
                detailed_output_path = root_path / "separate_reports" / f"complete_report_{cat}.json"
                with detailed_output_path.open("w", encoding="utf-8") as f:
                    json.dump(detailed, f, ensure_ascii=False, indent=2)

    # overall summary export
    summary_all = dict(brief_results.items())

    summary_path = root_path / "summary_all.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_all, f, ensure_ascii=False, indent=2)

    if provide_details:
        detailed_all_path = root_path / "complete_report_all.json"
        detailed_all = dict(detailed_results.items())
        with detailed_all_path.open("w", encoding="utf-8") as f:
            json.dump(detailed_all, f, ensure_ascii=False, indent=2)

    logger.info("Repurchase cycle analysis pipeline completed.")
