import pytest
import pandas as pd
import numpy as np
import matplotlib.figure as mplfig

from src.viz import (
    plot_category_distribution,
    plot_multi_category_kde,
    plot_summary_comparison,
)


@pytest.fixture
def interval_df():
    """模擬 interval_df：含多分類、多用戶、client_type。"""
    return pd.DataFrame(
        {
            "uid": ["U1", "U1", "U2", "U2", "U3", "U3", "U3"],
            "cat": ["A", "A", "B", "B", "B", "A", "A"],
            "interval_days": [10, 20, 5, 7, 9, 15, np.nan],
            "client_type": ["new", "new", "old", "old", "new", "old", "old"],
        }
    )


@pytest.fixture
def dist_report():
    """模擬 analyze_distribution 輸出結果。"""
    return pd.DataFrame(
        {
            "cat": ["A", "B"],
            "n": [3, 3],
            "mean": [15.0, 7.0],
            "median": [15.0, 7.0],
            "std": [5.0, 1.5],
            "cv": [0.33, 0.21],
            "kde_peaks": [[10, 20], [5, 8]],
        }
    )


# ----------------------------------------------------------------------
#  Layer 1: plot_category_distribution
# ----------------------------------------------------------------------

def test_plot_category_distribution_basic(interval_df):
    fig = plot_category_distribution(
        interval_df,
        category="A",
        kde_peaks=[10, 20],
        show=False,  # 不開啟視窗
    )
    assert isinstance(fig, mplfig.Figure)
    # 檢查軸標題
    ax = fig.axes[0]
    assert "Interval" in ax.get_xlabel()
    assert "Density" in ax.get_ylabel() or "Samples" in ax.get_ylabel()


def test_plot_category_distribution_with_client(interval_df):
    fig = plot_category_distribution(
        interval_df,
        category="B",
        client_col="client_type",
        kde_peaks=None,
        show=False,
    )
    assert isinstance(fig, mplfig.Figure)


def test_plot_category_distribution_invalid_category(interval_df):
    with pytest.raises(ValueError):
        plot_category_distribution(interval_df, category="C", show=False)


# ----------------------------------------------------------------------
#  Layer 2: plot_multi_category_kde
# ----------------------------------------------------------------------

def test_plot_multi_category_kde_basic(interval_df):
    kde_peaks_map = {"A": [10, 20], "B": [5, 8]}
    fig = plot_multi_category_kde(
        interval_df,
        interval_col="interval_days",
        category_col="cat",
        kde_peaks_map=kde_peaks_map,
        show=False,
    )
    assert isinstance(fig, mplfig.Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Interval (days)"


def test_plot_multi_category_kde_with_client(interval_df):
    fig = plot_multi_category_kde(
        interval_df,
        interval_col="interval_days",
        category_col="cat",
        client_col="client_type",
        show=False,
    )
    assert isinstance(fig, mplfig.Figure)


def test_plot_multi_category_kde_empty_df():
    df_empty = pd.DataFrame({"cat": [], "interval_days": []})
    with pytest.raises(ValueError):
        plot_multi_category_kde(df_empty, show=False)


# ----------------------------------------------------------------------
#  Layer 3: plot_summary_comparison
# ----------------------------------------------------------------------

def test_plot_summary_comparison_basic(dist_report):
    fig = plot_summary_comparison(
        summary_df=dist_report,
        value_col="mean",
        error_col="std",
        peaks_col="kde_peaks",
        show=False,
    )
    assert isinstance(fig, mplfig.Figure)
    ax = fig.axes[0]
    assert "Category" in ax.get_ylabel()


def test_plot_summary_comparison_no_errorbar(dist_report):
    fig = plot_summary_comparison(
        summary_df=dist_report,
        value_col="median",
        error_col=None,
        show_peaks=False,
        show=False,
    )
    assert isinstance(fig, mplfig.Figure)
