import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Iterable, List, Dict

sns.set_style("whitegrid")


def plot_category_distribution(
    df: pd.DataFrame,
    category: str,
    interval_col: str = "interval_days",
    client_col: Optional[str] = None,
    kde_peaks: Optional[List[float]] = None,
    figsize: tuple = (8, 5),
    title_prefix: str = "Purchase Interval Distribution",
    show: bool = True,
) -> plt.Figure:
    """
    單一分類的購買間隔分佈圖：
    - KDE 分佈線
    - Box Plot 顯示集中區
    - Strip Plot 顯示實際樣本散佈
    - 若提供 kde_peaks 則標註主峰位置

    Parameters
    ----------
    df : pd.DataFrame
        interval_df, 至少需包含 interval_col 與 cat。
    category : str
        要繪製的商品分類。
    interval_col : str, default='interval_days'
        間隔欄位名稱。
    client_col : Optional[str]
        若提供，則 strip 顏色依 client_type 區分。
    kde_peaks : Optional[List[float]]
        主要週期峰值位置清單，可從 analyze_distribution 結果取出。
    figsize : tuple, default=(8,5)
        圖片大小。
    title_prefix : str
        標題前綴。
    show : bool
        是否呼叫 plt.show()。

    Returns
    -------
    plt.Figure
        matplotlib Figure 物件。
    """
    use = df.copy()
    use = use[use["cat"] == category]
    use = use[pd.to_numeric(use[interval_col], errors="coerce").notna()]
    use = use[use[interval_col] > 0]

    if use.empty:
        raise ValueError(f"No valid data for category '{category}'.")

    fig, ax = plt.subplots(figsize=figsize)

    # KDE 層
    sns.kdeplot(
        data=use,
        x=interval_col,
        fill=True,
        alpha=0.3,
        linewidth=1.5,
        color="tab:blue",
        ax=ax,
    )

    # Box 層（橫向）
    sns.boxplot(
        data=use,
        x=interval_col,
        orient="h",
        color="tab:gray",
        width=0.2,
        fliersize=2.5,
        ax=ax,
    )

    # Strip 層
    sns.stripplot(
        data=use,
        x=interval_col,
        orient="h",
        size=4,
        alpha=0.6,
        jitter=0.25,
        hue=client_col if client_col else None,
        dodge=False,
        ax=ax,
    )

    # 若有峰值則標註
    if kde_peaks:
        for peak in kde_peaks:
            ax.axvline(peak, color="red", linestyle="--", linewidth=1)
            ax.text(peak, ax.get_ylim()[1] * 0.9, f"{peak:.0f}", color="red", fontsize=9, ha="center")

    # 標題與樣式
    ax.set_title(f"{title_prefix} - {category}")
    ax.set_xlabel("Interval (days)")
    ax.set_ylabel("Density / Samples")
    ax.legend().remove()
    sns.despine()
    plt.tight_layout()

    if show:
        plt.show()

    return fig


# -------------------------------------------------------------------
# 預留未來擴充介面
# -------------------------------------------------------------------

def plot_multi_category_kde(
    df: pd.DataFrame,
    interval_col: str = "interval_days",
    category_col: str = "cat",
    client_col: Optional[str] = None,
    kde_peaks_map: Optional[Dict[str, List[float]]] = None,
    figsize: tuple = (9, 6),
    title: str = "Multi-category Purchase Interval Distribution",
    show: bool = True,
) -> plt.Figure:
    """
    多分類購買週期 KDE 比較圖。

    每個分類畫一條 KDE 曲線，可用顏色區分分類，
    並可選擇在曲線上標註對應的主要峰值。

    Parameters
    ----------
    df : pd.DataFrame
        interval_df (ETL 輸出)，至少需包含 interval_col 與 category_col。
    interval_col : str, default='interval_days'
        間隔欄位名稱。
    category_col : str, default='cat'
        分類欄位。
    client_col : Optional[str]
        若指定，將以不同顏色區分用戶類型而非分類。
    kde_peaks_map : Optional[Dict[str, List[float]]]
        每個分類對應的峰值清單，可從 analyze_distribution 輸出組成：
        例如：{"A": [10, 30], "B": [5]}。
    figsize : tuple, default=(9, 6)
        圖片大小。
    title : str
        圖表標題。
    show : bool
        是否立即顯示圖表。

    Returns
    -------
    plt.Figure
        matplotlib Figure 物件。
    """
    use = df.copy()
    use = use[pd.to_numeric(use[interval_col], errors="coerce").notna()]
    use = use[use[interval_col] > 0]

    if use.empty:
        raise ValueError("No valid interval data to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    # KDE plot
    hue = client_col if client_col else category_col
    sns.kdeplot(
        data=use,
        x=interval_col,
        hue=hue,
        common_norm=False,
        fill=False,
        linewidth=1.8,
        ax=ax,
    )

    # 標註峰值（若有）
    if kde_peaks_map:
        for cat, peaks in kde_peaks_map.items():
            if not peaks:
                continue
            color = sns.color_palette("tab10")[hash(cat) % 10]
            for peak in peaks:
                ax.axvline(peak, color=color, linestyle="--", linewidth=1, alpha=0.6)
                ax.text(
                    peak,
                    ax.get_ylim()[1] * 0.9,
                    f"{peak:.0f}",
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="top",
                    alpha=0.8,
                )

    ax.set_title(title)
    ax.set_xlabel("Interval (days)")
    ax.set_ylabel("Density")
    sns.despine()
    plt.tight_layout()

    if show:
        plt.show()
    return fig



def plot_summary_comparison(
    summary_df: pd.DataFrame,
    category_col: str = "cat",
    value_col: str = "mean",
    error_col: Optional[str] = "std",
    peaks_col: Optional[str] = "kde_peaks",
    figsize: tuple = (9, 6),
    title: str = "Category-level Purchase Cycle Summary",
    show_peaks: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    繪製各分類購買週期的彙整比較圖。
    - 以 mean / median 等統計值為主軸
    - 可加入誤差棒（std / iqr / cv）
    - 可標註主要 KDE 峰值位置

    Parameters
    ----------
    summary_df : pd.DataFrame
        analyze_distribution() 的結果。
    category_col : str, default='cat'
        分類欄位。
    value_col : str, default='mean'
        要繪製的主要統計指標，例如 'mean' 或 'median'。
    error_col : Optional[str], default='std'
        誤差棒欄位，可設為 None 以不顯示。
    peaks_col : Optional[str], default='kde_peaks'
        若 show_peaks=True，則從此欄位取出峰值標註。
    figsize : tuple, default=(9,6)
        圖片大小。
    title : str
        圖表標題。
    show_peaks : bool, default=True
        是否在圖上標出每個分類的主要峰值位置。
    show : bool, default=True
        是否呼叫 plt.show()。

    Returns
    -------
    plt.Figure
        matplotlib Figure 物件。
    """
    df = summary_df.copy()
    df = df[[category_col, value_col] + ([error_col] if error_col else []) + ([peaks_col] if peaks_col else [])]
    df = df.dropna(subset=[value_col])

    # 排序：週期長的放右邊
    df = df.sort_values(by=value_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    # 主圖：bar + error
    sns.barplot(
        data=df,
        x=value_col,
        y=category_col,
        orient="h",
        palette="Blues_d",
        ax=ax,
        errorbar=None,
    )

    # 誤差棒（若指定）
    if error_col and error_col in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            val = row[value_col]
            err = row[error_col]
            if pd.notna(err) and err > 0:
                ax.errorbar(
                    x=val,
                    y=i,
                    xerr=err,
                    fmt="none",
                    ecolor="gray",
                    capsize=3,
                    alpha=0.7,
                )

    # 標註峰值（若有）
    if show_peaks and peaks_col and peaks_col in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            peaks = row[peaks_col]
            if isinstance(peaks, list) and len(peaks) > 0:
                for pk in peaks[:3]:  # 只顯示最多前三個
                    ax.plot(pk, i, "r|", markersize=10, markeredgewidth=1.2)

    ax.set_xlabel(f"{value_col.capitalize()} interval (days)")
    ax.set_ylabel("Category")
    ax.set_title(title)
    sns.despine()
    plt.tight_layout()

    if show:
        plt.show()
    return fig
