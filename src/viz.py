""" Visualization functions for purchase interval analysis. """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, List, Dict

sns.set_style("whitegrid")


# def plot_category_distribution(
#     df: pd.DataFrame,
#     category: str,
#     interval_col: str = "interval_days",
#     client_col: Optional[str] = None,
#     kde_peaks: Optional[List[float]] = None,
#     figsize: tuple = (8, 5),
#     title_prefix: str = "Purchase Interval Distribution",
#     show: bool = True,
# ) -> plt.Figure:
#     """
#     單一分類的購買間隔分佈圖：
#     - KDE 分佈線
#     - Box Plot 顯示集中區
#     - Strip Plot 顯示實際樣本散佈
#     - 若提供 kde_peaks 則標註主峰位置

#     Parameters
#     ----------
#     df : pd.DataFrame
#         interval_df, 至少需包含 interval_col 與 cat。
#     category : str
#         要繪製的商品分類。
#     interval_col : str, default='interval_days'
#         間隔欄位名稱。
#     client_col : Optional[str]
#         若提供，則 strip 顏色依 client_type 區分。
#     kde_peaks : Optional[List[float]]
#         主要週期峰值位置清單，可從 analyze_distribution 結果取出。
#     figsize : tuple, default=(8,5)
#         圖片大小。
#     title_prefix : str
#         標題前綴。
#     show : bool
#         是否呼叫 plt.show()。

#     Returns
#     -------
#     plt.Figure
#         matplotlib Figure 物件。
#     """
#     use = df.copy()
#     use = use[use["Category"] == category]
#     use = use[pd.to_numeric(use[interval_col], errors="coerce").notna()]
#     use = use[use[interval_col] > 0]

#     if use.empty:
#         raise ValueError(f"No valid data for category '{category}'.")

#     fig, ax = plt.subplots(figsize=figsize)

#     # KDE 層
#     sns.kdeplot(
#         data=use,
#         x=interval_col,
#         fill=True,
#         alpha=0.3,
#         linewidth=1.5,
#         color="tab:blue",
#         ax=ax,
#     )

#     # Box 層（橫向）
#     sns.boxplot(
#         data=use,
#         x=interval_col,
#         orient="h",
#         color="tab:gray",
#         width=0.2,
#         fliersize=2.5,
#         ax=ax,
#     )

#     # Strip 層
#     sns.stripplot(
#         data=use,
#         x=interval_col,
#         orient="h",
#         size=4,
#         alpha=0.6,
#         jitter=0.25,
#         hue=client_col if client_col else None,
#         dodge=False,
#         ax=ax,
#     )

#     # 若有峰值則標註
#     if kde_peaks:
#         for peak in kde_peaks:
#             ax.axvline(peak, color="red", linestyle="--", linewidth=1)
#             ax.text(peak, ax.get_ylim()[1] * 0.9, f"{peak:.0f}", color="red", fontsize=9, ha="center")

#     # 標題與樣式
#     ax.set_title(f"{title_prefix} - {category}")
#     ax.set_xlabel("Interval (days)")
#     ax.set_ylabel("Density / Samples")
#     ax.legend().remove()
#     sns.despine()
#     plt.tight_layout()

#     if show:
#         plt.show()

#     return fig

def plot_category_distribution(
    df: pd.DataFrame,
    category: str,
    interval_col: str = "interval_days",
    client_col: Optional[str] = None,
    kde_peaks: Optional[List[float]] = None,
    figsize: tuple = (10, 6),
    title_prefix: str = "Purchase Interval Distribution",
    show: bool = True,
) -> plt.Figure:
    """
    單一分類的購買間隔分佈圖：
    - KDE 分佈線 (向上，像雲朵)
    - Box Plot 顯示集中區 (中間)
    - Strip Plot 顯示實際樣本散佈 (向下，像雨滴)
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
    figsize : tuple, default=(10,6)
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
    use = use[use["Category"] == category]
    use = use[pd.to_numeric(use[interval_col], errors="coerce").notna()]
    use = use[use[interval_col] > 0]

    if use.empty:
        raise ValueError(f"No valid data for category '{category}'.")

    fig, ax = plt.subplots(figsize=figsize)
    
    import numpy as np
    from matplotlib.patches import Rectangle
    from scipy import stats

    # 設定三個區域的 y 位置
    kde_base = 2.0      # KDE 在上方
    box_base = 0.0      # Box 在中間 (y=0)
    strip_base = -2.0   # Strip 在下方

    # 1. KDE 層 (向上，像雲朵)
    # 計算 KDE
    kde_x = np.linspace(use[interval_col].min(), use[interval_col].max(), 100)
    kde = stats.gaussian_kde(use[interval_col])
    kde_y = kde(kde_x)
    
    # 正規化 KDE 並放置在上方區域
    kde_y_normalized = kde_y / kde_y.max() * 0.8  # 最高 0.8 單位
    
    ax.fill_between(kde_x, kde_base, kde_base + kde_y_normalized, 
                   alpha=0.3, color="tab:blue", label="KDE")
    ax.plot(kde_x, kde_base + kde_y_normalized, 
           color="tab:blue", linewidth=1.5)

    # 若有峰值則標註
    if kde_peaks:
        for peak in kde_peaks:
            ax.axvline(peak, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(peak, kde_base + 1.0, f"{peak:.0f}", 
                   color="red", fontsize=9, ha="center")

    # 2. Box 層（中間）
    q1 = use[interval_col].quantile(0.25)
    q2 = use[interval_col].quantile(0.5)  # median
    q3 = use[interval_col].quantile(0.75)
    iqr = q3 - q1
    whisker_low = max(use[interval_col].min(), q1 - 1.5 * iqr)
    whisker_high = min(use[interval_col].max(), q3 + 1.5 * iqr)
    
    box_height = 0.3
    
    # 繪製箱線圖元素
    # 箱子
    box = Rectangle((q1, box_base - box_height/2), q3 - q1, box_height, 
                   linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.7)
    ax.add_patch(box)
    
    # 中位數線
    ax.plot([q2, q2], [box_base - box_height/2, box_base + box_height/2], 
           'k-', linewidth=2)
    
    # 鬚線
    ax.plot([whisker_low, q1], [box_base, box_base], 'k-', linewidth=1)
    ax.plot([q3, whisker_high], [box_base, box_base], 'k-', linewidth=1)
    ax.plot([whisker_low, whisker_low], [box_base - box_height/4, box_base + box_height/4], 'k-', linewidth=1)
    ax.plot([whisker_high, whisker_high], [box_base - box_height/4, box_base + box_height/4], 'k-', linewidth=1)

    # 3. Strip 層 (向下，像雨滴)
    # 創建 jitter 效果
    np.random.seed(42)  # 確保一致性
    jitter = np.random.normal(0, 0.15, len(use))  # 垂直抖動
    
    # 顏色設定
    if client_col and client_col in use.columns:
        colors = use[client_col].map({'A': 'tab:orange', 'B': 'tab:green'}).fillna('tab:orange')
    else:
        colors = 'tab:orange'
    
    ax.scatter(use[interval_col], strip_base + jitter, 
              s=20, alpha=0.6, c=colors, edgecolors='none')

    # 設置樣式
    ax.set_title(f"{title_prefix} - {category}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Interval (days)", fontsize=12)
    
    # 隱藏 y 軸刻度和標籤，因為 y 軸沒有實際意義
    ax.set_yticks([])
    ax.set_ylabel("")
    
    # 設定 y 軸範圍
    ax.set_ylim(-3, 3.5)
    
    # 添加三個區域的標籤
    ax.text(ax.get_xlim()[0] * 0.98, kde_base + 0.4, "Distribution", 
           fontsize=10, ha="left", va="center", color="tab:blue", fontweight='bold')
    ax.text(ax.get_xlim()[0] * 0.98, box_base, "Summary", 
           fontsize=10, ha="left", va="center", color="gray", fontweight='bold')
    ax.text(ax.get_xlim()[0] * 0.98, strip_base, "Samples", 
           fontsize=10, ha="left", va="center", color="tab:orange", fontweight='bold')
    
    # 添加區域分隔線
    ax.axhline(y=1, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axhline(y=-1, color='lightgray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    sns.despine(left=True)  # 移除左邊框線
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
    category_col: str = "Category",
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

    # 明確關閉之前的圖表，避免重複繪製
    plt.close('all')

    fig, ax = plt.subplots(figsize=figsize)

    # KDE plot
    hue = client_col if client_col else category_col
    categories = use[hue].unique()
    palette = sns.color_palette("tab10", n_colors=len(categories))
    
    sns.kdeplot(
        data=use,
        x=interval_col,
        hue=hue,
        palette=palette,
        common_norm=False,
        fill=False,
        linewidth=1.8,
        ax=ax,
    )

    # 標註峰值（若有）
    if kde_peaks_map:
        # 獲取 seaborn 實際使用的顏色        
        # 获取图例中的颜色信息，确保与KDE线条颜色完全一致
        legend = ax.get_legend()
        if legend:
            color_map = {}
            for i, text in enumerate(legend.get_texts()):
                cat_name = text.get_text()
                color = legend.legend_handles[i].get_color()
                color_map[cat_name] = color
        else:
            color_map = {cat: palette[i] for i, cat in enumerate(sorted(categories))}
        for cat, peaks in kde_peaks_map.items():
            if not peaks:
                continue
            # color = sns.color_palette("tab10")[hash(cat) % 10]
            color = color_map[cat]

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
