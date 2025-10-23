from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats, signal

try:
    # 多峰性統計檢定（若未安裝，程式會自動略過但仍可執行）
    from diptest import diptest as _diptest
    _HAS_DIPTEST = True
except Exception:
    _HAS_DIPTEST = False

try:
    # 高效 KDE (快速 FFTKDE)
    from KDEpy import FFTKDE
    _HAS_KDEPY = True
except Exception:
    _HAS_KDEPY = False


def _iqr(x: np.ndarray) -> float:
    """Interquartile range (75th - 25th)."""
    q75, q25 = np.nanpercentile(x, [75, 25])
    return float(q75 - q25)


def _coefficient_of_variation(x: np.ndarray) -> float:
    """Coefficient of Variation (std / mean). 若 mean=0，回傳 NaN。"""
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    return float(s / m) if m not in (0.0, np.nan) else np.nan


def _shapiro_p(x: np.ndarray, max_n: int = 5000, random_state: int = 42) -> float:
    """
    Shapiro–Wilk 常態性檢定的 p-value。
    Shapiro 在 n>5000 時官方不建議；此處自動下採樣到 max_n。
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    if x.size > max_n:
        rng = np.random.default_rng(random_state)
        x = rng.choice(x, size=max_n, replace=False)
    try:
        _, p = stats.shapiro(x)
    except Exception:
        p = np.nan
    return float(p)


def _dip_test(x: np.ndarray) -> Tuple[float, float]:
    """Hartigan's Dip Test (statistic, p-value)。若未安裝 diptest，回傳 (NaN, NaN)。"""
    x = x[~np.isnan(x)]
    if (not _HAS_DIPTEST) or (x.size < 3):
        return float("nan"), float("nan")
    try:
        stat, p = _diptest(x)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _kde_peaks(
    x: np.ndarray,
    bandwidth: Optional[float] = None,
    grid_size: int = 2048,
    peak_prominence: Optional[float] = None
) -> Tuple[List[float], Dict[str, Any]]:
    """
    使用 KDE 找出分佈主峰位置（以 x 軸單位回傳）。
    - 預設使用 FFTKDE（若無 KDEpy，回退到 scipy.stats.gaussian_kde）。
    - 可設定 peak_prominence 以過濾弱峰。

    Returns
    -------
    peaks_x : List[float]
        KDE 曲線主峰的 x 位置清單。
    meta : Dict[str, Any]
        內含 'grid_x', 'grid_y', 'peaks_idx', 'prominences' 等調試資訊。
    """
    x = x[~np.isnan(x)]
    if x.size == 0:
        return [], {"grid_x": None, "grid_y": None, "peaks_idx": [], "prominences": []}

    # 建立估計網格
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmin == xmax:
        return [float(xmin)], {"grid_x": np.array([xmin]), "grid_y": np.array([1.0]), "peaks_idx": [0], "prominences": [1.0]}

    grid_x = np.linspace(xmin, xmax, grid_size)

    # KDE 估計曲線
    if _HAS_KDEPY:
        kde = FFTKDE(kernel="gaussian", bw=bandwidth or "silverman")
        grid_y = kde.fit(x).evaluate(grid_x)
    else:
        kde = stats.gaussian_kde(x, bw_method=bandwidth or "silverman")
        grid_y = kde(grid_x)

    # 尋找峰
    if peak_prominence is not None:
        peaks_idx, props = signal.find_peaks(grid_y, prominence=peak_prominence)
        prominences = props.get("prominences", np.array([]))
    else:
        peaks_idx = signal.find_peaks(grid_y)[0]
        prominences = np.array([])

    peaks_x = grid_x[peaks_idx].tolist()
    return peaks_x, {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "peaks_idx": peaks_idx.tolist(),
        "prominences": prominences.tolist(),
    }


def summarize_intervals(
    df: pd.DataFrame,
    interval_col: str = "interval_days",
    group_cols: Optional[Iterable[str]] = ("cat",),
    min_samples: int = 2
) -> pd.DataFrame:
    """
    針對購買間隔進行基礎統計摘要（分組可選）。
    - 自動忽略 NaN 與非正數間隔（<=0）以避免噪音。

    Returns
    -------
    pd.DataFrame
        每個群組一列，包含 n, mean, median, std, iqr, cv, skew, kurtosis 等欄位。
    """
    if group_cols is None:
        group_cols = []

    use = df.copy()
    # 僅保留有效間隔
    use[interval_col] = pd.to_numeric(use[interval_col], errors="coerce")
    use = use[use[interval_col].notna() & (use[interval_col] > 0)]

    def _agg(g: pd.Series) -> pd.Series:
        x = g.to_numpy(dtype=float)
        return pd.Series({
            "n": x.size,
            "mean": np.nanmean(x),
            "median": np.nanmedian(x),
            "std": np.nanstd(x, ddof=1) if x.size > 1 else np.nan,
            "iqr": _iqr(x) if x.size > 1 else np.nan,
            "cv": _coefficient_of_variation(x) if x.size > 1 else np.nan,
            "skew": stats.skew(x, bias=False) if x.size > 2 else np.nan,
            "kurtosis": stats.kurtosis(x, fisher=True, bias=False) if x.size > 3 else np.nan,
        })

    if group_cols:
        out = use.groupby(list(group_cols), dropna=False)[interval_col].apply(_agg).unstack()
    else:
        out = _agg(use[interval_col]).to_frame().T

    # 過濾樣本數不足
    out = out[out["n"] >= min_samples].reset_index()
    return out


def analyze_distribution(
    df: pd.DataFrame,
    interval_col: str = "interval_days",
    group_cols: Optional[Iterable[str]] = ("cat",),
    *,
    shapiro_max_n: int = 5000,
    kde_grid_size: int = 2048,
    kde_bandwidth: Optional[float] = None,
    peak_prominence: Optional[float] = None,
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    核心分析：對每個群組的購買間隔做
    1) 基礎統計摘要
    2) 常態性檢定（Shapiro）
    3) 多峰性檢測（Dip Test + KDE找峰）

    Returns
    -------
    pd.DataFrame
        每群一列，包含：
        - n, mean, median, std, iqr, cv, skew, kurtosis
        - shapiro_p
        - dip_stat, dip_p
        - n_kde_peaks, kde_peaks (List[float])
    """
    base = summarize_intervals(
        df,
        interval_col=interval_col,
        group_cols=group_cols,
        min_samples=min_samples,
    )

    if base.empty:
        # 回傳結構化空表
        cols = (list(group_cols) if group_cols else []) + [
            "n", "mean", "median", "std", "iqr", "cv", "skew", "kurtosis",
            "shapiro_p", "dip_stat", "dip_p", "n_kde_peaks", "kde_peaks",
        ]
        return pd.DataFrame(columns=cols)

    # 準備 join key
    key_cols = list(group_cols) if group_cols else []
    out_rows = []

    # 以群組為單位做檢定與找峰
    if key_cols:
        grouped = df.groupby(key_cols, dropna=False)
    else:
        # 只有一個群組（整體）
        df = df.copy()
        df["__all__"] = "__all__"
        grouped = df.groupby(["__all__"], dropna=False)

    for key, g in grouped:
        x = pd.to_numeric(g[interval_col], errors="coerce").to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        x = x[x > 0]  # 僅正值

        if x.size < min_samples:
            continue

        shapiro_p = _shapiro_p(x, max_n=shapiro_max_n)
        dip_stat, dip_p = _dip_test(x)
        peaks, _meta = _kde_peaks(
            x,
            bandwidth=kde_bandwidth,
            grid_size=kde_grid_size,
            peak_prominence=peak_prominence,
        )

        # 找對應的基礎統計列
        if key_cols:
            if not isinstance(key, tuple):
                key = (key,)
            base_row = base.loc[(base[key_cols] == pd.Series(key, index=key_cols)).all(axis=1)]
        else:
            base_row = base

        if base_row.empty:
            continue

        row = base_row.iloc[0].to_dict()
        # 補上檢定/峰值資訊
        row.update({
            "shapiro_p": shapiro_p,
            "dip_stat": dip_stat,
            "dip_p": dip_p,
            "n_kde_peaks": int(len(peaks)),
            "kde_peaks": [float(p) for p in peaks],
        })

        # 附上 key
        for i, kcol in enumerate(key_cols):
            row[kcol] = key[i]
        out_rows.append(row)

    # 組裝輸出（欄位順序調整）
    if not out_rows:
        cols = (list(group_cols) if group_cols else []) + [
            "n", "mean", "median", "std", "iqr", "cv", "skew", "kurtosis",
            "shapiro_p", "dip_stat", "dip_p", "n_kde_peaks", "kde_peaks",
        ]
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(out_rows)
    ordered_cols = (list(group_cols) if group_cols else []) + [
        "n", "mean", "median", "std", "iqr", "cv", "skew", "kurtosis",
        "shapiro_p", "dip_stat", "dip_p", "n_kde_peaks", "kde_peaks",
    ]
    # 補齊缺少的欄位
    for c in ordered_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[ordered_cols].sort_values(by=(list(group_cols) if group_cols else ["n"]), ascending=False).reset_index(drop=True)
