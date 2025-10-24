import pandas as pd
from src.etl import build_intervals
from src.stats import analyze_distribution
from src.viz import plot_multi_category_kde
from src.viz import plot_summary_comparison


def main():
    print("Hello from repurchasecycleanalysis!")
    print("Running ETL...")

    raw_df = pd.read_csv(
        "data/raw/uk_retail_transactions.csv",
        encoding='utf-8',
        encoding_errors='ignore',
        on_bad_lines="skip",
        delimiter=','
    )
    # 將 InvoiceDate 轉為 datetime
    raw_df['purchase_date'] = pd.to_datetime(raw_df['InvoiceDate'], errors='coerce')
    raw_df['uid'] = raw_df['CustomerID'].astype(str)
    raw_df['it_name'] = raw_df['Description'].astype(str)
    
    print('raw_df shape:', raw_df.shape)

    # 假設 raw_df 已就位
    interval_df = build_intervals(
        raw_df,
        min_purchase=2,
        category_col=""
    ).copy()

    # 以分類為群組（可改成 None 做整體）
    report = analyze_distribution(
        interval_df,
        interval_col="interval_days",
        group_cols=("Description",),
        shapiro_max_n=5000,
        kde_grid_size=2048,
        peak_prominence=None,   # 若想過濾弱峰再設數值，例如 1e-4
    )
    print(report.head())

    report.to_csv("data/processed/interval_analysis_report.csv", index=False)


if __name__ == "__main__":
    main()



# 1. 建立 interval_df
interval_df = build_intervals(raw_df, min_purchase=2, category_col="cat")

# 2. 產出統計報告
dist_report = analyze_distribution(interval_df)

# 3. 將每個分類的峰值整理成 dict
kde_peaks_map = {
    row["cat"]: row["kde_peaks"] for _, row in dist_report.iterrows()
    if isinstance(row["kde_peaks"], list)
}

# 4. 畫多分類 KDE
plot_multi_category_kde(
    interval_df,
    interval_col="interval_days",
    category_col="cat",
    kde_peaks_map=kde_peaks_map,
)



# dist_report 為 analyze_distribution 的結果
plot_summary_comparison(
    summary_df=dist_report,
    value_col="mean",     # 可換成 median
    error_col="std",      # 或 cv
    peaks_col="kde_peaks",
)