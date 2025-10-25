import pandas as pd
from src.etl import build_intervals
from src.stats import analyze_distribution
from src.viz import plot_multi_category_kde
from src.viz import plot_summary_comparison
from src.viz import plot_category_distribution

INPUT_FILE_PATH = "data/raw/sample_retail_transactions.csv"
CATEGORY_COL = "Category"
SAVE_INTERVAL_DATA_PATH = "data/processed/interval_data.csv"
SAVE_REPORT_PATH = "data/processed/interval_analysis_report.csv"


def main():
    print("="*50, '\nRepurchase Cycle Analysis\n', "="*50)

    # Load Data Process
    raw_df = pd.read_csv(
        INPUT_FILE_PATH,
        encoding='utf-8',
        encoding_errors='ignore',
        on_bad_lines="skip",
        delimiter=','
    )

    print('raw_df shape:', raw_df.shape)

    interval_df = build_intervals(
        raw_df,
        min_purchase=2,
        category_col=CATEGORY_COL
    ).copy()

    interval_df.to_csv(SAVE_INTERVAL_DATA_PATH, index=False)

    # Analyze Distribution
    report = analyze_distribution(
        interval_df,
        interval_col="interval_days",
        group_cols=(CATEGORY_COL,),
        shapiro_max_n=5000,
        kde_grid_size=2048,
        peak_prominence=None
    )

    print("\nDistribution Analysis Report (head):")
    print(report.head())

    # Save Report
    report.to_csv(SAVE_REPORT_PATH, index=False)

    # Visualization
    kde_peaks_map = {
        row[CATEGORY_COL]: row["kde_peaks"] for _, row in report.iterrows()
        if isinstance(row["kde_peaks"], list)
    }

    plot_category_distribution(
        interval_df,
        category="Groceries",
        interval_col="interval_days",
        client_col=None,
        kde_peaks=kde_peaks_map.get("Groceries", None)
    )

    # plot_multi_category_kde(
    #     interval_df,
    #     interval_col="interval_days",
    #     category_col=CATEGORY_COL,
    #     kde_peaks_map=kde_peaks_map
    # )

    # plot_summary_comparison(
    #     summary_df=report,
    #     category_col=CATEGORY_COL,
    #     value_col="mean",     # 可換成 median
    #     error_col="std",      # 或 cv
    #     peaks_col="kde_peaks",
    # )


if __name__ == "__main__":
    main()
