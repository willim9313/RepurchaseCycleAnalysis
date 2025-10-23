import pandas as pd
from src.etl import build_intervals
from src.stats import analyze_distribution


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

    # 假設 raw_df 已就位
    interval_df = build_intervals(raw_df, min_purchase=2, category_col="Description").copy()


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




if __name__ == "__main__":
    main()
