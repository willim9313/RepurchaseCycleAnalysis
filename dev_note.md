# ======================================================
# Repurchase Cycle Distribution Analysis - Module Spec v1.3
# ======================================================
# 各模組具備清晰 I/O 與可測試參數，支援 pytest / CLI pipeline 整合
# ------------------------------------------------------

modules:

  - id: 00_interval_derivation
    name: "交易間隔計算"
    input_schema:
      dataframe:
        columns:
          - uid: str           # 使用者識別碼（如 UserId / CustomerID）
          - cat: str           # 類別欄位（如 ItemName / Category）
          - order_date: datetime  # 訂單日期時間
        optional_columns:
          - order_no: str      # 訂單編號（可選）
          - quantity: float    # 數量（可選，保留）
          - unit_price: float  # 單價（可選，保留）
    output_schema:
      interval_df: DataFrame
        columns:
          - uid: str
          - cat: str
          - order_date: datetime
          - prev_order_date: datetime
          - interval_days: float
          - purchase_seq: int
      conversion_summary:
        type: dict
        fields:
          total_transactions: int
          unique_users: int
          unique_categories: int
          output_intervals: int
          single_purchase_dropped: int
    key_parameters:
      uid_col: "uid"              # 使用者欄位名稱
      cat_col: "cat"              # 類別欄位名稱
      date_col: "order_date"      # 日期欄位名稱
      groupby_cols: ["uid", "cat"]  # 分組計算間隔的欄位
      keep_first_purchase: false  # 是否保留首次購買（interval=NaN）
      date_format: null           # 日期格式（若需手動解析）
      extra_cols: []              # 保留的額外欄位
    methods:
      small: "pandas groupby + shift 逐組計算"
      medium: "pandas 向量化 sort + groupby transform"
      large: "DuckDB window function (LAG)"
    description: >
      將原始交易紀錄（transaction-based）轉換為購買間隔資料（interval-based）。
      對每個 (uid, cat) 組合，依時間排序後計算連續購買之間的天數差。
      此模組為 pipeline 的最前端，輸出資料供後續 data_cleaning 模組使用。

  - id: 01_data_cleaning
    name: "資料清理與檢查"
    input_schema:
      dataframe:
        columns:
          - uid: str
          - cat: str
          - interval_days: float
    output_schema:
      cleaned_df: DataFrame
      discard_summary:
        type: dict
        fields:
          total_rows: int
          removed_negatives: int
          removed_missing: int
          removed_outliers: int
    key_parameters:
      remove_negatives: true
      missing_strategy: "drop"  # or "impute_mean"
      outlier_method: "IQR"     # options: IQR, MAD, quantile
      outlier_threshold: 1.5
    methods:
      small: "逐筆檢查 + IQR 過濾"
      medium: "pandas 向量化檢查"
      large: "DuckDB SQL 條件篩選 / 分桶"
    description: >
      清理極端與錯誤值，保留捨棄紀錄；為所有後續模組的基礎。

  - id: 02_transform
    name: "尺度轉換"
    input_schema:
      cleaned_df: DataFrame
    output_schema:
      transformed_df: DataFrame
      transform_meta:
        type: dict
        fields:
          method: str
          skewness_before: float
          skewness_after: float
    key_parameters:
      method_candidates: ["log1p", "yeo_johnson", "none"]
      auto_select_by_skewness: true
      skew_threshold: 2.0
    methods:
      small: "逐一測試多種 transform 並記錄效果"
      medium: "log1p 為預設"
      large: "Yeo–Johnson（允許0與負值）"
    description: >
      對 interval_days 做平滑處理，降低偏態。所有模型內部可用轉換後尺度，
      但最後輸出需反轉至原尺度。

  - id: 03_visualization
    name: "探索性視覺化"
    input_schema:
      transformed_df: DataFrame
    output_schema:
      plots_dir: "Path to PNG/PDF"
      summary_stats:
        mean: float
        median: float
        std: float
        skew: float
    key_parameters:
      sample_ratio: 0.05
      kde_bandwidths: [0.3, 0.6, 1.0]
      plot_types: ["hist_kde", "violin", "box", "cdf"]
    methods:
      small: "全資料繪製"
      medium: "抽樣繪製"
      large: "分桶近似"
    description: >
      提供人工檢查資料形狀的視覺化；不影響自動化結果。
      可同時輸出多個 bandwidth 的 KDE 疊圖。

  - id: 04_unimodality_test
    name: "單峰性檢定"
    input_schema:
      transformed_df: DataFrame
    output_schema:
      unimodality_test_result:
        dip_p: float
        method_used: str
        decision: str  # unimodal / multimodal
    key_parameters:
      alpha: 0.05
      max_sample_for_test: 10000
    methods:
      small: ["Hartigan’s Dip Test", "Silverman’s Test"]
      medium: ["Dip (subsampled)", "Silverman"]
      large: ["KDE 極值法", "Smoothness/EMD heuristic"]
    description: >
      檢驗資料分布是否具顯著多峰特性。
      若 p < alpha 或出現多個極值 → 進入峰偵測階段。

  - id: 05_peak_detection
    name: "峰值偵測"
    input_schema:
      transformed_df: DataFrame
    output_schema:
      peaks_table:
        - pos: float
          height: float
          width: float
          prominence: float
      kde_plot_with_peaks: str  # 路徑
    key_parameters:
      bandwidth: 0.5
      prominence_min: 0.01
      height_min: 0.001
      grid_size: 512
    methods:
      small: ["KDE + scipy.find_peaks"]
      medium: ["KDEpy FFT + argrelmax"]
      large: ["MeanShift clustering on density"]
    description: >
      找出平滑密度中的主要峰位置與特徵指標。可視為潛在行為週期中心。

  - id: 06_modality_quantification
    name: "模態數量量化"
    input_schema:
      transformed_df: DataFrame
    output_schema:
      gmm_result:
        best_n_components: int
        aic_scores: list
        bic_scores: list
      consistency_check:
        kde_n_peaks: int
        gmm_n_components: int
        status: str  # consistent / inconsistent
    key_parameters:
      k_range: [1, 6]
      selection_metric: "BIC"
    methods:
      small: ["GMM 全迴圈 (1..K_max)"]
      medium: ["Subsample GMM + AIC/BIC"]
      large: ["DP-GMM / Variational Inference"]
    description: >
      以模型化方式估計最可能的模態數 (K)，與峰偵測結果交叉驗證。

  - id: 07_stability_assessment
    name: "穩定性檢驗"
    input_schema:
      transformed_df: DataFrame
      peaks_table: list
    output_schema:
      stable_peaks_table:
        - pos: float
          support_ratio: float
      stability_plot: str
    key_parameters:
      n_bootstrap: 100
      sample_fraction: 0.8
      support_threshold: 0.6
    methods:
      small: ["Bootstrap KDE"]
      medium: ["Bootstrap + Subsample Consensus"]
      large: ["Multi-bandwidth Consistency (快速近似)"]
    description: >
      驗證峰是否穩定存在於重抽樣下。可提供每個峰的支持率指標。

  - id: 08_reporting
    name: "結果整合與匯出"
    input_schema:
      unimodality_test_result: dict
      peaks_table: list
      gmm_result: dict
      stable_peaks_table: list
      summary_stats: dict
    output_schema:
      summary_json:
        cat: str
        n: int
        mean: float
        median: float
        std: float
        skew: float
        dip_p: float
        n_peaks: int
        peaks: list
        stable_peaks: list
        best_n_components: int
        consistency: str
        PEP: str
      figures:
        - distribution_plot
        - stability_plot
        - gmm_comparison_plot
    key_parameters:
      output_dir: "./reports"
      export_formats: ["json", "pdf", "png"]
    description: >
      匯整所有結果並輸出為 summary.json 與圖表。
      包含「PEP（建議）」欄位：
        - 建議分群（若多峰穩定）
        - 建議維持單群（若單峰）
        - 建議重抽樣或人工審查（若結果矛盾）
    Note: PEP Peak Existence Probability（峰值存在機率）

---

# ======================================================
# Global Controls
# ======================================================
pipeline_controller:
  auto_scale_by_data_size: true
  data_size_thresholds:
    small: 1e4
    medium: 1e6
  parallel_execution:
    enabled: true
    strategy: "per-category"
  random_seed: 42
  logging:
    level: "INFO"
    save_path: "./logs"

# ======================================================
# Test / Validation Guidelines
# ======================================================
pytest_guidelines:
  - test_data_cleaning:
      checks: ["negative_removed", "discard_log_valid"]
  - test_transform:
      checks: ["skew_reduction", "reversible_transform"]
  - test_unimodality_test:
      checks: ["p_value_range", "method_switch_by_size"]
  - test_peak_detection:
      checks: ["peak_count_nonzero", "position_in_range"]
  - test_stability:
      checks: ["support_ratio_in_0_1", "stability_plot_exists"]
  - test_reporting:
      checks: ["json_schema_valid", "PEP_exists"]


# ===============
support information
https://python-graph-gallery.com/raincloud-plot-with-matplotlib-and-ptitprince/


# ===============
from repurchase_cycle.modules.interval_calculation import run_interval_calculation
import pandas as pd

# 載入原始交易資料
df = pd.read_csv("data/raw/sample_retail_transactions.csv")

# 轉換為間隔資料
interval_df, summary = run_interval_calculation(
    df,
    mode="medium",
    mod_params={
        "uid_col": "UserId",
        "cat_col": "ItemName",  # 或 "Category"
        "date_col": "OrderDate",
        "groupby_cols": ["UserId", "ItemName"],
        "keep_first_purchase": False,
    }
)

print(summary)
# 輸出：interval_df 包含 uid, cat, order_date, prev_order_date, interval_days, purchase_seq