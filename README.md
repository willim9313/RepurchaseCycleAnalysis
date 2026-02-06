# Repurchase Cycle Analysis Pipeline

A modular, scalable Python pipeline for analyzing customer repurchase cycles across product categories. The system automatically detects distribution modality (unimodal, multimodal), identifies peak repurchase intervals, and assesses the stability of detected patterns through bootstrap validation.

## Purpose

This pipeline helps businesses understand customer purchasing behavior by:

- **Identifying repurchase patterns**: Detect when customers typically repurchase products
- **Segmenting by category**: Analyze different product categories independently
- **Quantifying modality**: Determine if repurchase cycles follow single or multiple patterns
- **Validating stability**: Ensure detected patterns are statistically robust

## Features

- **Adaptive Processing Modes**: Automatically selects optimal algorithms based on data size (small/medium/large)
- **Modular Architecture**: 9 independent, testable modules with clear interfaces
- **Multi-modal Detection**: Supports unimodal and multimodal distribution analysis
- **Bootstrap Stability Assessment**: Validates peak detection reliability
- **Comprehensive Visualization**: Generates raincloud plots, KDE, violin, box plots, and CDF charts
- **Flexible Configuration**: YAML-based configuration with sensible defaults
- **Multiple Output Formats**: JSON reports, PNG visualizations, detailed logs
- **Automatic Interval Conversion**: Converts raw transaction data to repurchase intervals on-the-fly

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Controller                               │
│                    (Mode Selection & Orchestration)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   ┌─────────┐               ┌─────────────┐             ┌───────────┐
   │  Small  │               │   Medium    │             │   Large   │
   │ < 10K   │               │ 10K - 1M    │             │   > 1M    │
   └─────────┘               └─────────────┘             └───────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Module Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
|  0. Interval Derivation   →   1. Data Cleaning   →   2. Transform           |
│         │                           │                     │                 │
│         ▼                           ▼                     ▼                 │
|  3. Visualization                                                           |
│         │                                                                   |
|         ▼                                                                   |
|  4. Unimodality Test                                                        │
│         │                                                                   │
│    ┌────┴────┐                                                              │
│    │ Unimodal│ ───────────────────────────────────────► 8. Reporting        │
│    └─────────┘                                                              │
│         │                                                                   │
│    ┌────┴──────┐                                                            │
│    │ Multimodal│                                                            │
│    └───────────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  5. Peak Detection  →  6. Modality Quantification  →  7. Stability          │
│                                                          Assessment         │
│                                                               │             │
│                                                               ▼             │
│                                                       8. Reporting          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Directory

```
RepurchaseCycleAnalysis/
├── repurchase_cycle/                 # Main package
│   ├── __init__.py
│   ├── __main__.py                   # CLI entry point
│   ├── pipeline.py                   # Pipeline orchestration
│   ├── config_loader.py              # YAML configuration loader
│   ├── logging_utils.py              # Logging configuration
│   └── modules/                      # Processing modules
│       ├── data_loader.py            # Multi-format data loading
│       ├── interval_derivation.py    # Transaction → interval conversion
│       ├── data_cleaning.py          # Outlier removal & missing handling
│       ├── transform.py              # Data transformation (log1p, sqrt, etc.)
│       ├── visualization.py          # Plot generation
│       ├── unimodality_test.py       # Dip test for modality detection
│       ├── peak_detection.py         # KDE peak identification
│       ├── modality_quantification.py # GMM-based mode counting
│       ├── stability_assessment.py   # Bootstrap peak validation
│       ├── reporting.py              # Report generation
│       └── ptitprince.py             # Raincloud plot (adapted from PtitPrince)
│
├── scripts/                          # Utility scripts
│   ├── main.py                       # ETL runner
│   ├── sample_data_etl.py            # Interval calculation from transactions
│   └── generate_sample_data.py       # Synthetic data generator
│
├── tests/                            # Test suite
│   ├── test_visualization.py
│   ├── test_pipeline_integration.py
│   └── ...
│
├── configs/                           # Configuration files
│   └── default_config.yml            # Default pipeline settings
│
├── data/
│   ├── raw/                          # Raw transaction data
│   └── processed/                    # Processed interval data
│
└── reports/                          # Generated outputs
    ├── summary_all.json              # Aggregated results
    ├── complete_report_all.json      # Detailed analysis
    ├── validation_plots/             # Peak detection visualizations
    └── separate_reports/             # Per-category reports
```

## Module Descriptions

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| **interval_derivation** | Convert transaction data to repurchase intervals (days between purchases) | Raw transaction DataFrame | Interval DataFrame, conversion summary |
| **data_loader** | Load data from CSV, Parquet, Excel, JSON, Feather, Pickle | File path | DataFrame |
| **data_cleaning** | Remove negatives, handle missing values, filter outliers (IQR/MAD/Quantile) | Raw DataFrame | Cleaned DataFrame, discard summary |
| **transform** | Apply transformations (log1p, sqrt, box-cox) to reduce skewness | Cleaned DataFrame | Transformed DataFrame, transform metadata |
| **visualization** | Generate distribution plots (hist_kde, violin, box, cdf, raincloud) | Transformed DataFrame | Plot directory, summary statistics |
| **unimodality_test** | Hartigan's dip test for unimodality detection | Transformed DataFrame | Decision (unimodal/multimodal), p-value |
| **peak_detection** | Identify peaks in KDE density estimate using scipy.signal | Transformed DataFrame | Peaks table (pos, height, width, prominence), KDE plot |
| **modality_quantification** | Fit GMM models (k=1..n), select best k via BIC/AIC | Transformed DataFrame | Best k, model parameters, consistency check |
| **stability_assessment** | Bootstrap validation of detected peaks with support ratio | Transformed DataFrame, peaks table | Stable peaks table, stability plot |
| **reporting** | Compile results into structured JSON reports | All module outputs | Brief summary, detailed report |

## Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Setup

**Using uv (recommended):**

```bash
# Clone repository
git clone <repository-url>
cd RepurchaseCycleAnalysis

# Install with uv
uv sync
```

**Using pip:**

```bash
# Clone repository
git clone <repository-url>
cd RepurchaseCycleAnalysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Dependencies

Core dependencies include:
- `pandas`, `numpy` - Data manipulation
- `scipy` - Statistical tests (dip test, KDE, signal processing)
- `scikit-learn` - GMM fitting for modality quantification
- `matplotlib`, `seaborn` - Visualization
- `duckdb` - Large data processing (SQL-based filtering)
- `pyyaml` - Configuration parsing
- `pytest` - Testing

## Usage

### Quick Start

1. **Generate sample data:**

```bash
python scripts/generate_sample_data.py --size all
```

This generates 4 datasets with different distribution characteristics:
- **Small** (< 10K): Electronics - single peak (~90 days)
- **Medium** (10K-1M): Groceries - double peak (7 & 21 days)
- **Medium Uniform**: Stationery - uniform distribution (5-60 days)
- **Large** (> 1M): Supplements - triple peak (30, 60, 90 days)

2. **Prepare interval data from transactions:**

```bash
python scripts/main.py
```

3. **Run the analysis pipeline:**

```bash
uv run python -m repurchase_cycle \
    --input-path ./data/processed/interval_data.csv \
    --output-dir ./reports
```

### Configuration

The pipeline uses YAML configuration files. Create a custom config or modify the defaults:

```yaml
# configs/default_config.yml
pipeline_controller:
  auto_scale_by_data_size: true
  data_size_thresholds:
    small: 1e4        # Use small mode for < 10K rows
    medium: 1e6       # Use medium mode for 10K - 1M rows
  random_seed: 42
  reports_path: "./reports"
  logging:
    level: "INFO"
    save_path: "./logs"

modules:
  interval_derivation:
    uid_col: "UserId"
    cat_col: "Category"
    date_col: "OrderDate"
    groupby_cols: ["UserId", "Category"]
    keep_first_purchase: false
    date_format: null
    extra_cols: []
    min_intervals_per_group: 2

  data_cleaning:
    remove_negatives: true
    missing_strategy: "drop"          # or "impute_mean"
    outlier_method: "IQR"             # "IQR", "MAD", "QUANTILE"
    outlier_threshold: 1.5
    quantile_bounds: [0.05, 0.95]     # for QUANTILE method
    min_group_size_for_stats: 3       # Fallback for small groups

  transform:
    method_candidates: ["log1p", "yeo_johnson", "none"]
    auto_select_by_skewness: true
    skew_threshold: 2.0

  visualization:
    sample_ratio: 0.05
    kde_bandwidths: [0.3, 0.6, 1.0]
    plot_types: ["raincloud"]
    orient: "h"
    palette: "Set2"
    sigma: 0.2                        # Jitter for raincloud

  unimodality_test:
    alpha: 0.05

  peak_detection:
    grid_size: 512
    prominence_min: 0.01

  modality_quantification:
    k_range: [1, 5]
    selection_metric: "BIC"           # or "AIC"

  stability_assessment:
    n_bootstrap: 100
    sample_fraction: 0.8
    support_threshold: 0.6

  reporting:
    provide_details: true
    separate_category_report: true
```

Run with custom config:

```bash
uv run python -m repurchase_cycle \
    --input-path ./data/processed/interval_data.csv \
    --config ./config/my_config.yml \
    --output-dir ./reports
```

### Custom Usage

**Programmatic API:**

```python
import pandas as pd
from repurchase_cycle.pipeline import run_all_categories, run_category_pipeline
from repurchase_cycle.config import load_config

# Load configuration
cfg = load_config("./configs/default_config.yml")

# Load interval data
df = pd.read_csv("./data/processed/interval_data.csv")

# Run full pipeline for all categories
run_all_categories(df, cfg)

# Or run for specific categories (comma-separated)
run_all_categories(df, cfg, cats=["Electronics", "Groceries"])

# Or analyze a single category
df_electronics = df[df["cat"] == "Electronics"]
result = run_category_pipeline(df_electronics, "Electronics", cfg)
```

**Using individual modules:**

```python
from repurchase_cycle.modules.data_cleaning import run_data_cleaning
from repurchase_cycle.modules.transform import run_transform
from repurchase_cycle.modules.visualization import run_visualization
from repurchase_cycle.modules.unimodality_test import run_unimodality_test
from repurchase_cycle.modules.interval_derivation import run_interval_calculation

# Step 0: Convert transactions to intervals
interval_df, summary = run_interval_calculation(
    df,
    mode="small",
    mod_params={"uid_col": "UserId", "cat_col": "Category", "date_col": "OrderDate"}
)

# Step 1: Clean data
cleaned_df, discard_summary = run_data_cleaning(
    interval_df,
    mode="small",
    mod_params={"outlier_method": "IQR", "outlier_threshold": 1.5}
)

# Step 2: Transform
transformed_df, transform_meta = run_transform(
    cleaned_df,
    mode="small",
    mod_params={"method_candidates": ["log1p", "none"]}
)

# Step 3: Visualize
plots_dir, stats = run_visualization(
    transformed_df,
    mode="small",
    viz_prefix="electronics",
    output_dir="./reports"
)

# Step 4: Test unimodality
result = run_unimodality_test(transformed_df, mode="small")
print(f"Decision: {result['decision']}, p-value: {result['dip_p']}")
```

## Data Requirements

### Input Data Format

The pipeline expects **raw transaction data** with at least three columns:

| Column | Type | Description | Default Name |
|--------|------|-------------|---------------|
| User ID | string | Unique user/customer identifier | `UserId` |
| Category | string | Product category name | `Category` |
| Order Date | datetime | Purchase date | `OrderDate` |

**Example transaction CSV:**

```csv
UserId,Category,OrderDate,Amount
U000001,Electronics,2023-01-15,899.99
U000001,Electronics,2023-03-25,450.50
U000001,Groceries,2023-01-20,45.30
U000001,Groceries,2023-01-27,52.15
U000002,Electronics,2023-02-10,299.99
U000002,Electronics,2023-05-12,1200.00
U000003,Groceries,2023-03-01,38.99
```

**Generating from raw transactions:**

If you have raw transaction data, use the ETL script:

```python
from scripts.sample_data_etl import build_intervals

# Raw data format: UserId, OrderDate, Category, ...
interval_df = build_intervals(
    raw_df,
    min_purchase=2,        # Minimum purchases per user per category
    category_col="Category"
)
```

### Output Data Format

**Summary Report (`summary_all.json`):**

```json
{
  "Electronics": {
    "summary": {
      "n": 8500,
      "mean": 42.5,
      "median": 38.0,
      "std": 15.2,
      "skew": 0.85
    },
    "figures": {
      "hist_kde": "./reports/Electronics/interval_hist_kde.png",
      "violin": "./reports/Electronics/interval_violin.png"
    }
  },
  "Groceries": {
    "summary": {...},
    "figures": {...}
  }
}
```

**Detailed Report (`complete_report_all.json`):**

Includes additional fields:
- `discard_summary`: Rows removed during cleaning (negatives, missing, outliers)
- `transform_meta`: Transformation method applied and skewness metrics
- `unimodality_test`: Dip test statistic, p-value, and decision
- `peaks_table`: Detected peaks with position, height, width, prominence
- `modality_result`: GMM model selection (best k, BIC/AIC scores)
- `stable_peaks_table`: Bootstrap-validated peaks with support ratios

**Visualization Outputs:**

| Plot Type | Description |
|-----------|-------------|
| `interval_hist_kde.png` | Histogram with KDE overlay |
| `interval_violin.png` | Violin plot showing distribution shape |
| `interval_box.png` | Box plot with outlier indicators |
| `interval_cdf.png` | Cumulative distribution function |
| `interval_raincloud.png` | Combined violin + box + strip plot (requires `cat` column) |
| `*_peak_detection.png` | KDE with detected peaks marked |
| `*_stability_assessment_peaks.png` | Bootstrap support ratios per peak |

## Processing Modes

The pipeline automatically selects processing strategies based on data size:

| Mode | Row Count | Strategy |
|------|-----------|----------|
| **small** | < 10,000 | Full data processing, detailed KDE, 100 bootstrap iterations |
| **medium** | 10K - 1M | Sampling for visualization (configurable ratio), 80 bootstrap iterations |
| **large** | > 1M | DuckDB SQL for cleaning, histogram approximation, 50 bootstrap iterations |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_visualization.py
pytest tests/test_pipeline_integration.py

# Run with coverage
pytest tests/ --cov=repurchase_cycle --cov-report=html
```

## Acknowledgements

### Visualization

The raincloud plot visualization utilities in this project (`repurchase_cycle/modules/ptitprince.py`) are adapted from the open-source [PtitPrince](https://github.com/pog87/PtitPrince) project by Davide Poggiali.

The original implementation has been modified for compatibility with seaborn 0.13+ and to better fit this project's styling and usage requirements.

PtitPrince is licensed under the BSD 3-Clause License.

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request