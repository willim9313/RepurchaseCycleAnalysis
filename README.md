# Repurchase Cycle Analysis

A Python-based tool for analyzing customer repurchase cycles across different product categories. This project helps identify purchasing patterns, calculate interval statistics, and visualize customer behavior.

## Features

- **ETL Pipeline**: Process raw transaction data into interval-based datasets
- **Statistical Analysis**: Calculate repurchase cycle statistics including KDE peaks, mean/median intervals, and distribution tests
- **Visualization**: Generate comprehensive plots for single and multi-category analysis
- **Category Comparison**: Compare repurchase patterns across different product categories

## Project Structure

```
RepurchaseCycleAnalysis/
├── data/
│   ├── raw/                    # Raw transaction data
│   └── processed/              # Processed interval data and reports
├── src/
│   ├── etl.py                 # Data transformation functions
│   ├── stats.py               # Statistical analysis functions
│   └── viz.py                 # Visualization functions
├── tests/
│   ├── test_etl.py            # ETL unit tests
│   ├── test_stats.py          # Statistics unit tests
│   └── test_viz.py            # Visualization unit tests
├── notebooks/
│   └── visualization.ipynb    # Interactive analysis notebook
├── main.py                    # Main analysis pipeline
└── generate_sample_data.py    # Sample data generator
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RepurchaseCycleAnalysis
```

2. Install required dependencies:
```bash
pip install pandas numpy scipy matplotlib seaborn pytest
```

## Usage

### 1. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates sample retail transaction data in `data/raw/sample_retail_transactions.csv`.

### 2. Run Analysis Pipeline

```bash
python main.py
```

This will:
- Load raw transaction data
- Build interval datasets
- Perform statistical analysis
- Generate visualizations
- Save results to `data/processed/`

### 3. Using Individual Components

```python
from src.etl import build_intervals
from src.stats import analyze_distribution
from src.viz import plot_multi_category_kde, plot_summary_comparison

# Build intervals from raw data
interval_df = build_intervals(
    raw_df,
    min_purchase=2,
    category_col="Category"
)

# Analyze distributions
report = analyze_distribution(
    interval_df,
    interval_col="interval_days",
    group_cols=("Category",),
    kde_grid_size=2048
)

# Visualize results
plot_multi_category_kde(
    interval_df,
    interval_col="interval_days",
    category_col="Category",
    show=True
)
```

## Key Functions

### ETL (`src/etl.py`)

- `build_intervals()`: Transform raw transaction data into interval-based format

### Statistics (`src/stats.py`)

- `analyze_distribution()`: Perform comprehensive distribution analysis including KDE, normality tests, and peak detection

### Visualization (`src/viz.py`)

- `plot_category_distribution()`: Plot distribution for a single category
- `plot_multi_category_kde()`: Compare KDE distributions across categories
- `plot_summary_comparison()`: Compare statistical summaries across categories

## Output Files

- `data/processed/interval_data.csv`: Processed interval dataset
- `data/processed/interval_analysis_report.csv`: Statistical analysis report with metrics for each category

## Testing

Run the test suite:

```bash
pytest tests/
```

Individual test files:
```bash
pytest tests/test_etl.py
pytest tests/test_stats.py
pytest tests/test_viz.py
```

## Example Analysis

The analysis typically reveals:

1. **Repurchase Intervals**: Time between consecutive purchases for each category
2. **KDE Peaks**: Common repurchase cycle patterns
3. **Distribution Statistics**: Mean, median, standard deviation, and quartiles
4. **Normality Tests**: Shapiro-Wilk test results for distribution assessment

## Requirements

- Python 3.11+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pytest (for testing)

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.