'''
Generate Sample Retail Transaction Data
======================================
本程式會生成多種量級的交易資料，對應不同的回購週期分佈特性：
- Small (< 10K): 單峰分佈 (Electronics)
- Medium (10K ~ 1M): 雙峰分佈 (Groceries)  
- Medium Uniform (10K ~ 1M): 無峰值/均勻分佈 (Stationery)
- Large (> 1M): 三峰分佈 (Supplements)

生成的資料將儲存為 CSV 檔案，供後續分析使用。
'''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# 設定參數
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
OUTPUT_DIR = 'data/raw'

# 根據 default_config.yml 的 data_size_thresholds 設定量級
# small: < 1e4 (10,000)
# medium: 1e4 ~ 1e6 (10,000 ~ 1,000,000)
# large: > 1e6 (1,000,000+)

DATA_SIZE_CONFIG = {
    'small': {
        'target_records': 8000,      # < 10K，留點緩衝
        'n_users': 150,
        'output_file': 'small_electronics_transactions.csv'
    },
    'medium': {
        'target_records': 100000,    # 10萬筆，屬於 medium 範圍
        'n_users': 2000,
        'output_file': 'medium_groceries_transactions.csv'
    },
    'medium_uniform': {
        'target_records': 80000,     # 8萬筆，屬於 medium 範圍
        'n_users': 1500,
        'output_file': 'medium_stationery_transactions.csv'
    },
    'large': {
        'target_records': 1200000,   # 120萬筆，> 1M
        'n_users': 15000,
        'output_file': 'large_supplements_transactions.csv'
    }
}

# 定義分類的回購週期特性
CATEGORY_CONFIGS = {
    'Electronics': {
        'items': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smartwatch'],
        'distribution': 'single_peak',
        'mean_interval': 90,        # 單峰：約90天週期
        'std_interval': 10,
        'description': '單峰分佈 - 電子產品回購週期約90天'
    },
    'Groceries': {
        'items': ['Milk', 'Bread', 'Eggs', 'Coffee', 'Rice'],
        'distribution': 'double_peak',
        'peaks': [7, 21],           # 雙峰：7天和21天
        'peak_weights': [0.6, 0.4], # 權重分配
        'std_interval': 2,
        'description': '雙峰分佈 - 日用品回購週期7天或21天'
    },
    'Supplements': {
        'items': ['Vitamin C', 'Protein Powder', 'Multivitamin', 'Fish Oil', 'Calcium'],
        'distribution': 'triple_peak',
        'peaks': [30, 60, 90],      # 三峰：30天、60天、90天
        'peak_weights': [0.5, 0.3, 0.2],
        'std_interval': 5,
        'description': '三峰分佈 - 保健品回購週期30/60/90天'
    },
    'Stationery': {
        'items': ['Notebook', 'Pen', 'Pencil', 'Eraser', 'Folder', 'Sticky Notes', 'Highlighter'],
        'distribution': 'uniform',
        'min_interval': 5,          # 最短間隔5天
        'max_interval': 60,         # 最長間隔60天
        'description': '均勻分佈 - 文具用品回購週期5~60天隨機'
    }
}


def generate_purchase_dates(cat_config, start, end):
    """
    為單一使用者生成購買日期序列，依據分類特性決定購買間隔
    
    :param cat_config: 分類配置
    :param start: 起始日期
    :param end: 結束日期
    :return: 購買日期列表
    """
    purchases = []
    current_date = start + timedelta(days=np.random.randint(0, 30))

    while current_date <= end:
        purchases.append(current_date)

        # 根據分類特性決定下次購買間隔
        if cat_config['distribution'] == 'single_peak':
            interval = np.random.normal(
                cat_config['mean_interval'],
                cat_config['std_interval']
            )
        elif cat_config['distribution'] == 'double_peak':
            # 雙峰：依權重隨機選擇其中一個峰值
            peak = np.random.choice(
                cat_config['peaks'], 
                p=cat_config['peak_weights']
            )
            interval = np.random.normal(peak, cat_config['std_interval'])
        elif cat_config['distribution'] == 'triple_peak':
            # 三峰：依權重隨機選擇其中一個峰值
            peak = np.random.choice(
                cat_config['peaks'], 
                p=cat_config['peak_weights']
            )
            interval = np.random.normal(peak, cat_config['std_interval'])
        elif cat_config['distribution'] == 'uniform':
            min_int = cat_config['min_interval']
            max_int = cat_config['max_interval']
            # 直接抽取整數，每個值機率相等
            interval = np.random.randint(min_int, max_int + 1)
            
        else:
            interval = 30

        current_date += timedelta(days=interval)

    return purchases


def generate_category_data(
    category_name: str,
    n_users: int,
    target_records: int,
    start_date: datetime,
    end_date: datetime,
    output_path: str
) -> pd.DataFrame:
    """
    為單一分類生成交易資料

    :param category_name: 分類名稱 (Electronics/Groceries/Supplements)
    :param n_users: 使用者數量
    :param target_records: 目標記錄數
    :param start_date: 起始日期
    :param end_date: 結束日期
    :param output_path: 輸出檔案路徑
    :return: DataFrame
    """
    cat_config = CATEGORY_CONFIGS[category_name]
    data = []

    print(f"\n生成 {category_name} 資料...")
    print(f"  配置: {cat_config['description']}")

    # 預估每位使用者的購買次數，調整使用者數以達到目標記錄數
    # 先用少量使用者測試平均購買次數
    test_purchases = []
    for _ in range(min(100, n_users)):
        dates = generate_purchase_dates(cat_config, start_date, end_date)
        test_purchases.append(len(dates))

    avg_purchases_per_user = np.mean(test_purchases)

    # 調整使用者數以接近目標記錄數
    adjusted_n_users = int(target_records / avg_purchases_per_user * 1.05)  # 多5%緩衝
    adjusted_n_users = max(adjusted_n_users, 50)  # 至少50位使用者

    print(f"  預估每用戶購買次數: {avg_purchases_per_user:.1f}")
    print(f"  調整後使用者數: {adjusted_n_users}")

    # 生成資料
    for user_id in range(1, adjusted_n_users + 1):
        purchase_dates = generate_purchase_dates(cat_config, start_date, end_date)

        for purchase_date in purchase_dates:
            item = np.random.choice(cat_config['items'])

            data.append({
                'UserId': f'U{user_id:06d}',
                'OrderNo': f'INV{len(data)+1:08d}',
                'OrderDate': purchase_date.strftime('%Y-%m-%d %H:%M:%S'),
                'ItemName': item,
                'Category': category_name,
                'Quantity': np.random.randint(1, 5),
                'UnitPrice': round(np.random.uniform(5, 100), 2),
                'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia'])
            })

        # 進度顯示
        if user_id % 1000 == 0:
            print(f"    已處理 {user_id} 位使用者，累計 {len(data)} 筆記錄...")

        # 如果已達到目標記錄數，提前結束
        if len(data) >= target_records:
            break

    # 建立 DataFrame 並排序
    df = pd.DataFrame(data)
    df = df.sort_values(['UserId', 'OrderDate']).reset_index(drop=True)

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 儲存為 CSV
    df.to_csv(output_path, index=False)

    return df


def generate_all_datasets(output_dir: str = OUTPUT_DIR):
    """
    生成所有量級的資料集

    :param output_dir: 輸出目錄
    """
    print("=" * 60)
    print("開始生成模擬零售交易資料")
    print("=" * 60)

    results = {}

    # Small: Electronics (單峰)
    small_config = DATA_SIZE_CONFIG['small']
    small_path = os.path.join(output_dir, small_config['output_file'])
    small_df = generate_category_data(
        category_name='Electronics',
        n_users=small_config['n_users'],
        target_records=small_config['target_records'],
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=small_path
    )
    results['small'] = {'df': small_df, 'path': small_path, 'category': 'Electronics'}

    # Medium: Groceries (雙峰)
    medium_config = DATA_SIZE_CONFIG['medium']
    medium_path = os.path.join(output_dir, medium_config['output_file'])
    medium_df = generate_category_data(
        category_name='Groceries',
        n_users=medium_config['n_users'],
        target_records=medium_config['target_records'],
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=medium_path
    )
    results['medium'] = {'df': medium_df, 'path': medium_path, 'category': 'Groceries'}
    
    # Medium Uniform: Stationery (無峰值/均勻分佈)
    medium_uniform_config = DATA_SIZE_CONFIG['medium_uniform']
    medium_uniform_path = os.path.join(output_dir, medium_uniform_config['output_file'])
    medium_uniform_df = generate_category_data(
        category_name='Stationery',
        n_users=medium_uniform_config['n_users'],
        target_records=medium_uniform_config['target_records'],
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=medium_uniform_path
    )
    results['medium_uniform'] = {'df': medium_uniform_df, 'path': medium_uniform_path, 'category': 'Stationery'}
    
    # Large: Supplements (三峰)
    large_config = DATA_SIZE_CONFIG['large']
    large_path = os.path.join(output_dir, large_config['output_file'])
    large_df = generate_category_data(
        category_name='Supplements',
        n_users=large_config['n_users'],
        target_records=large_config['target_records'],
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=large_path
    )
    results['large'] = {'df': large_df, 'path': large_path, 'category': 'Supplements'}

    # 印出摘要報告
    print("\n" + "=" * 60)
    print("[ 模擬零售交易資料生成完成! ]")
    print("=" * 60)

    for size_name, result in results.items():
        df = result['df']
        config = DATA_SIZE_CONFIG[size_name]
        cat_config = CATEGORY_CONFIGS[result['category']]

        print(f"\n【{size_name.upper()}】{result['category']}")
        print(f"  分佈類型: {cat_config['distribution']}")
        print(f"  目標記錄數: {config['target_records']:,}")
        print(f"  實際記錄數: {len(df):,}")
        print(f"  使用者數: {df['UserId'].nunique():,}")
        print(f"  日期範圍: {df['OrderDate'].min()} ~ {df['OrderDate'].max()}")
        print(f"  儲存位置: {result['path']}")

        # 驗證量級是否正確
        record_count = len(df)
        if size_name == 'small' and record_count < 1e4:
            print(f"  ✓ 符合 small 量級 (< 10,000)")
        elif size_name in ['medium', 'medium_uniform'] and 1e4 <= record_count < 1e6:
            print(f"  ✓ 符合 medium 量級 (10,000 ~ 1,000,000)")
        elif size_name == 'large' and record_count >= 1e6:
            print(f"  ✓ 符合 large 量級 (> 1,000,000)")
        else:
            print(f"  ⚠ 量級可能需要調整")

    print("\n" + "=" * 60)

    # 同時生成一個合併的資料集（可選）
    combined_path = os.path.join(output_dir, 'combined_transactions.csv')
    combined_df = pd.concat([r['df'] for r in results.values()], ignore_index=True)
    combined_df = combined_df.sort_values(['Category', 'UserId', 'OrderDate']).reset_index(drop=True)
    combined_df.to_csv(combined_path, index=False)
    print(f"\n合併資料集已儲存: {combined_path}")
    print(f"  總記錄數: {len(combined_df):,}")

    return results


def generate_single_dataset(
    size: str,
    output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """
    生成單一量級的資料集

    :param size: 量級名稱 ('small', 'medium', 'medium_uniform', 'large')
    :param output_dir: 輸出目錄
    :return: DataFrame
    """
    if size not in DATA_SIZE_CONFIG:
        raise ValueError(f"size 必須是 'small', 'medium', 'medium_uniform', 或 'large'，收到: {size}")

    size_to_category = {
        'small': 'Electronics',
        'medium': 'Groceries',
        'medium_uniform': 'Stationery',
        'large': 'Supplements'
    }

    config = DATA_SIZE_CONFIG[size]
    category = size_to_category[size]
    output_path = os.path.join(output_dir, config['output_file'])

    df = generate_category_data(
        category_name=category,
        n_users=config['n_users'],
        target_records=config['target_records'],
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=output_path
    )
    
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成模擬零售交易資料')
    parser.add_argument(
        '--size', 
        type=str, 
        choices=['small', 'medium', 'medium_uniform', 'large', 'all'],
        default='all',
        help='要生成的資料量級 (預設: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'輸出目錄 (預設: {OUTPUT_DIR})'
    )

    args = parser.parse_args()

    if args.size == 'all':
        generate_all_datasets(output_dir=args.output_dir)
    else:
        generate_single_dataset(size=args.size, output_dir=args.output_dir)
