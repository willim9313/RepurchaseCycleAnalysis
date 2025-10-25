'''
生成模擬零售交易資料
本程式會生成包含多種商品分類的交易資料, 並模擬不同分類的回購週期特性.
生成的資料將儲存為 CSV 檔案, 供後續分析使用.
'''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# 設定參數
N_USERS = 200
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)
OUTPUT_PATH = 'data/raw/sample_retail_transactions.csv'


# 定義三種分類的回購週期特性
categories = {
    'Electronics': {
        'items': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smartwatch'],
        'distribution': 'single_peak',  # 單峰：約180天週期
        'mean_interval': 180,
        'std_interval': 5
    },
    'Groceries': {
        'items': ['Milk', 'Bread', 'Eggs', 'Coffee', 'Rice'],
        'distribution': 'double_peak',  # 雙峰：7天和14天
        'peaks': [7, 14],
        'std_interval': 2
    },
    'Supplements': {
        'items': ['Vitamin C', 'Protein Powder', 'Multivitamin', 'Fish Oil', 'Calcium'],
        'distribution': 'triple_peak',  # 三峰：30天、60天、90天
        'peaks': [30, 60, 90],
        'std_interval': 5
    }
}


def generate_purchase_dates(user_id, cat_config, start, end):
    """
    為單一使用者生成購買日期序列, 依據分類特性決定購買間隔
    :param user_id: 使用者ID
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
            # 雙峰：隨機選擇其中一個峰值
            peak = np.random.choice(cat_config['peaks'])
            interval = np.random.normal(peak, cat_config['std_interval'])
        elif cat_config['distribution'] == 'triple_peak':
            # 三峰：隨機選擇其中一個峰值，權重不同
            peak = np.random.choice(cat_config['peaks'], p=[0.5, 0.3, 0.2])
            interval = np.random.normal(peak, cat_config['std_interval'])

        # 確保間隔至少為1天
        interval = max(1, int(interval))
        current_date += timedelta(days=interval)

    return purchases


def generate_sample_data(n_users, start_date, end_date, output_path):
    """
    生成模擬零售交易資料
    :param n_users: 使用者數量
    :param start_date: 起始日期
    :param end_date: 結束日期
    :param output_path: 輸出檔案路徑
    :return: None
    生成的資料將儲存為 CSV 檔案
    """
    # 生成資料
    data = []

    for user_id in range(1, n_users + 1):
        # 每個使用者隨機選擇1-3個分類購買
        n_categories = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        user_categories = np.random.choice(
            list(categories.keys()),
            size=n_categories,
            replace=False
        )

        for cat_name in user_categories:
            cat_config = categories[cat_name]

            # 生成該使用者在此分類的購買日期
            purchase_dates = generate_purchase_dates(
                user_id,
                cat_config,
                start_date,
                end_date
            )

            # 為每次購買選擇商品
            for purchase_date in purchase_dates:
                item = np.random.choice(cat_config['items'])

                data.append({
                    'UserId': f'U{user_id:04d}',
                    'OrderNo': f'INV{len(data)+1:06d}',
                    'OrderDate': purchase_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'ItemName': item,
                    'Category': cat_name,
                    'Quantity': np.random.randint(1, 5),
                    'UnitPrice': np.random.uniform(5, 100),
                    'Country': np.random.choice(['USA', 'UK', 'Canada', 'Australia'])
                })

    # 建立 DataFrame 並排序
    df = pd.DataFrame(data)
    df = df.sort_values(['UserId', 'OrderDate']).reset_index(drop=True)

    # 儲存為 CSV
    output_path = 'data/raw/sample_retail_transactions.csv'
    df.to_csv(output_path, index=False)
    
    print("="*50)
    print("[ 模擬零售交易資料生成完成! ]")
    print(f" 已生成 {len(df)} 筆交易資料")
    print(f" 儲存位置: {output_path}")
    print("\n 資料摘要:")
    print(f"  - 使用者數: {df['UserId'].nunique()}")
    print(f"  - 分類數: {df['Category'].nunique()}")
    print(f"  - 日期範圍: {df['OrderDate'].min()} ~ {df['OrderDate'].max()}")
    print("\n各分類交易數:")
    print(df['Category'].value_counts())
    print("="*50)


if __name__ == "__main__":
    generate_sample_data(
        n_users=N_USERS,
        start_date=START_DATE,
        end_date=END_DATE,
        output_path=OUTPUT_PATH
    )
