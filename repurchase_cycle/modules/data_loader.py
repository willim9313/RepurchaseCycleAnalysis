# repurchase_cycle/modules/data_loader.py
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_data(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """
    根據檔案副檔名自動選擇適當的pandas讀取方法，支援csv, parquet, excel, json等格式。
    其他pandas讀取參數可透過kwargs傳入。
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in ['.csv']:
        return pd.read_csv(file_path, **kwargs)
    elif suffix in ['.parquet']:
        return pd.read_parquet(file_path, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, **kwargs)
    elif suffix in ['.json']:
        return pd.read_json(file_path, **kwargs)
    elif suffix in ['.feather']:
        return pd.read_feather(file_path, **kwargs)
    elif suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
