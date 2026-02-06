# ======================================================
# test_interval_derivation.py
# ======================================================
# 單元測試模組: 區間計算與轉換
# 對應模組: run_interval_calculation()
# ------------------------------------------------------

import pytest
import pandas as pd
import numpy as np
from repurchase_cycle.modules.interval_derivation import (
    run_interval_calculation,
    _parse_dates,
    sanitize_context,
    DEFAULT_INTERVAL_PARAMS
)


# ======================================================
# Fixture: 測試資料生成
# ======================================================
@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """標準交易資料：多用戶多類別"""
    data = {
        "UserId": ["U1", "U1", "U1", "U1", "U2", "U2", "U2", "U3"],
        "Category": ["A", "A", "A", "B", "A", "A", "B", "A"],
        "OrderDate": pd.to_datetime([
            "2024-01-01", "2024-01-15", "2024-02-01",  # U1-A: 14天, 17天
            "2024-01-10",                              # U1-B: 僅一筆
            "2024-02-01", "2024-02-10",                # U2-A: 9天
            "2024-03-01",                              # U2-B: 僅一筆
            "2024-01-01"                               # U3-A: 僅一筆
        ])
    }
    return pd.DataFrame(data)


@pytest.fixture
def string_date_df() -> pd.DataFrame:
    """字串格式日期資料"""
    return pd.DataFrame({
        "UserId": ["U1", "U1", "U1"],
        "Category": ["A", "A", "A"],
        "OrderDate": ["2024/01/01", "2024/01/10", "2024/01/20"]
    })


@pytest.fixture
def custom_columns_df() -> pd.DataFrame:
    """自訂欄位名稱資料"""
    return pd.DataFrame({
        "customer_id": ["C1", "C1", "C1"],
        "product_type": ["X", "X", "X"],
        "purchase_date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        "amount": [100, 200, 150]
    })


@pytest.fixture
def extra_cols_df() -> pd.DataFrame:
    """包含額外欄位的資料"""
    return pd.DataFrame({
        "UserId": ["U1", "U1", "U1"],
        "Category": ["A", "A", "A"],
        "OrderDate": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        "Amount": [100, 200, 150],
        "Channel": ["online", "store", "online"]
    })


# ======================================================
# Helper: 檢查 conversion_summary 結構
# ======================================================
def assert_summary_structure(summary: dict):
    """驗證 summary 字典包含所有必要欄位且類型正確"""
    expected_keys = {
        "total_transactions",
        "unique_users",
        "unique_categories",
        "output_intervals",
        "single_purchase_dropped",
        "dropped_due_to_insufficient_intervals"
    }
    assert set(summary.keys()) == expected_keys, f"Keys mismatch: {summary.keys()}"
    for k in expected_keys:
        assert isinstance(summary[k], int), f"{k} should be int, got {type(summary[k])}"


# ======================================================
# Test 1: 基本行為檢查
# ======================================================
class TestBasicBehavior:
    """基本區間計算功能測試"""

    def test_returns_correct_types(self, sample_transactions):
        """確認回傳類型正確"""
        interval_df, summary = run_interval_calculation(sample_transactions)
        assert isinstance(interval_df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert_summary_structure(summary)

    def test_output_columns(self, sample_transactions):
        """確認輸出包含必要欄位"""
        interval_df, _ = run_interval_calculation(sample_transactions)
        expected_cols = {"uid", "cat", "order_date", "prev_order_date", "interval_days", "purchase_seq"}
        assert expected_cols.issubset(set(interval_df.columns))

    def test_interval_calculation_correct(self, sample_transactions):
        """確認區間計算正確"""
        interval_df, _ = run_interval_calculation(sample_transactions)
        
        # U1-A 群組應有 14天 和 17天 的區間
        u1a = interval_df[(interval_df["uid"] == "U1") & (interval_df["cat"] == "A")]
        intervals = u1a["interval_days"].dropna().tolist()
        
        assert 14.0 in intervals or pytest.approx(14.0) in intervals
        assert 17.0 in intervals or pytest.approx(17.0) in intervals

    def test_purchase_sequence_correct(self, sample_transactions):
        """確認購買序號正確"""
        params = {"keep_first_purchase": True}
        interval_df, _ = run_interval_calculation(sample_transactions, mod_params=params)
        
        # U1-A 群組應有序號 1, 2, 3
        u1a = interval_df[(interval_df["uid"] == "U1") & (interval_df["cat"] == "A")]
        seqs = sorted(u1a["purchase_seq"].tolist())
        assert seqs == [1, 2, 3]

    def test_summary_counts_correct(self, sample_transactions):
        """確認摘要統計正確"""
        _, summary = run_interval_calculation(sample_transactions)
        
        assert summary["total_transactions"] == 8
        assert summary["unique_users"] == 3
        assert summary["unique_categories"] == 2


# ======================================================
# Test 2: 首次購買處理
# ======================================================
class TestFirstPurchaseHandling:
    """首次購買處理測試"""

    def test_drop_first_purchase_default(self, sample_transactions):
        """預設移除首次購買"""
        interval_df, summary = run_interval_calculation(sample_transactions)
        
        # 首次購買的 prev_order_date 應為 NaN，預設應被移除
        assert interval_df["prev_order_date"].isna().sum() == 0
        assert summary["single_purchase_dropped"] > 0

    def test_keep_first_purchase(self, sample_transactions):
        """測試保留首次購買"""
        params = {"keep_first_purchase": True}
        interval_df, summary = run_interval_calculation(sample_transactions, mod_params=params)
        
        # 首次購買應被保留，interval_days 為 NaN
        assert interval_df["interval_days"].isna().sum() > 0
        assert summary["single_purchase_dropped"] == 0


# ======================================================
# Test 3: 最小區間數過濾
# ======================================================
class TestMinIntervalsFilter:
    """最小區間數過濾測試"""

    def test_min_intervals_default(self, sample_transactions):
        """預設最小區間數為 2"""
        interval_df, summary = run_interval_calculation(sample_transactions)
        
        # 只有一筆購買的群組應被過濾
        # U1-B, U2-B, U3-A 各只有一筆，應被過濾
        assert summary["dropped_due_to_insufficient_intervals"] >= 0

    def test_min_intervals_custom(self, sample_transactions):
        """自訂最小區間數"""
        params = {"min_intervals_per_group": 3}
        interval_df, summary = run_interval_calculation(sample_transactions, mod_params=params)
        
        # U1-A 有 3 筆交易（2 個區間），應被過濾
        # 只有符合條件的群組保留
        for uid_cat, group in interval_df.groupby(["uid", "cat"]):
            assert len(group) >= 3

    def test_min_intervals_one(self, sample_transactions):
        """最小區間數為 1"""
        params = {"min_intervals_per_group": 1, "keep_first_purchase": True}
        interval_df, summary = run_interval_calculation(sample_transactions, mod_params=params)
        
        # 應保留更多資料
        assert len(interval_df) > 0


# ======================================================
# Test 4: 日期解析測試
# ======================================================
class TestDateParsing:
    """日期解析測試"""

    def test_string_date_auto_parse(self, string_date_df):
        """自動解析字串日期"""
        interval_df, _ = run_interval_calculation(string_date_df)
        
        assert pd.api.types.is_datetime64_any_dtype(interval_df["order_date"])

    def test_custom_date_format(self):
        """自訂日期格式"""
        df = pd.DataFrame({
            "UserId": ["U1", "U1", "U1"],
            "Category": ["A", "A", "A"],
            "OrderDate": ["01-01-2024", "01-15-2024", "02-01-2024"]
        })
        params = {"date_format": "%m-%d-%Y"}
        interval_df, _ = run_interval_calculation(df, mod_params=params)
        
        assert pd.api.types.is_datetime64_any_dtype(interval_df["order_date"])
        # 確認區間計算正確（14天）
        assert 14.0 in interval_df["interval_days"].tolist()

    def test_already_datetime(self, sample_transactions):
        """已經是 datetime 類型"""
        interval_df, _ = run_interval_calculation(sample_transactions)
        assert pd.api.types.is_datetime64_any_dtype(interval_df["order_date"])


# ======================================================
# Test 5: 自訂欄位名稱
# ======================================================
class TestCustomColumns:
    """自訂欄位名稱測試"""

    def test_custom_column_names(self, custom_columns_df):
        """測試自訂欄位名稱"""
        params = {
            "uid_col": "customer_id",
            "cat_col": "product_type",
            "date_col": "purchase_date",
            "groupby_cols": ["customer_id", "product_type"]
        }
        interval_df, summary = run_interval_calculation(custom_columns_df, mod_params=params)
        
        assert_summary_structure(summary)
        # 輸出應使用標準化欄位名
        assert "uid" in interval_df.columns
        assert "cat" in interval_df.columns

    def test_extra_cols_preserved(self, extra_cols_df):
        """測試額外欄位保留"""
        params = {"extra_cols": ["Amount", "Channel"]}
        interval_df, _ = run_interval_calculation(extra_cols_df, mod_params=params)
        
        assert "Amount" in interval_df.columns
        assert "Channel" in interval_df.columns


# ======================================================
# Test 6: 模式切換測試
# ======================================================
@pytest.mark.parametrize("mode", ["small", "medium", "large"])
def test_mode_switching(sample_transactions, mode):
    """測試不同執行模式"""
    interval_df, summary = run_interval_calculation(sample_transactions, mode=mode)
    assert_summary_structure(summary)
    assert len(interval_df) > 0
    assert "interval_days" in interval_df.columns


class TestLargeMode:
    """Large 模式（DuckDB）專項測試"""

    def test_large_mode_basic(self, sample_transactions):
        """Large 模式基本測試"""
        interval_df, summary = run_interval_calculation(sample_transactions, mode="large")
        assert_summary_structure(summary)
        assert len(interval_df) > 0

    def test_large_mode_consistency_with_small(self, sample_transactions):
        """Large 模式與 Small 模式結果應一致"""
        small_df, small_summary = run_interval_calculation(sample_transactions, mode="small")
        large_df, large_summary = run_interval_calculation(sample_transactions, mode="large")
        
        # 輸出數量應相同
        assert len(small_df) == len(large_df)
        assert small_summary["output_intervals"] == large_summary["output_intervals"]
        
        # 區間值應相同（排序後比較）
        small_intervals = sorted(small_df["interval_days"].dropna().tolist())
        large_intervals = sorted(large_df["interval_days"].dropna().tolist())
        
        for s, l in zip(small_intervals, large_intervals):
            assert s == pytest.approx(l)

    def test_large_mode_with_custom_params(self, custom_columns_df):
        """Large 模式自訂參數"""
        params = {
            "uid_col": "customer_id",
            "cat_col": "product_type",
            "date_col": "purchase_date",
            "groupby_cols": ["customer_id", "product_type"]
        }
        interval_df, summary = run_interval_calculation(
            custom_columns_df, mode="large", mod_params=params
        )
        assert_summary_structure(summary)


# ======================================================
# Test 7: 錯誤輸入檢查
# ======================================================
class TestErrorHandling:
    """錯誤處理測試"""

    def test_missing_required_columns(self):
        """測試缺少必要欄位"""
        df = pd.DataFrame({
            "UserId": ["U1", "U2"],
            "OrderDate": pd.to_datetime(["2024-01-01", "2024-01-02"])
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            run_interval_calculation(df)

    def test_invalid_mode(self, sample_transactions):
        """測試無效的執行模式"""
        with pytest.raises(ValueError, match="Invalid mode"):
            run_interval_calculation(sample_transactions, mode="invalid_mode")

    def test_missing_custom_columns(self, sample_transactions):
        """測試指定不存在的欄位"""
        params = {"uid_col": "nonexistent_col"}
        with pytest.raises(ValueError, match="Missing required columns"):
            run_interval_calculation(sample_transactions, mod_params=params)


# ======================================================
# Test 8: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_empty_dataframe(self):
        """測試空資料框"""
        df = pd.DataFrame({
            "UserId": [],
            "Category": [],
            "OrderDate": pd.to_datetime([])
        })
        interval_df, summary = run_interval_calculation(df)
        
        assert len(interval_df) == 0
        assert summary["total_transactions"] == 0

    def test_single_transaction(self):
        """測試僅一筆交易"""
        df = pd.DataFrame({
            "UserId": ["U1"],
            "Category": ["A"],
            "OrderDate": pd.to_datetime(["2024-01-01"])
        })
        interval_df, summary = run_interval_calculation(df)
        
        # 僅一筆無法計算區間
        assert len(interval_df) == 0
        assert summary["single_purchase_dropped"] == 1

    def test_same_day_purchases(self):
        """測試同一天多次購買"""
        df = pd.DataFrame({
            "UserId": ["U1", "U1", "U1"],
            "Category": ["A", "A", "A"],
            "OrderDate": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"])
        })
        interval_df, summary = run_interval_calculation(df)
        
        # 同一天的區間應為 0
        assert (interval_df["interval_days"] == 0).all()

    def test_single_user_multiple_categories(self):
        """測試單一用戶多類別"""
        df = pd.DataFrame({
            "UserId": ["U1"] * 6,
            "Category": ["A", "A", "A", "B", "B", "B"],
            "OrderDate": pd.to_datetime([
                "2024-01-01", "2024-01-10", "2024-01-20",
                "2024-02-01", "2024-02-15", "2024-03-01"
            ])
        })
        interval_df, summary = run_interval_calculation(df)
        
        # 應有兩個群組
        assert interval_df["cat"].nunique() == 2
        assert summary["unique_categories"] == 2

    def test_very_long_interval(self):
        """測試超長區間"""
        df = pd.DataFrame({
            "UserId": ["U1", "U1", "U1"],
            "Category": ["A", "A", "A"],
            "OrderDate": pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"])
        })
        interval_df, _ = run_interval_calculation(df)
        
        # 應正確計算約 730 天的區間
        intervals = interval_df["interval_days"].tolist()
        assert any(i > 700 for i in intervals)


# ======================================================
# Test 9: _parse_dates 單元測試
# ======================================================
class TestParseDates:
    """日期解析函數單元測試"""

    def test_parse_string_dates(self):
        """測試字串日期解析"""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"]
        })
        result = _parse_dates(df, "date")
        
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_parse_with_format(self):
        """測試指定格式解析"""
        df = pd.DataFrame({
            "date": ["01/01/2024", "01/02/2024", "01/03/2024"]
        })
        result = _parse_dates(df, "date", date_format="%m/%d/%Y")
        
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["date"].iloc[0].year == 2024

    def test_already_datetime_no_change(self):
        """測試已是 datetime 不做改變"""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"])
        })
        result = _parse_dates(df, "date")
        
        assert pd.api.types.is_datetime64_any_dtype(result["date"])


# ======================================================
# Test 10: sanitize_context 單元測試
# ======================================================
class TestSanitizeContext:
    """上下文清理函數單元測試"""

    def test_remove_illegal_characters(self):
        """測試移除非法字元"""
        assert sanitize_context("file/name") == "file_name"
        assert sanitize_context("file\\name") == "file_name"
        assert sanitize_context("file:name") == "file_name"
        assert sanitize_context('file*name') == "file_name"
        assert sanitize_context('file?name') == "file_name"
        assert sanitize_context('file"name') == "file_name"
        assert sanitize_context("file<name>") == "file_name"
        assert sanitize_context("file|name") == "file_name"

    def test_strip_whitespace(self):
        """測試去除首尾空白"""
        assert sanitize_context("  name  ") == "name"
        assert sanitize_context("\tname\n") == "name"

    def test_merge_consecutive_underscores(self):
        """測試合併連續底線"""
        assert sanitize_context("file__name") == "file_name"
        assert sanitize_context("file___name") == "file_name"
        assert sanitize_context("file  name") == "file_name"

    def test_empty_string(self):
        """測試空字串"""
        assert sanitize_context("") == ""
        assert sanitize_context(None) is None

    def test_normal_string(self):
        """測試正常字串不變"""
        assert sanitize_context("normal_name") == "normal_name"
        assert sanitize_context("Category1") == "Category1"


# ======================================================
# Test 11: 複雜情境測試
# ======================================================
class TestComplexScenarios:
    """複雜情境測試"""

    def test_multiple_users_categories_mixed(self):
        """多用戶多類別混合測試"""
        np.random.seed(42)
        users = [f"U{i}" for i in range(5)]
        categories = ["A", "B", "C"]
        
        records = []
        for user in users:
            for cat in categories:
                n_purchases = np.random.randint(2, 6)
                dates = pd.date_range("2024-01-01", periods=n_purchases, freq="10D")
                for d in dates:
                    records.append({"UserId": user, "Category": cat, "OrderDate": d})
        
        df = pd.DataFrame(records)
        interval_df, summary = run_interval_calculation(df)
        
        assert_summary_structure(summary)
        assert summary["unique_users"] == 5
        assert summary["unique_categories"] == 3
        assert len(interval_df) > 0

    def test_unsorted_input_data(self):
        """測試未排序輸入資料"""
        df = pd.DataFrame({
            "UserId": ["U1", "U1", "U1"],
            "Category": ["A", "A", "A"],
            "OrderDate": pd.to_datetime(["2024-02-01", "2024-01-01", "2024-01-15"])
        })
        interval_df, _ = run_interval_calculation(df)
        
        # 應正確排序並計算區間
        intervals = sorted(interval_df["interval_days"].tolist())
        assert intervals == [14.0, 17.0]

    def test_default_params_unchanged(self, sample_transactions):
        """確認 DEFAULT_INTERVAL_PARAMS 不被修改"""
        original_params = DEFAULT_INTERVAL_PARAMS.copy()
        _ = run_interval_calculation(sample_transactions, mod_params={"keep_first_purchase": True})
        
        assert DEFAULT_INTERVAL_PARAMS == original_params
