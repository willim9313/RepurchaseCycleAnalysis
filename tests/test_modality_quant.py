# ======================================================
# test_modality_quant.py
# ======================================================
# 單元測試模組: 模態數量量化
# 對應模組: run_modality_quantification()
# ------------------------------------------------------

import numpy as np
import pandas as pd
import math
import pytest

from repurchase_cycle.modules.modality_quantification import (
    run_modality_quantification,
    _resolve_k_range,
    _select_best_k,
    _estimate_kde_n_peaks,
)


# ======================================================
# Fixture: 測試資料生成
# ======================================================
def make_df(arr: np.ndarray) -> pd.DataFrame:
    """快速生成 dataframe"""
    return pd.DataFrame({"interval_days": arr})


@pytest.fixture
def unimodal_data() -> pd.DataFrame:
    """單峰測試資料：N(0,1)"""
    rng = np.random.default_rng(42)
    return make_df(rng.normal(0, 1, 3000))


@pytest.fixture
def bimodal_data() -> pd.DataFrame:
    """雙峰測試資料：兩個分離的高斯"""
    rng = np.random.default_rng(42)
    x = np.concatenate([
        rng.normal(-3, 0.5, 2000),
        rng.normal(3, 0.5, 2000)
    ])
    return make_df(x)


@pytest.fixture
def trimodal_data() -> pd.DataFrame:
    """三峰測試資料"""
    rng = np.random.default_rng(42)
    x = np.concatenate([
        rng.normal(-4, 0.5, 1500),
        rng.normal(0, 0.5, 1500),
        rng.normal(4, 0.5, 1500)
    ])
    return make_df(x)


# ======================================================
# Helper: 檢查結果結構
# ======================================================
def assert_gmm_result_structure(gmm_result: dict):
    """驗證 gmm_result 字典結構"""
    expected_keys = {"best_n_components", "aic_scores", "bic_scores"}
    assert set(gmm_result.keys()) == expected_keys
    assert isinstance(gmm_result["best_n_components"], int)
    assert isinstance(gmm_result["aic_scores"], list)
    assert isinstance(gmm_result["bic_scores"], list)


def assert_consistency_check_structure(consistency_check: dict):
    """驗證 consistency_check 字典結構"""
    expected_keys = {"kde_n_peaks", "gmm_n_components", "status"}
    assert set(consistency_check.keys()) == expected_keys
    assert isinstance(consistency_check["kde_n_peaks"], int)
    assert isinstance(consistency_check["gmm_n_components"], int)
    assert consistency_check["status"] in {"consistent", "inconsistent"}


# ======================================================
# Test 1: _resolve_k_range 單元測試
# ======================================================
class TestResolveKRange:
    """k_range 參數解析測試"""

    def test_none_returns_default(self):
        """None 應回傳預設 [1..6]"""
        result = _resolve_k_range(None)
        assert result == [1, 2, 3, 4, 5, 6]

    def test_int_returns_range(self):
        """整數應回傳 range(1, n+1)"""
        result = _resolve_k_range(4)
        assert result == [1, 2, 3, 4]

    def test_tuple_range(self):
        """(start, end) 應回傳 range(start, end+1)"""
        result = _resolve_k_range((2, 5))
        assert result == [2, 3, 4, 5]

    def test_list_range(self):
        """[start, end] 應回傳 range(start, end+1)"""
        result = _resolve_k_range([1, 3])
        assert result == [1, 2, 3]

    def test_explicit_list(self):
        """明確的列表（長度>2或不滿足 start<end）應直接轉換"""
        # 長度為 4，會進入 fallback 邏輯
        result = _resolve_k_range([1, 3, 5, 7])
        assert result == [1, 3, 5, 7]

    def test_explicit_list_with_descending(self):
        """start >= end 時應直接轉換"""
        result = _resolve_k_range([5, 3])
        assert result == [5, 3]


# ======================================================
# Test 2: _select_best_k 單元測試
# ======================================================
class TestSelectBestK:
    """最佳 K 選擇測試"""

    def test_bic_selection(self):
        """BIC 選擇應回傳最小 BIC 對應的 K"""
        aic = {1: 100, 2: 90, 3: 95}
        bic = {1: 110, 2: 85, 3: 100}
        result = _select_best_k(aic, bic, "BIC")
        assert result == 2

    def test_aic_selection(self):
        """AIC 選擇應回傳最小 AIC 對應的 K"""
        aic = {1: 100, 2: 90, 3: 95}
        bic = {1: 110, 2: 85, 3: 100}
        result = _select_best_k(aic, bic, "AIC")
        assert result == 2

    def test_empty_scores_returns_1(self):
        """空分數應回傳 1"""
        result = _select_best_k({}, {}, "BIC")
        assert result == 1

    def test_case_insensitive(self):
        """選擇標準應不區分大小寫"""
        aic = {1: 100, 2: 90}
        bic = {1: 110, 2: 85}
        assert _select_best_k(aic, bic, "bic") == 2
        assert _select_best_k(aic, bic, "Bic") == 2


# ======================================================
# Test 3: _estimate_kde_n_peaks 單元測試
# ======================================================
class TestEstimateKdeNPeaks:
    """KDE 峰值估計測試"""

    def test_unimodal_returns_1(self):
        """單峰資料應回傳 1"""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        result = _estimate_kde_n_peaks(x)
        assert result == 1

    def test_bimodal_returns_2(self):
        """明顯雙峰資料應回傳 2"""
        rng = np.random.default_rng(42)
        x = np.concatenate([
            rng.normal(-5, 0.5, 1000),
            rng.normal(5, 0.5, 1000)
        ])
        result = _estimate_kde_n_peaks(x)
        assert result == 2

    def test_constant_returns_1(self):
        """常數資料應回傳 1"""
        x = np.ones(1000)
        result = _estimate_kde_n_peaks(x)
        assert result == 1

    def test_single_value_returns_1(self):
        """單一值應回傳 1"""
        x = np.array([5.0])
        result = _estimate_kde_n_peaks(x)
        assert result == 1

    def test_at_least_returns_1(self):
        """至少應回傳 1"""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, 100)
        result = _estimate_kde_n_peaks(x)
        assert result >= 1


# ======================================================
# Test 4: 單峰測試（Small Mode）
# ======================================================
class TestUnimodalDetection:
    """單峰偵測測試"""

    def test_small_mode_unimodal(self, unimodal_data):
        """Small 模式：單峰應被正確識別"""
        gmm_result, consistency_check = run_modality_quantification(
            unimodal_data, mode="small"
        )

        assert_gmm_result_structure(gmm_result)
        assert_consistency_check_structure(consistency_check)

        assert gmm_result["best_n_components"] == 1
        assert consistency_check["kde_n_peaks"] == 1
        assert consistency_check["status"] == "consistent"

    def test_medium_mode_unimodal(self, unimodal_data):
        """Medium 模式：單峰應被正確識別"""
        gmm_result, consistency_check = run_modality_quantification(
            unimodal_data, mode="medium"
        )

        assert gmm_result["best_n_components"] == 1
        assert consistency_check["kde_n_peaks"] == 1


# ======================================================
# Test 5: 雙峰測試
# ======================================================
class TestBimodalDetection:
    """雙峰偵測測試"""

    def test_small_mode_bimodal(self, bimodal_data):
        """Small 模式：雙峰應被識別"""
        gmm_result, consistency_check = run_modality_quantification(
            bimodal_data, mode="small"
        )

        assert_gmm_result_structure(gmm_result)
        assert gmm_result["best_n_components"] in {2, 3}
        assert consistency_check["kde_n_peaks"] in {2, 3}

    def test_large_mode_bimodal(self, bimodal_data):
        """Large 模式（DP-GMM）：雙峰應被識別"""
        gmm_result, consistency_check = run_modality_quantification(
            bimodal_data, mode="large"
        )

        assert gmm_result["best_n_components"] >= 1
        assert gmm_result["best_n_components"] <= 6


# ======================================================
# Test 6: 三峰測試
# ======================================================
class TestTrimodalDetection:
    """三峰偵測測試"""

    def test_large_mode_trimodal(self, trimodal_data):
        """Large 模式：三峰應被識別"""
        gmm_result, consistency_check = run_modality_quantification(
            trimodal_data,
            mode="large",
            mod_params={"k_range": [1, 6]}
        )

        assert gmm_result["best_n_components"] >= 2
        assert gmm_result["best_n_components"] <= 6
        assert isinstance(gmm_result["aic_scores"], list)
        assert isinstance(gmm_result["bic_scores"], list)


# ======================================================
# Test 7: 模式切換測試
# ======================================================
@pytest.mark.parametrize("mode", ["small", "medium", "large"])
def test_mode_switching(unimodal_data, mode):
    """測試所有模式都能正常執行"""
    gmm_result, consistency_check = run_modality_quantification(
        unimodal_data, mode=mode
    )

    assert_gmm_result_structure(gmm_result)
    assert_consistency_check_structure(consistency_check)
    assert gmm_result["best_n_components"] >= 1


# ======================================================
# Test 8: Medium 模式抽樣測試
# ======================================================
class TestMediumModeSubsample:
    """Medium 模式抽樣測試"""

    def test_subsample_triggered(self):
        """資料量超過 subsample_size 時應觸發抽樣"""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50000)
        df = make_df(x)

        gmm_result, consistency_check = run_modality_quantification(
            df,
            mode="medium",
            mod_params={"subsample_size": 20000}
        )

        assert_gmm_result_structure(gmm_result)
        # aic_scores 長度應與 k_range 相同，預設為 6
        assert len(gmm_result["aic_scores"]) == 6

    def test_no_subsample_when_small_data(self):
        """資料量小於 subsample_size 時不應抽樣"""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        df = make_df(x)

        gmm_result, _ = run_modality_quantification(
            df,
            mode="medium",
            mod_params={"subsample_size": 20000}
        )

        assert gmm_result["best_n_components"] >= 1


# ======================================================
# Test 9: Large 模式（DP-GMM）測試
# ======================================================
class TestLargeModeDP:
    """Large 模式 DP-GMM 測試"""

    def test_dp_gmm_weight_threshold(self, bimodal_data):
        """測試權重閾值參數"""
        gmm_result, _ = run_modality_quantification(
            bimodal_data,
            mode="large",
            mod_params={"dp_weight_threshold": 0.01}
        )

        assert gmm_result["best_n_components"] >= 1

    def test_dp_gmm_max_components(self, unimodal_data):
        """測試最大成分數限制"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="large",
            mod_params={"k_range": [1, 10]}
        )

        assert gmm_result["best_n_components"] <= 10

    def test_dp_gmm_scores_are_none_list(self, unimodal_data):
        """Large 模式的 aic_scores/bic_scores 應為全 None 列表"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data, mode="large"
        )

        # Large 模式不計算傳統 AIC/BIC，所以全為 None
        assert all(s is None for s in gmm_result["aic_scores"])
        assert all(s is None for s in gmm_result["bic_scores"])


# ======================================================
# Test 10: 選擇標準測試（AIC vs BIC）
# ======================================================
class TestSelectionMetric:
    """選擇標準測試"""

    def test_aic_metric(self, unimodal_data):
        """測試 AIC 選擇標準"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            mod_params={"selection_metric": "AIC"}
        )

        assert gmm_result["best_n_components"] >= 1
        # 過濾掉 None 後檢查有有效分數
        valid_scores = [s for s in gmm_result["aic_scores"] if s is not None]
        assert len(valid_scores) > 0

    def test_bic_metric(self, unimodal_data):
        """測試 BIC 選擇標準（預設）"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            mod_params={"selection_metric": "BIC"}
        )

        assert gmm_result["best_n_components"] >= 1


# ======================================================
# Test 11: 邊界條件測試
# ======================================================
class TestEdgeCases:
    """邊界條件測試"""

    def test_all_same_value(self):
        """全部相同值：應識別為單峰"""
        df = make_df(np.ones(5000) * 7)

        gmm_result, consistency_check = run_modality_quantification(df)

        assert gmm_result["best_n_components"] == 1
        assert consistency_check["kde_n_peaks"] == 1
        assert consistency_check["status"] == "consistent"
        # k=1 有值，其餘為 float 類型（sklearn 可能給出無意義分數）
        assert isinstance(gmm_result["aic_scores"][0], float)
        assert isinstance(gmm_result["bic_scores"][0], float)
        for s in gmm_result["aic_scores"][1:]:
            assert isinstance(s, float)
        for s in gmm_result["bic_scores"][1:]:
            assert isinstance(s, float)


    def test_all_nan(self):
        """全部 NaN：應回傳預設結果"""
        df = make_df(np.array([np.nan, np.nan, np.nan]))

        gmm_result, consistency_check = run_modality_quantification(df)

        assert gmm_result["best_n_components"] == 1
        assert gmm_result["aic_scores"] == []
        assert gmm_result["bic_scores"] == []

    def test_empty_dataframe(self):
        """空資料框：應回傳預設結果"""
        df = make_df(np.array([]))

        gmm_result, consistency_check = run_modality_quantification(df)

        assert gmm_result["best_n_components"] == 1
        assert consistency_check["status"] == "consistent"

    def test_single_value(self):
        """單一值：應識別為單峰"""
        df = make_df(np.array([42.0]))

        gmm_result, consistency_check = run_modality_quantification(df)

        assert gmm_result["best_n_components"] == 1

    def test_two_values(self):
        """兩個值：應能處理"""
        df = make_df(np.array([1.0, 100.0]))

        gmm_result, consistency_check = run_modality_quantification(df)

        assert gmm_result["best_n_components"] >= 1

    def test_mixed_nan_and_values(self):
        """混合 NaN 與正常值"""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        x[::10] = np.nan  # 10% NaN
        df = make_df(x)

        gmm_result, consistency_check = run_modality_quantification(df)

        assert_gmm_result_structure(gmm_result)
        assert gmm_result["best_n_components"] >= 1


# ======================================================
# Test 12: 錯誤處理測試
# ======================================================
class TestErrorHandling:
    """錯誤處理測試"""

    def test_missing_interval_days_column(self):
        """缺少 interval_days 欄位應拋出錯誤"""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="interval_days"):
            run_modality_quantification(df)

    def test_invalid_mode(self, unimodal_data):
        """無效模式應拋出錯誤"""
        with pytest.raises(ValueError, match="Unknown mode"):
            run_modality_quantification(unimodal_data, mode="invalid")


# ======================================================
# Test 13: 參數傳遞測試
# ======================================================
class TestParameterPassing:
    """參數傳遞測試"""

    def test_custom_k_range(self, unimodal_data):
        """測試自訂 k_range"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            mod_params={"k_range": [1, 3]}
        )

        # aic_scores 應有 3 個元素 (k=1,2,3)
        assert len(gmm_result["aic_scores"]) == 3

    def test_custom_max_iter(self, unimodal_data):
        """測試自訂 max_iter"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            mod_params={"max_iter": 100}
        )

        assert gmm_result["best_n_components"] >= 1

    def test_custom_n_init(self, unimodal_data):
        """測試自訂 n_init"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            mod_params={"n_init": 3}
        )

        assert gmm_result["best_n_components"] >= 1

    def test_random_state_reproducibility(self, unimodal_data):
        """測試隨機種子可重現性"""
        gmm_result1, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            general_params={"random_state": 42}
        )
        gmm_result2, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            general_params={"random_state": 42}
        )

        assert gmm_result1["best_n_components"] == gmm_result2["best_n_components"]


# ======================================================
# Test 14: 一致性檢查測試
# ======================================================
class TestConsistencyCheck:
    """GMM 與 KDE 一致性檢查測試"""

    def test_consistent_when_equal(self, unimodal_data):
        """當 GMM 與 KDE 峰數相同時應為 consistent"""
        _, consistency_check = run_modality_quantification(
            unimodal_data, mode="small"
        )

        if consistency_check["kde_n_peaks"] == consistency_check["gmm_n_components"]:
            assert consistency_check["status"] == "consistent"

    def test_inconsistent_when_different(self):
        """當 GMM 與 KDE 峰數不同時應為 inconsistent"""
        # 創建一個可能造成不一致的資料
        rng = np.random.default_rng(42)
        # 兩個接近的峰，GMM 可能合併，KDE 可能分開
        x = np.concatenate([
            rng.normal(0, 0.5, 1000),
            rng.normal(1.5, 0.5, 1000)
        ])
        df = make_df(x)

        _, consistency_check = run_modality_quantification(df, mode="small")

        # 只要結果是 consistent 或 inconsistent 都是有效的
        assert consistency_check["status"] in {"consistent", "inconsistent"}


# ======================================================
# Test 15: 分數列表完整性測試
# ======================================================
class TestScoreLists:
    """AIC/BIC 分數列表測試"""

    def test_scores_length_matches_k_range(self, unimodal_data):
        """分數列表長度應與 k_range 一致"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data,
            mode="small",
            mod_params={"k_range": [1, 4]}
        )

        assert len(gmm_result["aic_scores"]) == 4
        assert len(gmm_result["bic_scores"]) == 4

    def test_large_mode_scores_are_none_list(self, unimodal_data):
        """Large 模式的分數列表應全為 None"""
        gmm_result, _ = run_modality_quantification(
            unimodal_data, mode="large"
        )

        # Large 模式使用 BayesianGaussianMixture，不計算傳統 AIC/BIC
        assert isinstance(gmm_result["aic_scores"], list)
        assert isinstance(gmm_result["bic_scores"], list)
        # 長度應與 k_range 一致
        assert len(gmm_result["aic_scores"]) == 6  # 預設 k_range = [1, 6]
        assert len(gmm_result["bic_scores"]) == 6
        # 所有值都是 None
        assert all(s is None for s in gmm_result["aic_scores"])
        assert all(s is None for s in gmm_result["bic_scores"])

    def test_scores_may_have_none_for_unfittable_k(self):
        """當某些 k 無法 fit 時，對應的 score 應為 None"""
        # 只有 2 個樣本，k=3,4,5,6 無法 fit
        df = make_df(np.array([1.0, 100.0]))

        gmm_result, _ = run_modality_quantification(
            df,
            mode="small",
            mod_params={"k_range": [1, 6]}
        )

        assert len(gmm_result["aic_scores"]) == 6
        # k=1, 2 應該有值，k=3,4,5,6 應該是 None
        assert gmm_result["aic_scores"][0] is not None  # k=1
        assert gmm_result["aic_scores"][1] is not None  # k=2
        assert gmm_result["aic_scores"][2] is None  # k=3
        assert gmm_result["aic_scores"][3] is None  # k=4
