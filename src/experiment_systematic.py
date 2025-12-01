# experiment_systematic.py
"""
系統抽樣實驗程式
===============

本程式實作並比較兩種抽樣方法在頻繁項目集挖掘中的效果：
1. 標準抽樣（Standard Sampling）：有放回隨機抽樣
2. 系統抽樣（Systematic Sampling）：固定間隔取樣

系統抽樣原理：
- 將機率分布正規化到 [0,1] 區間（累積機率分布）
- 機率越大的交易，在累積機率軸上佔據的區間越大
- 用固定間隔（1/k）取樣，隨機選擇起點
- 類似「不放回」的效果，降低估計方差

輸出：
- 終端機顯示實驗進度和數值結果
- 圖表儲存至 systematic_results_{MIN_SUP}/ 資料夾
"""

import math
import os
from typing import List, FrozenSet, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sampling import get_sampling_probs
from miner import _eclat_mine, brute_force_frequent_itemsets
from metrics import compute_non_common_output_ratio, compute_support_error_rate

# ==========================================
# 類型定義
# ==========================================
Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]

# ==========================================
# 全域參數（可調整）
# ==========================================
MAX_LEN = None          # itemset 最大長度（None = 無限制）
MAX_TRANSACTIONS = 1500000  # 限制交易數量（None = 使用全部）
MIN_SUP = 0.005         # 最小支援度

# ==========================================
# 圖表設定
# ==========================================
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Microsoft JhengHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 系統抽樣實作
# ==========================================

def systematic_sampling(
    transactions: List[Transaction],
    probs: List[float],
    k: int,
    random_seed: int = None,
) -> Tuple[List[Transaction], List[float]]:
    """
    系統抽樣（Systematic Sampling）
    
    演算法步驟：
        1. 計算累積機率分布 CDF
        2. 將 [0,1] 區間等分成 k 份，間隔 = 1/k
        3. 在 [0, 1/k) 隨機選擇起點
        4. 以固定間隔取樣，取樣點落在哪個交易的累積機率區間，就選中該交易
    
    優勢：
        - 樣本分布更均勻，覆蓋整個機率空間
        - 降低估計方差（相比標準有放回抽樣）
        - 類似「不放回」的效果
    
    參數：
        transactions: 完整資料集
        probs: 每筆交易的抽樣機率（需已正規化，總和 = 1）
        k: 樣本數
        random_seed: 隨機種子（用於可重現性）
    
    回傳：
        sample_txns: 抽樣後的交易列表
        weights: 每筆樣本的權重（用於 Horvitz-Thompson 估計）
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(transactions)
    k = min(k, n)
    
    # Step 1: 計算累積機率分布
    cum_probs = np.cumsum(probs)
    
    # Step 2: 計算固定間隔
    step = 1.0 / k
    
    # Step 3: 隨機選擇起點（在第一個區間內）
    start_point = np.random.uniform(0, step)
    
    # Step 4: 生成所有取樣點
    points = np.arange(start_point, 1.0, step)[:k]  # 確保恰好 k 個點
    
    # Step 5: 找到每個取樣點對應的交易索引
    indices = np.searchsorted(cum_probs, points)
    indices = np.clip(indices, 0, n - 1)
    
    # Step 6: 構造樣本和權重
    sample_txns = []
    weights = []
    
    for idx in indices:
        T = transactions[idx]
        pT = probs[idx]
        if pT <= 1e-9:  # 避免除以 0
            continue
        sample_txns.append(T)
        # Horvitz-Thompson 權重 = 1 / (p(T) * k)
        weights.append(1.0 / (pT * k))
    
    return sample_txns, weights


def standard_sampling(
    transactions: List[Transaction],
    probs: List[float],
    k: int,
    random_seed: int = None,
) -> Tuple[List[Transaction], List[float]]:
    """
    標準抽樣（Standard Sampling with Replacement）
    
    使用 NumPy 的 random.choice 進行有放回隨機抽樣。
    這是論文中的基準方法（Baseline）。
    
    特性：
        - 每次抽樣獨立
        - 同一筆交易可能被重複抽到
        - 估計方差較高（但估計無偏）
    
    參數：
        transactions: 完整資料集
        probs: 每筆交易的抽樣機率（需已正規化）
        k: 樣本數
        random_seed: 隨機種子
    
    回傳：
        sample_txns: 抽樣後的交易列表
        weights: 每筆樣本的權重
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(transactions)
    
    # 有放回隨機抽樣
    indices = np.random.choice(n, size=k, p=probs, replace=True)
    
    # 構造樣本和權重
    sample_txns = []
    weights = []
    
    for idx in indices:
        T = transactions[idx]
        pT = probs[idx]
        if pT <= 1e-9:
            continue
        sample_txns.append(T)
        weights.append(1.0 / (pT * k))
    
    return sample_txns, weights


# ==========================================
# 挖礦函數（使用不同抽樣方法）
# ==========================================

def approximate_itemset_miner_with_sampling_method(
    transactions: List[Transaction],
    min_sup_abs: int,
    k: int,
    sampling: str = "uniform",
    use_systematic: bool = False,
    max_len: int = None,
    random_seed: int = None,
):
    """
    支援系統抽樣的 approximate mining
    
    參數：
        transactions: 完整資料集
        min_sup_abs: 最小支援度（絕對值）
        k: 樣本數
        sampling: 抽樣方法 (uniform/nonuni1/nonuni2)
        use_systematic: 是否使用系統抽樣
        max_len: itemset 最大長度
        random_seed: 隨機種子
    
    回傳：
        approx_frequent: 估計的 frequent itemsets
        estimated_support: 所有估計的 support
    """
    n = len(transactions)
    if n == 0 or k <= 0:
        return {}, {}
    
    if max_len is None:
        max_len = max(len(t) for t in transactions)
    
    # 1. 計算抽樣機率
    probs = get_sampling_probs(
        transactions=transactions,
        sampling=sampling,
        min_sup_abs=min_sup_abs,
    )
    
    # 2. 根據方法進行抽樣
    if use_systematic:
        sample_txns, weights = systematic_sampling(
            transactions, probs, k, random_seed
        )
    else:
        sample_txns, weights = standard_sampling(
            transactions, probs, k, random_seed
        )
    
    if not sample_txns:
        return {}, {}
    
    # 3. 在加權樣本上跑 Eclat
    est_support = _eclat_mine(
        transactions=sample_txns,
        weights=weights,
        min_sup=min_sup_abs,
        max_len=max_len,
    )
    
    # 4. 篩選結果
    approx_frequent = dict(est_support)
    return approx_frequent, est_support


# ==========================================
# 讀取資料
# ==========================================

def load_transactions(path: str, sep: str = " ") -> List[Transaction]:
    """讀取交易資料"""
    txns: List[Transaction] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(sep)
            txns.append(frozenset(items))
    return txns


# ==========================================
# 實驗流程
# ==========================================

def _run_single_method(
    method_name: str,
    transactions: List[Transaction],
    min_sup_abs: int,
    sample_size: int,
    sampling_type: str,
    use_systematic: bool,
    exact_freq: Dict,
    max_len: int,
    random_seed: int,
) -> Tuple[float, float]:
    """
    執行單一方法的實驗（輔助函數）
    
    參數：
        method_name: 方法名稱（用於顯示）
        transactions: 交易資料
        min_sup_abs: 絕對支援度門檻
        sample_size: 樣本數
        sampling_type: 抽樣機率類型 ("nonuni1" 或 "nonuni2")
        use_systematic: 是否使用系統抽樣
        exact_freq: Ground truth frequent itemsets
        max_len: itemset 最大長度
        random_seed: 隨機種子
    
    回傳：
        (non_common_ratio, support_error_rate)
    """
    approx_freq, est_sup = approximate_itemset_miner_with_sampling_method(
        transactions, min_sup_abs, sample_size,
        sampling=sampling_type,
        use_systematic=use_systematic,
        max_len=max_len,
        random_seed=random_seed
    )
    
    nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
    se_rate = compute_support_error_rate(exact_freq, est_sup)
    
    print(f"  - {method_name}: ratio={nc_ratio:.4f}, error={se_rate:.4f}")
    
    return nc_ratio, se_rate


def run_experiment_with_comparison(
    dataset_name: str,
    transactions: List[Transaction],
    min_sup_ratio: float,
    sample_rates: List[float],
    max_len: int,
    random_seed: int = 42,
) -> Dict:
    """
    對單一資料集進行完整實驗
    
    實驗設計：
        - 比較 4 種方法：nonuni1_std, nonuni1_sys, nonuni2_std, nonuni2_sys
        - 計算 2 個指標：Non-common Output Ratio 和 Support Error Rate
        - 測試多個抽樣比例
    
    參數：
        dataset_name: 資料集名稱
        transactions: 交易資料
        min_sup_ratio: 最小支援度比例
        sample_rates: 抽樣比例列表（如 [0.1, 0.2, 0.3]）
        max_len: itemset 最大長度
        random_seed: 隨機種子
    
    回傳：
        包含實驗結果的字典
    """
    n_txn = len(transactions)
    if n_txn == 0:
        return None
    
    # 顯示資料集資訊
    print(f"\n{'='*60}")
    print(f"資料集: {dataset_name}")
    print(f"交易數: {n_txn}")
    print(f"MinSup Ratio: {min_sup_ratio}")
    print(f"{'='*60}\n")
    
    # 計算絕對支援度
    min_sup_abs = max(1, math.ceil(min_sup_ratio * n_txn))
    
    # 計算 Ground Truth（使用 Eclat）
    print(f"計算 Ground Truth...")
    exact_freq = brute_force_frequent_itemsets(
        transactions, min_sup_abs=min_sup_abs, max_len=max_len
    )
    print(f"  - Frequent itemsets: {len(exact_freq)}")
    
    # 準備結果容器
    methods = ["nonuni1_std", "nonuni1_sys", "nonuni2_std", "nonuni2_sys"]
    results = {
        "ratio": {m: [] for m in methods},
        "error": {m: [] for m in methods}
    }
    
    # 定義要測試的方法配置
    method_configs = [
        ("nonuni1_std", "nonuni1", False),
        ("nonuni1_sys", "nonuni1", True),
        ("nonuni2_std", "nonuni2", False),
        ("nonuni2_sys", "nonuni2", True),
    ]
    
    # 對每個 sample_rate 進行實驗
    for sr in sample_rates:
        sample_size = max(1, math.ceil(sr * n_txn))
        print(f"Sampling Rate: {int(sr*100)}% (k={sample_size})")
        
        # 測試所有方法
        for method_name, sampling_type, use_sys in method_configs:
            nc_ratio, se_rate = _run_single_method(
                method_name=method_name,
                transactions=transactions,
                min_sup_abs=min_sup_abs,
                sample_size=sample_size,
                sampling_type=sampling_type,
                use_systematic=use_sys,
                exact_freq=exact_freq,
                max_len=max_len,
                random_seed=random_seed,
            )
            results["ratio"][method_name].append(nc_ratio)
            results["error"][method_name].append(se_rate)
    
    return {
        "dataset": dataset_name,
        "min_sup_ratio": min_sup_ratio,
        "sample_rates": sample_rates,
        "results": results
    }


# 繪圖樣式配置
PLOT_STYLES = {
    "nonuni1_std": {
        "color": "skyblue",
        "linestyle": "--",
        "marker": "o",
        "label": "Non-Uni 1 (標準)"
    },
    "nonuni1_sys": {
        "color": "blue",
        "linestyle": "-",
        "marker": "s",
        "label": "Non-Uni 1 (系統抽樣)"
    },
    "nonuni2_std": {
        "color": "salmon",
        "linestyle": "--",
        "marker": "^",
        "label": "Non-Uni 2 (標準)"
    },
    "nonuni2_sys": {
        "color": "red",
        "linestyle": "-",
        "marker": "D",
        "label": "Non-Uni 2 (系統抽樣)"
    },
}


def _plot_single_metric(
    x_values: List[float],
    x_labels: List[str],
    results: Dict,
    metric_name: str,
    metric_key: str,
    dataset: str,
    min_sup: float,
    output_path: str,
):
    """
    繪製單一指標的比較圖（輔助函數）
    
    參數：
        x_values: X 軸數值
        x_labels: X 軸標籤
        results: 實驗結果
        metric_name: 指標名稱（用於顯示）
        metric_key: 指標在 results 中的 key
        dataset: 資料集名稱
        min_sup: 最小支援度
        output_path: 輸出檔案路徑
    """
    plt.figure(figsize=(10, 6))
    
    for method in ["nonuni1_std", "nonuni1_sys", "nonuni2_std", "nonuni2_sys"]:
        style = PLOT_STYLES[method]
        plt.plot(
            x_values,
            results[metric_key][method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2,
            markersize=8,
        )
    
    plt.title(
        f"{dataset} - {metric_name} (MinSup={min_sup})\n(數值越低越好)",
        fontsize=14
    )
    plt.xlabel("抽樣比例 (%)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(x_values, x_labels)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    print(f"儲存圖表: {output_path}")
    plt.close()


def plot_comparison(experiment_result: Dict, output_dir: str = None):
    """
    繪製實驗結果比較圖
    
    產生兩張圖表：
        1. Non-common Output Ratio 比較圖
        2. Support Error Rate 比較圖
    
    圖表特徵：
        - 虛線（--）：標準抽樣
        - 實線（─）：系統抽樣
        - 藍色：Non-Uni 1（基於交易長度）
        - 紅色：Non-Uni 2（基於高頻項目數）
    
    參數：
        experiment_result: 實驗結果字典
        output_dir: 輸出資料夾（None 則使用預設）
    """
    if experiment_result is None:
        return
    
    # 設定輸出資料夾
    if output_dir is None:
        output_dir = f"systematic_results_{MIN_SUP}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取資料
    dataset = experiment_result["dataset"]
    min_sup = experiment_result["min_sup_ratio"]
    sample_rates = experiment_result["sample_rates"]
    results = experiment_result["results"]
    
    # 準備 X 軸資料
    x_values = [sr * 100 for sr in sample_rates]
    x_labels = [f"{int(sr*100)}%" for sr in sample_rates]
    
    # 繪製 Non-common Output Ratio
    ratio_path = f"{output_dir}/{dataset}_minsup{min_sup}_ratio.png"
    _plot_single_metric(
        x_values, x_labels, results,
        metric_name="Non-common Output Ratio",
        metric_key="ratio",
        dataset=dataset,
        min_sup=min_sup,
        output_path=ratio_path,
    )
    
    # 繪製 Support Error Rate
    error_path = f"{output_dir}/{dataset}_minsup{min_sup}_error.png"
    _plot_single_metric(
        x_values, x_labels, results,
        metric_name="Support Error Rate",
        metric_key="error",
        dataset=dataset,
        min_sup=min_sup,
        output_path=error_path,
    )


# ==========================================
# 主程式
# ==========================================

def get_dataset_config() -> Dict:
    """
    取得資料集配置
    
    回傳：
        資料集配置字典
    
    注意：
        - 可修改 min_sup_ratio 來測試不同的最小支援度
        - 可修改 sample_rates 來測試不同的抽樣比例
        - 結果會儲存到 systematic_results_{MIN_SUP}/ 資料夾
    """
    return {
        "Retail": {
            "path": "data/retail.txt",
            "min_sup_ratio": MIN_SUP,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "BMS1": {
            "path": "data/BMS1_itemset_mining.txt",
            "min_sup_ratio": MIN_SUP,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "BMS2": {
            "path": "data/BMS2_itemset_mining.txt",
            "min_sup_ratio": MIN_SUP,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "Chainstore": {
            "path": "data/chainstoreFIM.txt",
            "min_sup_ratio": MIN_SUP,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
    }


def main():
    """
    主實驗流程
    
    執行步驟：
        1. 讀取所有資料集
        2. 對每個資料集執行實驗（4 種方法 × 5 個抽樣比例）
        3. 計算 Non-common Output Ratio 和 Support Error Rate
        4. 產生比較圖表並儲存
    
    輸出：
        - 終端機：顯示實驗進度和數值結果
        - 圖表：儲存至 systematic_results_{MIN_SUP}/ 資料夾
    """
    print("=" * 60)
    print("系統抽樣實驗 - 開始")
    print(f"全域參數：MAX_LEN={MAX_LEN}, MIN_SUP={MIN_SUP}")
    print("=" * 60)
    
    # 取得資料集配置
    datasets = get_dataset_config()
    
    # 對每個資料集進行實驗
    for ds_name, cfg in datasets.items():
        try:
            # 讀取資料
            txns = load_transactions(cfg["path"])
            
            # 限制交易數（避免記憶體不足）
            max_tx = cfg.get("max_transactions")
            if max_tx is not None and len(txns) > max_tx:
                print(f"\n[提示] {ds_name} 原有 {len(txns)} 筆交易，限制為 {max_tx} 筆")
                txns = txns[:max_tx]
            
            # 執行實驗
            result = run_experiment_with_comparison(
                dataset_name=ds_name,
                transactions=txns,
                min_sup_ratio=cfg["min_sup_ratio"],
                sample_rates=cfg["sample_rates"],
                max_len=cfg["max_len"],
                random_seed=42,
            )
            
            # 繪製圖表
            plot_comparison(result)
            
        except FileNotFoundError:
            print(f"\n❌ 錯誤：找不到資料集 {cfg['path']}")
            print(f"   請確認檔案是否存在於 data/ 資料夾中")
            continue
            
        except Exception as e:
            print(f"\n❌ 錯誤：資料集 {ds_name} 處理失敗")
            print(f"   錯誤訊息: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("系統抽樣實驗 - 完成")
    print(f"圖表已儲存至 systematic_results_{MIN_SUP}/ 資料夾")
    print("=" * 60)


if __name__ == "__main__":
    main()

