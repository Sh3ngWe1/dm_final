# experiment_systematic.py
"""
系統抽樣實驗程式：
- 實作系統抽樣（Systematic Sampling）方法
- 比較原版抽樣（有放回）vs 系統抽樣（固定間隔）
- 測試所有資料集
- 生成比較圖表（虛線：原版，實線：系統抽樣版）

系統抽樣概念：
1. 將機率分布正規化到 [0,1] 區間（累積機率）
2. 機率越大的交易，區間越大
3. 用固定間隔（1/k）取樣，隨機起點
4. 類似「不放回」的效果，減少方差
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, FrozenSet, Dict, Tuple
import os

from sampling import get_sampling_probs
from miner import _eclat_mine
from metrics import (
    compute_non_common_output_ratio,
    compute_support_error_rate,
)

Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]

MAX_LEN = 7
MAX_TRANSACTIONS = 1500000

# 設定中文字型
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
    系統抽樣（Systematic Sampling）：
    
    1. 計算累積機率分布
    2. 將 [0,1] 區間等分成 k 份，間隔 = 1/k
    3. 在 [0, 1/k) 隨機選擇起點
    4. 以固定間隔取樣
    
    參數：
        transactions: 完整資料集
        probs: 每筆交易的抽樣機率
        k: 樣本數
        random_seed: 隨機種子
    
    回傳：
        sample_txns: 抽樣後的交易
        weights: 每筆樣本的權重
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(transactions)
    k = min(k, n)
    
    # 計算累積機率
    cum_probs = np.cumsum(probs)
    
    # 固定間隔
    step = 1.0 / k
    
    # 隨機起點（在第一個區間內）
    start_point = np.random.uniform(0, step)
    
    # 生成取樣點
    points = np.arange(start_point, 1.0, step)
    if len(points) > k:
        points = points[:k]
    
    # 找到對應的索引
    indices = np.searchsorted(cum_probs, points)
    indices = np.clip(indices, 0, n - 1)
    
    # 構造樣本和權重
    sample_txns = []
    weights = []
    
    for idx in indices:
        T = transactions[idx]
        pT = probs[idx]
        if pT <= 0.0:
            continue
        sample_txns.append(T)
        # 權重 = 1 / (p(T) * k)
        weights.append(1.0 / (pT * k))
    
    return sample_txns, weights


def standard_sampling(
    transactions: List[Transaction],
    probs: List[float],
    k: int,
    random_seed: int = None,
) -> Tuple[List[Transaction], List[float]]:
    """
    標準抽樣（有放回）：
    
    使用 np.random.choice 進行有放回抽樣
    
    參數：
        transactions: 完整資料集
        probs: 每筆交易的抽樣機率
        k: 樣本數
        random_seed: 隨機種子
    
    回傳：
        sample_txns: 抽樣後的交易
        weights: 每筆樣本的權重
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(transactions)
    
    # 有放回抽樣
    indices = np.random.choice(n, size=k, p=probs, replace=True)
    
    # 構造樣本和權重
    sample_txns = []
    weights = []
    
    for idx in indices:
        T = transactions[idx]
        pT = probs[idx]
        if pT <= 0.0:
            continue
        sample_txns.append(T)
        # 權重 = 1 / (p(T) * k)
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

def run_experiment_with_comparison(
    dataset_name: str,
    transactions: List[Transaction],
    min_sup_ratio: float,
    sample_rates: List[float],
    max_len: int,
    random_seed: int = 42,
):
    """
    對單一資料集進行完整實驗：
    - 比較 nonuni1 (std vs sys) 和 nonuni2 (std vs sys)
    - 計算 non-common output ratio 和 support error rate
    
    回傳：結果字典
    """
    n_txn = len(transactions)
    if n_txn == 0:
        return None
    
    print(f"\n{'='*60}")
    print(f"資料集: {dataset_name}")
    print(f"交易數: {n_txn}")
    print(f"MinSup Ratio: {min_sup_ratio}")
    print(f"{'='*60}\n")
    
    # 計算絕對支援度
    min_sup_abs = max(1, math.ceil(min_sup_ratio * n_txn))
    
    # 計算 Ground Truth（使用 Eclat）
    print(f"計算 Ground Truth...")
    from miner import brute_force_frequent_itemsets
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
    
    # 對每個 sample_rate 進行實驗
    for sr in sample_rates:
        sample_size = max(1, math.ceil(sr * n_txn))
        print(f"Sampling Rate: {int(sr*100)}% (k={sample_size})")
        
        # nonuni1 - standard
        approx_freq, est_sup = approximate_itemset_miner_with_sampling_method(
            transactions, min_sup_abs, sample_size,
            sampling="nonuni1", use_systematic=False,
            max_len=max_len, random_seed=random_seed
        )
        nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
        se_rate = compute_support_error_rate(exact_freq, est_sup)
        results["ratio"]["nonuni1_std"].append(nc_ratio)
        results["error"]["nonuni1_std"].append(se_rate)
        print(f"  - nonuni1_std: ratio={nc_ratio:.4f}, error={se_rate:.4f}")
        
        # nonuni1 - systematic
        approx_freq, est_sup = approximate_itemset_miner_with_sampling_method(
            transactions, min_sup_abs, sample_size,
            sampling="nonuni1", use_systematic=True,
            max_len=max_len, random_seed=random_seed
        )
        nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
        se_rate = compute_support_error_rate(exact_freq, est_sup)
        results["ratio"]["nonuni1_sys"].append(nc_ratio)
        results["error"]["nonuni1_sys"].append(se_rate)
        print(f"  - nonuni1_sys: ratio={nc_ratio:.4f}, error={se_rate:.4f}")
        
        # nonuni2 - standard
        approx_freq, est_sup = approximate_itemset_miner_with_sampling_method(
            transactions, min_sup_abs, sample_size,
            sampling="nonuni2", use_systematic=False,
            max_len=max_len, random_seed=random_seed
        )
        nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
        se_rate = compute_support_error_rate(exact_freq, est_sup)
        results["ratio"]["nonuni2_std"].append(nc_ratio)
        results["error"]["nonuni2_std"].append(se_rate)
        print(f"  - nonuni2_std: ratio={nc_ratio:.4f}, error={se_rate:.4f}")
        
        # nonuni2 - systematic
        approx_freq, est_sup = approximate_itemset_miner_with_sampling_method(
            transactions, min_sup_abs, sample_size,
            sampling="nonuni2", use_systematic=True,
            max_len=max_len, random_seed=random_seed
        )
        nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
        se_rate = compute_support_error_rate(exact_freq, est_sup)
        results["ratio"]["nonuni2_sys"].append(nc_ratio)
        results["error"]["nonuni2_sys"].append(se_rate)
        print(f"  - nonuni2_sys: ratio={nc_ratio:.4f}, error={se_rate:.4f}")
    
    return {
        "dataset": dataset_name,
        "min_sup_ratio": min_sup_ratio,
        "sample_rates": sample_rates,
        "results": results
    }


def plot_comparison(experiment_result, output_dir="systematic_results"):
    """
    繪製比較圖：
    - 虛線：原版（std）
    - 實線：系統抽樣版（sys）
    """
    if experiment_result is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = experiment_result["dataset"]
    min_sup = experiment_result["min_sup_ratio"]
    sample_rates = experiment_result["sample_rates"]
    results = experiment_result["results"]
    
    x_values = [sr * 100 for sr in sample_rates]
    x_labels = [f"{int(sr*100)}%" for sr in sample_rates]
    
    # 定義樣式
    styles = {
        "nonuni1_std": {"color": "skyblue", "linestyle": "--", "marker": "o", "label": "Non-Uni 1 (標準)"},
        "nonuni1_sys": {"color": "blue", "linestyle": "-", "marker": "s", "label": "Non-Uni 1 (系統抽樣)"},
        "nonuni2_std": {"color": "salmon", "linestyle": "--", "marker": "^", "label": "Non-Uni 2 (標準)"},
        "nonuni2_sys": {"color": "red", "linestyle": "-", "marker": "D", "label": "Non-Uni 2 (系統抽樣)"},
    }
    
    # 圖 1: Non-common Output Ratio
    plt.figure(figsize=(10, 6))
    for method in ["nonuni1_std", "nonuni1_sys", "nonuni2_std", "nonuni2_sys"]:
        style = styles[method]
        plt.plot(
            x_values,
            results["ratio"][method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2,
            markersize=8,
        )
    
    plt.title(f"{dataset} - Non-common Output Ratio (MinSup={min_sup})\n(數值越低越好)", fontsize=14)
    plt.xlabel("抽樣比例 (%)", fontsize=12)
    plt.ylabel("Non-common Output Ratio", fontsize=12)
    plt.xticks(x_values, x_labels)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    filename1 = f"{output_dir}/{dataset}_minsup{min_sup}_ratio.png"
    plt.savefig(filename1, dpi=150)
    print(f"儲存圖表: {filename1}")
    plt.close()
    
    # 圖 2: Support Error Rate
    plt.figure(figsize=(10, 6))
    for method in ["nonuni1_std", "nonuni1_sys", "nonuni2_std", "nonuni2_sys"]:
        style = styles[method]
        plt.plot(
            x_values,
            results["error"][method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
            linewidth=2,
            markersize=8,
        )
    
    plt.title(f"{dataset} - Support Error Rate (MinSup={min_sup})\n(數值越低越好)", fontsize=14)
    plt.xlabel("抽樣比例 (%)", fontsize=12)
    plt.ylabel("Support Error Rate", fontsize=12)
    plt.xticks(x_values, x_labels)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    filename2 = f"{output_dir}/{dataset}_minsup{min_sup}_error.png"
    plt.savefig(filename2, dpi=150)
    print(f"儲存圖表: {filename2}")
    plt.close()


# ==========================================
# 主程式
# ==========================================

def main():
    """主實驗流程"""
    
    # 資料集配置
    DATASETS = {
        "Retail": {
            "path": "data/retail.txt",
            "min_sup_ratio": 0.02,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "BMS1": {
            "path": "data/BMS1_itemset_mining.txt",
            "min_sup_ratio": 0.02,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "BMS2": {
            "path": "data/BMS2_itemset_mining.txt",
            "min_sup_ratio": 0.02,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "Chainstore": {
            "path": "data/chainstoreFIM.txt",
            "min_sup_ratio": 0.01,
            "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
    }
    
    print("="*60)
    print("系統抽樣實驗 - 開始")
    print("="*60)
    
    # 對每個資料集進行實驗
    for ds_name, cfg in DATASETS.items():
        try:
            # 讀取資料
            txns = load_transactions(cfg["path"])
            
            # 限制交易數
            max_tx = cfg.get("max_transactions")
            if max_tx is not None and len(txns) > max_tx:
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
            
        except Exception as e:
            print(f"\n錯誤：資料集 {ds_name} 處理失敗")
            print(f"錯誤訊息: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("系統抽樣實驗 - 完成")
    print("="*60)


if __name__ == "__main__":
    main()

