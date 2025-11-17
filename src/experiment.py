# experiment.py
"""
實驗主程式：
- 讀取多個 dataset
- 在完整 DB 上用 brute force 找 true frequent itemsets
- 用三種 sampling 方法做 approximate mining
- 計算 non-common output ratio & support error rate
- 以 CSV 格式輸出結果（可以用 > 轉存成檔案）
"""

import math
from typing import List, FrozenSet, Dict, Tuple

from sampling import get_sampling_probs  # 這行沒直接用，但保留以防你之後擴充
from miner import brute_force_frequent_itemsets, approximate_itemset_miner
from metrics import (
    compute_non_common_output_ratio,
    compute_support_error_rate,
)

Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]

MAX_LEN = 3
MAX_TRANSACTIONS = None

# ---------- 讀檔工具 ----------

def load_transactions(path: str, sep: str = " ") -> List[Transaction]:
    """
    讀 SPMF 格式的交易檔：
    - 每一行是一筆交易
    - item 用空白（或 sep）分隔
    """
    txns: List[Transaction] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(sep)
            txns.append(frozenset(items))
    return txns


# ---------- 單一組設定：跑一輪實驗 ----------

def run_single_setting(
    dataset_name: str,
    transactions: List[Transaction],
    min_sup_ratio: float,
    sample_rate: float,
    sampling: str,
    max_len: int,
    random_seed: int = 42,
) -> Dict:
    """
    在固定：
    - dataset
    - min_sup_ratio
    - sample_rate
    - sampling method
    下，跑一輪實驗並回傳指標。
    """
    n_txn = len(transactions)
    if n_txn == 0:
        return {
            "dataset": dataset_name,
            "min_sup_ratio": min_sup_ratio,
            "sample_rate": sample_rate,
            "sampling": sampling,
            "n_txn": 0,
            "min_sup_abs": 0,
            "sample_size": 0,
            "true_num_freq": 0,
            "approx_num_freq": 0,
            "non_common_output_ratio": float("nan"),
            "support_error_rate": float("nan"),
        }

    # min_sup & sample_size 轉成筆數
    min_sup_abs = max(1, math.ceil(min_sup_ratio * n_txn))
    sample_size = max(1, math.ceil(sample_rate * n_txn))

    # 1. ground truth（完整 DB）
    exact_freq = brute_force_frequent_itemsets(
        transactions, min_sup_abs=min_sup_abs, max_len=max_len
    )

    # 2. 抽樣 approximate
    approx_freq, est_sup = approximate_itemset_miner(
        transactions,
        min_sup_abs=min_sup_abs,
        k=sample_size,
        sampling=sampling,
        max_len=max_len,
        random_seed=random_seed,
    )

    # 3. 指標
    nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
    se_rate = compute_support_error_rate(exact_freq, est_sup)

    return {
        "dataset": dataset_name,
        "min_sup_ratio": min_sup_ratio,
        "sample_rate": sample_rate,
        "sampling": sampling,
        "n_txn": n_txn,
        "min_sup_abs": min_sup_abs,
        "sample_size": sample_size,
        "true_num_freq": len(exact_freq),
        "approx_num_freq": len(approx_freq),
        "non_common_output_ratio": nc_ratio,
        "support_error_rate": se_rate,
    }


# ---------- 主實驗流程 ----------

def run_full_experiments():
    """
    這裡設定要跑哪些 dataset / min_sup_ratio / sample_rate / max_len。

    **注意：**
    真實 SPMF dataset 很大，用 brute-force miner 會爆時間。
    建議：
        1. 先只跑 retail
        2. 或在下面加上 max_transactions 做子樣本測試
    """
    # 1. 設定 datasets 配置
    DATASETS = {
        "retail": {
            "path": "data/retail.txt",
            "min_sup_ratios": [0.02, 0.03],   # 可以依照 paper 調
            "sample_rates": [0.1, 0.2, 0.3],
            "max_len": MAX_LEN,                     # itemset 長度上限（避免爆炸）
            "max_transactions": MAX_TRANSACTIONS,         # 先只取前 3000 筆當 demo
        },
        "bms1": {
            "path": "data/BMS1_itemset_mining.txt",
            "min_sup_ratios": [0.02, 0.03],
            "sample_rates": [0.1, 0.2, 0.3],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "bms2": {
            "path": "data/BMS2_itemset_mining.txt",
            "min_sup_ratios": [0.02, 0.03],
            "sample_rates": [0.1, 0.2, 0.3],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "chainstore": {
            "path": "data/chainstoreFIM.txt",
            "min_sup_ratios": [0.01, 0.02],
            "sample_rates": [0.1, 0.2],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
    }

    sampling_methods = ["uniform", "nonuni1", "nonuni2"]

    # 2. 印 CSV header
    print(
        "dataset,min_sup_ratio,sample_rate,sampling,"
        "n_txn,min_sup_abs,sample_size,true_num_freq,approx_num_freq,"
        "non_common_output_ratio,support_error_rate"
    )

    # 3. 逐 dataset / min_sup_ratio / sample_rate / sampling 跑
    for ds_name, cfg in DATASETS.items():
        # 讀資料
        txns = load_transactions(cfg["path"])
        # 若有 max_transactions，就只取前 N 筆（避免實驗太大）
        max_tx = cfg.get("max_transactions")
        if max_tx is not None and len(txns) > max_tx:
            txns = txns[:max_tx]

        max_len = cfg["max_len"]

        for msr in cfg["min_sup_ratios"]:
            for sr in cfg["sample_rates"]:
                for sm in sampling_methods:
                    res = run_single_setting(
                        dataset_name=ds_name,
                        transactions=txns,
                        min_sup_ratio=msr,
                        sample_rate=sr,
                        sampling=sm,
                        max_len=max_len,
                        random_seed=42,
                    )

                    # 以 CSV 一行輸出
                    print(
                        f"{res['dataset']},"
                        f"{res['min_sup_ratio']},"
                        f"{res['sample_rate']},"
                        f"{res['sampling']},"
                        f"{res['n_txn']},"
                        f"{res['min_sup_abs']},"
                        f"{res['sample_size']},"
                        f"{res['true_num_freq']},"
                        f"{res['approx_num_freq']},"
                        f"{res['non_common_output_ratio']},"
                        f"{res['support_error_rate']}"
                    )


# ---------- 直接執行 ----------

if __name__ == "__main__":
    run_full_experiments()
