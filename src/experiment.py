import math
from typing import List, FrozenSet, Dict, Tuple

from miner import brute_force_frequent_itemsets, approximate_itemset_miner
from metrics import (
    compute_non_common_output_ratio,
    compute_support_error_rate,
)

Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]

MAX_LEN = 3
# MAX_TRANSACTIONS = None
MAX_TRANSACTIONS = 1500000

def load_transactions(path: str, sep: str = " ") -> List[Transaction]:
    txns: List[Transaction] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split(sep)
            txns.append(frozenset(items))
    return txns


def run_single_setting(
    dataset_name: str,
    transactions: List[Transaction],
    min_sup_ratio: float,
    sample_rate: float,
    sampling: str,
    max_len: int,
    random_seed: int = 42,
) -> Dict:
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

    min_sup_abs = max(1, math.ceil(min_sup_ratio * n_txn))
    sample_size = max(1, math.ceil(sample_rate * n_txn))

    exact_freq = brute_force_frequent_itemsets(
        transactions, min_sup_abs=min_sup_abs, max_len=max_len
    )

    approx_freq, est_sup = approximate_itemset_miner(
        transactions,
        min_sup_abs=min_sup_abs,
        k=sample_size,
        sampling=sampling,
        max_len=max_len,
        random_seed=random_seed,
    )

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


def run_full_experiments():
    DATASETS = {
        "retail": {
            "path": "data/retail.txt",
            "min_sup_ratios": [0.02, 0.03],
            "sample_rates": [0.1, 0.2, 0.3],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
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

    print(
        "dataset,min_sup_ratio,sample_rate,sampling,"
        "n_txn,min_sup_abs,sample_size,true_num_freq,approx_num_freq,"
        "non_common_output_ratio,support_error_rate"
    )

    for ds_name, cfg in DATASETS.items():
        txns = load_transactions(cfg["path"])
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


if __name__ == "__main__":
    run_full_experiments()
