import math
import os
from typing import List, FrozenSet, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sampling import get_sampling_probs
from miner import _eclat_mine, brute_force_frequent_itemsets
from metrics import compute_non_common_output_ratio, compute_support_error_rate

Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]

MAX_LEN = None
MAX_TRANSACTIONS = 1500000
MIN_SUP = 0.005

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Microsoft JhengHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

def systematic_sampling(
    transactions: List[Transaction],
    probs: List[float],
    k: int,
    random_seed: int = None,
) -> Tuple[List[Transaction], List[float]]:
    if random_seed is not None:
        np.random.seed(random_seed)
    n = len(transactions)
    k = min(k, n)
    cum_probs = np.cumsum(probs)
    step = 1.0 / k
    start_point = np.random.uniform(0, step)
    points = np.arange(start_point, 1.0, step)[:k]
    indices = np.searchsorted(cum_probs, points)
    indices = np.clip(indices, 0, n - 1)
    
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


def standard_sampling(
    transactions: List[Transaction],
    probs: List[float],
    k: int,
    random_seed: int = None,
) -> Tuple[List[Transaction], List[float]]:
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(transactions)
    
    indices = np.random.choice(n, size=k, p=probs, replace=True)
    
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

def approximate_itemset_miner_with_sampling_method(
    transactions: List[Transaction],
    min_sup_abs: int,
    k: int,
    sampling: str = "uniform",
    use_systematic: bool = False,
    max_len: int = None,
    random_seed: int = None,
):
    n = len(transactions)
    if n == 0 or k <= 0:
        return {}, {}
    
    if max_len is None:
        max_len = max(len(t) for t in transactions)
    
    probs = get_sampling_probs(
        transactions=transactions,
        sampling=sampling,
        min_sup_abs=min_sup_abs,
    )
    
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
    
    est_support = _eclat_mine(
        transactions=sample_txns,
        weights=weights,
        min_sup=min_sup_abs,
        max_len=max_len,
    )
    
    approx_frequent = dict(est_support)
    return approx_frequent, est_support


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
    n_txn = len(transactions)
    if n_txn == 0:
        return None
    
    print(f"\n{'='*60}")
    print(f"資料集: {dataset_name}")
    print(f"交易數: {n_txn}")
    print(f"MinSup Ratio: {min_sup_ratio}")
    print(f"{'='*60}\n")
    
    min_sup_abs = max(1, math.ceil(min_sup_ratio * n_txn))
    
    print(f"計算 Ground Truth...")
    exact_freq = brute_force_frequent_itemsets(
        transactions, min_sup_abs=min_sup_abs, max_len=max_len
    )
    print(f"  - Frequent itemsets: {len(exact_freq)}")
    
    methods = ["nonuni1_std", "nonuni1_sys", "nonuni2_std", "nonuni2_sys"]
    results = {
        "ratio": {m: [] for m in methods},
        "error": {m: [] for m in methods}
    }
    
    method_configs = [
        ("nonuni1_std", "nonuni1", False),
        ("nonuni1_sys", "nonuni1", True),
        ("nonuni2_std", "nonuni2", False),
        ("nonuni2_sys", "nonuni2", True),
    ]
    
    for sr in sample_rates:
        sample_size = max(1, math.ceil(sr * n_txn))
        print(f"Sampling Rate: {int(sr*100)}% (k={sample_size})")
        
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


PLOT_STYLES = {
    "nonuni1_std": {
        "color": "skyblue",
        "linestyle": "--",
        "marker": "o",
        "label": "Non-Uni 1 (standard)"
    },
    "nonuni1_sys": {
        "color": "blue",
        "linestyle": "-",
        "marker": "s",
        "label": "Non-Uni 1 (systematic)"
    },
    "nonuni2_std": {
        "color": "salmon",
        "linestyle": "--",
        "marker": "^",
        "label": "Non-Uni 2 (standard)"
    },
    "nonuni2_sys": {
        "color": "red",
        "linestyle": "-",
        "marker": "D",
        "label": "Non-Uni 2 (systematic)"
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
    print(f"Save chart: {output_path}")
    plt.close()


def plot_comparison(experiment_result: Dict, output_dir: str = None):
    if experiment_result is None:
        return
    
    if output_dir is None:
        output_dir = f"systematic_results_{MIN_SUP}"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = experiment_result["dataset"]
    min_sup = experiment_result["min_sup_ratio"]
    sample_rates = experiment_result["sample_rates"]
    results = experiment_result["results"]
    
    x_values = [sr * 100 for sr in sample_rates]
    x_labels = [f"{int(sr*100)}%" for sr in sample_rates]
    
    ratio_path = f"{output_dir}/{dataset}_minsup{min_sup}_ratio.png"
    _plot_single_metric(
        x_values, x_labels, results,
        metric_name="Non-common Output Ratio",
        metric_key="ratio",
        dataset=dataset,
        min_sup=min_sup,
        output_path=ratio_path,
    )
    
    error_path = f"{output_dir}/{dataset}_minsup{min_sup}_error.png"
    _plot_single_metric(
        x_values, x_labels, results,
        metric_name="Support Error Rate",
        metric_key="error",
        dataset=dataset,
        min_sup=min_sup,
        output_path=error_path,
    )


def get_dataset_config() -> Dict:
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
    print("=" * 60)
    print("Systematic sampling experiment - started")
    print(f"Global parameters: MAX_LEN={MAX_LEN}, MIN_SUP={MIN_SUP}")
    print("=" * 60)
    
    datasets = get_dataset_config()
    
    for ds_name, cfg in datasets.items():
        try:
            txns = load_transactions(cfg["path"])
            
            max_tx = cfg.get("max_transactions")
            if max_tx is not None and len(txns) > max_tx:
                print(f"\n[Hint] {ds_name} originally has {len(txns)} transactions, limited to {max_tx} transactions")
                txns = txns[:max_tx]
            
            result = run_experiment_with_comparison(
                dataset_name=ds_name,
                transactions=txns,
                min_sup_ratio=cfg["min_sup_ratio"],
                sample_rates=cfg["sample_rates"],
                max_len=cfg["max_len"],
                random_seed=42,
            )
            
            plot_comparison(result)
            
        except FileNotFoundError:
            print(f"\nError: dataset {cfg['path']} not found")
            print(f"   Please check if the file exists in the data/ folder")
            continue
            
        except Exception as e:
            print(f"\nError: dataset {ds_name} processing failed")
            print(f"   Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("Systematic sampling experiment - completed")
    print(f"Charts saved to systematic_results_{MIN_SUP}/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()

