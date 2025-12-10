import math
import os
import sys
from typing import List, FrozenSet, Dict, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt

from miner import brute_force_frequent_itemsets, approximate_itemset_miner
from metrics import (
    compute_non_common_output_ratio,
    compute_support_error_rate,
)

Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]

MAX_LEN = 10
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

    print(f"  [1/4] Calculate ground truth (brute force)...", file=sys.stderr, flush=True)
    exact_freq = brute_force_frequent_itemsets(
        transactions, min_sup_abs=min_sup_abs, max_len=max_len
    )

    print(f"  [2/4] Run sampling mining (sampling={sampling})...", file=sys.stderr, flush=True)
    approx_freq, est_sup = approximate_itemset_miner(
        transactions,
        min_sup_abs=min_sup_abs,
        k=sample_size,
        sampling=sampling,
        max_len=max_len,
        random_seed=random_seed,
    )

    print(f"  [3/4] Calculate non-common output ratio...", file=sys.stderr, flush=True)
    nc_ratio = compute_non_common_output_ratio(exact_freq, approx_freq)
    print(f"  [4/4] Calculate support error rate...", file=sys.stderr, flush=True)
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


def plot_for_dataset(dataset_name: str, results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    sample_rates = sorted({r["sample_rate"] for r in results})
    min_sups = sorted({r["min_sup_ratio"] for r in results})
    sampling_methods = sorted({r["sampling"] for r in results})

    for sr in sample_rates:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for sm in sampling_methods:
            rows = [
                r for r in results
                if r["sample_rate"] == sr and r["sampling"] == sm
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda x: x["min_sup_ratio"])
            xs = [r["min_sup_ratio"] for r in rows]
            ys_nc = [r["non_common_output_ratio"] for r in rows]
            ys_se = [r["support_error_rate"] for r in rows]

            ax1.plot(xs, ys_nc, marker="o", label=f"{sm} (non-common)")
            ax2.plot(xs, ys_se, linestyle="--", marker="x", label=f"{sm} (support err)")

        ax1.set_xlabel("min_sup_ratio")
        ax1.set_ylabel("non-common output ratio")
        ax2.set_ylabel("support error rate")
        ax1.set_title(f"{dataset_name} - Effect of min_sup (K={sr})")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{dataset_name}_vary_minsup_K{sr}.png")
        plt.savefig(out_path)
        plt.close(fig)

    for ms in min_sups:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for sm in sampling_methods:
            rows = [
                r for r in results
                if abs(r["min_sup_ratio"] - ms) < 1e-12 and r["sampling"] == sm
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda x: x["sample_rate"])
            xs = [r["sample_rate"] for r in rows]
            ys_nc = [r["non_common_output_ratio"] for r in rows]
            ys_se = [r["support_error_rate"] for r in rows]

            ax1.plot(xs, ys_nc, marker="o", label=f"{sm} (non-common)")
            ax2.plot(xs, ys_se, linestyle="--", marker="x", label=f"{sm} (support err)")

        ax1.set_xlabel("sample_rate (K)")
        ax1.set_ylabel("non-common output ratio")
        ax2.set_ylabel("support error rate")
        ax1.set_title(f"{dataset_name} - Effect of K (min_sup={ms})")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{dataset_name}_vary_k_minsup{ms}.png")
        plt.savefig(out_path)
        plt.close(fig)


def run_full_experiments():
    DATASETS = {
        # "OnlineRetail": {
        #     "path": "data/online_retail.txt",
        #     "min_sup_ratios": [0.002, 0.01],
        #     "sample_rates": [0.005, 0.01],
        #     "max_len": MAX_LEN,
        #     "max_transactions": MAX_TRANSACTIONS,
        # },
        "chainstore": {
            "path": "data/chainstoreFIM.txt",
            "min_sup_ratios": [0.005, 0.01],
            "sample_rates": [0.005, 0.01],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "BMS2": {
            "path": "data/BMS2_itemset_mining.txt",
            "min_sup_ratios": [0.005],
            "sample_rates": [0.005, 0.01, 0.05],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "T10I4D100K": {
            "path": "data/T10I4D100K.txt",
            "min_sup_ratios": [0.005, 0.01],
            "sample_rates": [0.05],
            "max_len": MAX_LEN,
            "max_transactions": MAX_TRANSACTIONS,
        },
        "Retail": {
            "path": "data/retail.txt",
            "min_sup_ratios": [0.005, 0.01],
            "sample_rates": [0.05],
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

    all_results: List[Dict] = []
    
    total_experiments = 0
    for ds_name, cfg in DATASETS.items():
        if os.path.exists(cfg["path"]):
            total_experiments += len(cfg["min_sup_ratios"]) * len(cfg["sample_rates"]) * len(sampling_methods)
    
    current_experiment = 0

    for ds_name, cfg in DATASETS.items():
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"# [WARN] Dataset file not found, skip: {ds_name} ({path})")
            continue

        print(f"# [INFO] Loading dataset: {ds_name} from {path}")
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Processing dataset: {ds_name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        
        txns = load_transactions(path)
        max_tx = cfg.get("max_transactions")
        if max_tx is not None and len(txns) > max_tx:
            txns = txns[:max_tx]

        max_len = cfg["max_len"]

        for msr in cfg["min_sup_ratios"]:
            for sr in cfg["sample_rates"]:
                for sm in sampling_methods:
                    current_experiment += 1
                    print(f"\n[{current_experiment}/{total_experiments}] {ds_name} | min_sup={msr}, sample_rate={sr}, sampling={sm}", 
                          file=sys.stderr, flush=True)
                    
                    res = run_single_setting(
                        dataset_name=ds_name,
                        transactions=txns,
                        min_sup_ratio=msr,
                        sample_rate=sr,
                        sampling=sm,
                        max_len=max_len,
                        random_seed=42,
                    )
                    all_results.append(res)
                    
                    print(f"Completed (non-common ratio: {res['non_common_output_ratio']:.4f}, support error: {res['support_error_rate']:.4f})", 
                          file=sys.stderr, flush=True)

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

    output_dir = "new_results"
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Start generating plots...", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr, flush=True)
    
    dataset_names = sorted({r["dataset"] for r in all_results})
    for idx, ds_name in enumerate(dataset_names, 1):
        ds_results = [r for r in all_results if r["dataset"] == ds_name]
        if not ds_results:
            continue
        print(f"[{idx}/{len(dataset_names)}] Generating plots for {ds_name}...", file=sys.stderr, flush=True)
        plot_for_dataset(ds_name, ds_results, output_dir)
        print(f"Plots saved to {output_dir}/{ds_name}...", file=sys.stderr, flush=True)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"All experiments completed!", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr, flush=True)
    
    print("\n\n# ===== SUMMARY (for ChatGPT analysis) =====")
    summary = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        summary[r["dataset"]][r["sampling"]].append(r)

    for ds_name in sorted(summary.keys()):
        print(f"\n## Dataset: {ds_name}")
        for sm, rows in summary[ds_name].items():
            avg_nc = sum(r["non_common_output_ratio"] for r in rows) / len(rows)
            avg_se = sum(r["support_error_rate"] for r in rows) / len(rows)
            print(
                f"- sampling = {sm}: "
                f"avg_non_common_output_ratio = {avg_nc:.4f}, "
                f"avg_support_error_rate = {avg_se:.4f} "
                f"(over {len(rows)} settings)"
            )
        best_nc = min(
            summary[ds_name].items(),
            key=lambda kv: sum(r["non_common_output_ratio"] for r in kv[1]) / len(kv[1]),
        )[0]
        best_se = min(
            summary[ds_name].items(),
            key=lambda kv: sum(r["support_error_rate"] for r in kv[1]) / len(kv[1]),
        )[0]
        print(f"-> Best (non-common output ratio): {best_nc}")
        print(f"-> Best (support error rate): {best_se}")


if __name__ == "__main__":
    run_full_experiments()
