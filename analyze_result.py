import csv
from collections import defaultdict
from statistics import mean
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent
RESULT_PATH = ROOT_DIR / "final_result.csv"
OUTPUT_DIR = ROOT_DIR / "results"

def load_results(path: Path):
    rows = []
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-8"]
    last_error = None
    
    for encoding in encodings:
        try:
            rows = []
            with open(path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if not r or not any(r.values()):
                        continue
                    r["min_sup_ratio"] = float(r["min_sup_ratio"])
                    r["sample_rate"] = float(r["sample_rate"])
                    r["n_txn"] = int(r["n_txn"])
                    r["min_sup_abs"] = int(r["min_sup_abs"])
                    r["sample_size"] = int(r["sample_size"])
                    r["true_num_freq"] = int(r["true_num_freq"])
                    r["approx_num_freq"] = int(r["approx_num_freq"])
                    r["non_common_output_ratio"] = float(r["non_common_output_ratio"])
                    r["support_error_rate"] = float(r["support_error_rate"])
                    rows.append(r)
            if rows:
                return rows
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            if encoding == encodings[-1]:
                raise
            continue
    
    raise ValueError(f"無法使用任何編碼讀取文件：{path}。最後的錯誤：{last_error}")

def summarize_by_dataset_and_sampling(rows):
    grouped = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["sampling"])
        grouped[key].append(r)

    summary = {}
    for key, rs in grouped.items():
        avg_nc = mean(r["non_common_output_ratio"] for r in rs)
        avg_se = mean(r["support_error_rate"] for r in rs)
        summary[key] = {
            "avg_non_common": avg_nc,
            "avg_support_error": avg_se,
            "num_runs": len(rs),
        }
    return summary


def print_summary_table(summary):
    print("\n=== Summary table (by dataset and sampling) ===")
    print(f"{'dataset':<12} {'sampling':<8} {'runs':>4} "
          f"{'avg_non_common':>16} {'avg_support_error':>18}")
    print("-" * 65)

    for (ds, sm), stats in sorted(summary.items()):
        print(
            f"{ds:<12} {sm:<8} {stats['num_runs']:>4} "
            f"{stats['avg_non_common']:>16.4f} {stats['avg_support_error']:>18.4f}"
        )


def print_best_sampling_per_dataset(summary):
    best = {}

    for (ds, sm), stats in summary.items():
        se = stats["avg_support_error"]
        if se != se:
            continue
        if ds not in best or se < best[ds]["avg_support_error"]:
            best[ds] = {
                "sampling": sm,
                "avg_support_error": se,
                "avg_non_common": stats["avg_non_common"],
                "num_runs": stats["num_runs"],
            }

    print("\n=== Best sampling (by avg_support_error) ===")
    print(f"{'dataset':<12} {'best_sampling':<12} "
          f"{'avg_support_error':>18} {'avg_non_common':>16} {'runs':>4}")
    print("-" * 70)
    for ds, info in sorted(best.items()):
        print(
            f"{ds:<12} {info['sampling']:<12} "
            f"{info['avg_support_error']:>18.4f} "
            f"{info['avg_non_common']:>16.4f} {info['num_runs']:>4}"
        )


def plot_metric_for_all(rows, metric_key: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(lambda: defaultdict(list))

    for r in rows:
        ds = r["dataset"]
        ms = r["min_sup_ratio"]
        sm = r["sampling"]
        sr = r["sample_rate"]
        val = r[metric_key]
        if val != val:
            continue
        grouped[(ds, ms)][sm].append((sr, val))

    for (ds, ms), sm_map in grouped.items():
        plt.figure()
        for sm in ["uniform", "nonuni1", "nonuni2"]:
            if sm not in sm_map:
                continue
            points = sm_map[sm]
            points = sorted(points, key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, marker="o", label=sm)

        plt.xlabel("sample_rate")
        if metric_key == "non_common_output_ratio":
            plt.ylabel("Non-common output ratio")
            met_short = "non_common"
        else:
            plt.ylabel("Support error rate")
            met_short = "support_error"

        plt.title(f"{ds} (min_sup_ratio={ms}) - {metric_key}")
        plt.legend()
        plt.grid(True)

        ms_str = str(ms).replace(".", "p")
        out_path = OUTPUT_DIR / f"{ds}_ms{ms_str}_{met_short}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Plots saved to {out_path}")


def main():
    if not RESULT_PATH.exists():
        print(f"Cannot find result.csv, expected path: {RESULT_PATH}.")
        print("Please run: python src/experiment.py > result.csv.")
        return

    print(f"Reading results file: {RESULT_PATH}")
    rows = load_results(RESULT_PATH)
    print(f"Total {len(rows)} experiments results loaded.")

    summary = summarize_by_dataset_and_sampling(rows)
    print_summary_table(summary)
    print_best_sampling_per_dataset(summary)

    print("\nStart plotting (non_common_output_ratio)...")
    plot_metric_for_all(rows, metric_key="non_common_output_ratio")

    print("\nStart plotting (support_error_rate)...")
    plot_metric_for_all(rows, metric_key="support_error_rate")

    print("\nAll done! Plots saved in results/ folder.")


if __name__ == "__main__":
    main()
