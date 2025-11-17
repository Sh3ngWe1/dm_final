# parse_results.py
"""
讀取 experiment.py 輸出的 result.csv，做以下事情：

1. 解析數據，轉成適當的數值型別
2. 輸出摘要表：
   - 每個 (dataset, sampling) 的平均 non_common_output_ratio / support_error_rate
   - 每個 dataset 的最佳 sampling（以 support_error_rate 平均值為準）
3. 繪圖（類似論文風格）：
   - x 軸：sample_rate
   - y 軸：non_common_output_ratio / support_error_rate
   - 每張圖固定：dataset + min_sup_ratio
   - 每張圖畫出 3 條線：uniform / nonuni1 / nonuni2
   - 圖檔輸出到專案根目錄的 results/ 資料夾
"""

import csv
from collections import defaultdict
from statistics import mean
from pathlib import Path

import matplotlib.pyplot as plt  # 只用 matplotlib，顏色用預設的


# 預設 result.csv 路徑（專案根目錄下）
ROOT_DIR = Path(__file__).resolve().parent
RESULT_PATH = ROOT_DIR / "final_result.csv"
OUTPUT_DIR = ROOT_DIR / "results"


# =============== 讀取與基本整理 ===============

def load_results(path: Path):
    rows = []
    # 嘗試多種編碼，因為 Windows 可能會用 UTF-16 保存 CSV
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-8"]
    last_error = None
    
    for encoding in encodings:
        try:
            rows = []  # 重置 rows
            with open(path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # 跳過空行
                    if not r or not any(r.values()):
                        continue
                    # 轉成適當型別
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
            # 如果成功讀取到數據，返回結果
            if rows:
                return rows
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            # 如果是其他錯誤（如 CSV 解析錯誤），也嘗試下一個編碼
            last_error = e
            if encoding == encodings[-1]:
                # 最後一個編碼也失敗，重新拋出錯誤
                raise
            continue
    
    # 如果所有編碼都失敗
    raise ValueError(f"無法使用任何編碼讀取文件：{path}。最後的錯誤：{last_error}")


# =============== 文本摘要：平均表與最佳 sampling ===============

def summarize_by_dataset_and_sampling(rows):
    """
    回傳：
        summary[(dataset, sampling)] = {
            "avg_non_common": ...,
            "avg_support_error": ...,
            "num_runs": ...,
        }
    """
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
    """
    印出每個 (dataset, sampling) 的平均表現。
    """
    print("\n=== 平均表現（依 dataset, sampling） ===")
    print(f"{'dataset':<12} {'sampling':<8} {'runs':>4} "
          f"{'avg_non_common':>16} {'avg_support_error':>18}")
    print("-" * 65)

    # 排序：先按 dataset，再按 sampling
    for (ds, sm), stats in sorted(summary.items()):
        print(
            f"{ds:<12} {sm:<8} {stats['num_runs']:>4} "
            f"{stats['avg_non_common']:>16.4f} {stats['avg_support_error']:>18.4f}"
        )


def print_best_sampling_per_dataset(summary):
    """
    依 dataset，選出「平均 support_error_rate 最小」的 sampling。
    """
    best = {}

    for (ds, sm), stats in summary.items():
        se = stats["avg_support_error"]
        if se != se:  # NaN 濾掉
            continue
        if ds not in best or se < best[ds]["avg_support_error"]:
            best[ds] = {
                "sampling": sm,
                "avg_support_error": se,
                "avg_non_common": stats["avg_non_common"],
                "num_runs": stats["num_runs"],
            }

    print("\n=== 每個 dataset 的最佳 sampling（以 avg_support_error 為基準） ===")
    print(f"{'dataset':<12} {'best_sampling':<12} "
          f"{'avg_support_error':>18} {'avg_non_common':>16} {'runs':>4}")
    print("-" * 70)
    for ds, info in sorted(best.items()):
        print(
            f"{ds:<12} {info['sampling']:<12} "
            f"{info['avg_support_error']:>18.4f} "
            f"{info['avg_non_common']:>16.4f} {info['num_runs']:>4}"
        )


# =============== 畫圖：類似論文的 line chart ===============

def plot_metric_for_all(rows, metric_key: str):
    """
    對於每個 dataset + min_sup_ratio，
    畫出 metric 隨 sample_rate 變化的圖（3 條線：uniform/nonuni1/nonuni2）。

    metric_key: 'non_common_output_ratio' 或 'support_error_rate'
    """
    # 建立輸出資料夾
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 先依 dataset + min_sup_ratio group 再按 sampling 分
    # 結構： grouped[(dataset, min_sup_ratio)][sampling] = list of (sample_rate, metric)
    grouped = defaultdict(lambda: defaultdict(list))

    for r in rows:
        ds = r["dataset"]
        ms = r["min_sup_ratio"]
        sm = r["sampling"]
        sr = r["sample_rate"]
        val = r[metric_key]
        if val != val:  # 排除 NaN
            continue
        grouped[(ds, ms)][sm].append((sr, val))

    for (ds, ms), sm_map in grouped.items():
        # 建立圖
        plt.figure()
        # 依 sampling 畫線，確保 sample_rate 排序
        for sm in ["uniform", "nonuni1", "nonuni2"]:
            if sm not in sm_map:
                continue
            points = sm_map[sm]
            points = sorted(points, key=lambda x: x[0])  # 按 sample_rate 排序
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

        # 檔名：results/{dataset}_ms{min_sup_ratio}_{metric}.png
        ms_str = str(ms).replace(".", "p")
        out_path = OUTPUT_DIR / f"{ds}_ms{ms_str}_{met_short}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"已輸出圖檔：{out_path}")


# =============== main ===============

def main():
    if not RESULT_PATH.exists():
        print(f"找不到 result.csv，預期路徑：{RESULT_PATH}")
        print("請先執行：python src/experiment.py > result.csv")
        return

    print(f"讀取結果檔：{RESULT_PATH}")
    rows = load_results(RESULT_PATH)
    print(f"總共有 {len(rows)} 筆實驗結果。")

    # 1. 文本摘要
    summary = summarize_by_dataset_and_sampling(rows)
    print_summary_table(summary)
    print_best_sampling_per_dataset(summary)

    # 2. 畫圖（兩種 metric 各一輪）
    print("\n開始繪圖（non_common_output_ratio）...")
    plot_metric_for_all(rows, metric_key="non_common_output_ratio")

    print("\n開始繪圖（support_error_rate）...")
    plot_metric_for_all(rows, metric_key="support_error_rate")

    print("\n✅ 全部處理完成！圖檔在 results/ 資料夾中。")


if __name__ == "__main__":
    main()
