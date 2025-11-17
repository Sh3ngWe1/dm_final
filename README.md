# Non-uniform Sampling for Large Itemset Mining (Reimplementation)

本專案是論文 **「Non-uniform Sampling Methods for Large Itemset Mining」** 的完整 Python 復現與實驗平台。

---

## 1. 安裝環境與套件

建議使用虛擬環境：

```bash
python -m venv .dmvenv
source .dmvenv/bin/activate       # macOS / Linux
.dmvenv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## 2. 專案結構

```
final_project/
├── data/                      # 放所有 dataset（retail, BMS1, BMS2, chainstore）
├── src/
│   ├── sampling.py            # 三種 sampling 方法 (uniform, non-uni1, non-uni2)
│   ├── miner.py               # Weighted Eclat 實作（exact + approximate）
│   └── experiment.py          # 主實驗程式
├──  analyze_result.py      # 結果分析與繪圖
├── results/                   # analyze_result.py 產生的圖表
├── result.csv                 # experiment.py 產出的結果
├── requirements.txt
└── README.md
```

---

## 3. 調整全域參數：`MAX_LEN` 與 `MAX_TRANSACTIONS`

在 `src/experiment.py` 中：

```python
MAX_LEN = 3
MAX_TRANSACTIONS = None
```

### `MAX_LEN`
- 控制 **itemset 的最大長度**（例如 3 ⇒ 只產生長度 1–3 的 frequent itemsets）
- 數值越大，運算量越高
- Debug 建議 3；想更接近論文可調成 5-7

### `MAX_TRANSACTIONS`
- 控制 **最多使用多少筆交易資料**
- `None` ⇒ 使用完整 dataset
- 可以先設成 3000 加速測試

---

## 4. 執行完整實驗：生成 `result.csv`

```bash
python src/experiment.py > result.csv
```

這會做：

- 用 Eclat 挖出 exact frequent itemsets  
- 用三種 sampling（uniform、non-uni1、non-uni2）做 approximate mining  
- 計算  
  - non_common_output_ratio  
  - support_error_rate  
- 把所有結果輸出成 CSV

---

## 5. 分析結果並產生圖表

```bash
python src/analyze_result.py
```

內容包括：

- 自動計算每個 dataset / sampling 的平均錯誤
- 找出最佳 sampling 方法
- 產生折線圖（依 dataset + min_sup_ratio）
- 圖片會輸出到：`results/`

---

## 6. 圖片輸出位置

```
results/
  ├── retail_ms0p02_non_common.png
  ├── retail_ms0p02_support_error.png
  ├── bms1_ms0p03_non_common.png
  └── ...
```

每張圖都包含三種 sampling 方法（uniform / non-uni1 / non-uni2）的比較。

---
#Todo: 

1. 把 max_len 改成 7（跟論文更近）
2. 把 min_sup 調成 0.005 / 0.01（跟原始 paper 一樣）
3. 把 Eclat 換成 LCM（SPMF 同等級）
4. 用固定 seed 完全做 deterministic run
