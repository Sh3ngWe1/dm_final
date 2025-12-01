# 系統抽樣於頻繁項目集挖掘之研究

# Systematic Sampling for Frequent Itemset Mining

> 本專案實作並比較**標準有放回抽樣**與**系統抽樣**在頻繁項目集挖掘中的效果，並提出最佳組合建議。

---

## 📚 專案文件

- **[README.md](README.md)**（本文件）：快速開始與操作指南
- **[README_SYS.md](README_SYS.md)**：系統抽樣原理、理論背景與技術細節
- **[RESULT_ANALYSIS.md](RESULT_ANALYSIS.md)**：實驗結果分析與解釋

---

## 🚀 快速開始

### 1. 安裝環境與套件

建議使用虛擬環境：

```bash
# 建立虛擬環境
python -m venv .dmvenv

# 啟動虛擬環境
source .dmvenv/bin/activate       # macOS / Linux
.dmvenv\Scripts\activate          # Windows

# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 準備資料集

確保以下資料集位於 `data/` 資料夾：

```
data/
├── retail.txt
├── BMS1_itemset_mining.txt
├── BMS2_itemset_mining.txt
└── chainstoreFIM.txt
```

---

## 🔬 執行實驗

### 主實驗：系統抽樣比較

```bash
python src/experiment_systematic.py
```

#### 實驗內容：

- ✅ 測試 4 個資料集（Retail, BMS1, BMS2, Chainstore）
- ✅ 比較 4 種方法：
  - Non-Uni 1 (標準) - 基於交易長度
  - Non-Uni 1 (系統抽樣)
  - Non-Uni 2 (標準) - 基於高頻項目數
  - Non-Uni 2 (系統抽樣)
- ✅ 評估 2 個指標：
  - Non-common Output Ratio（集合差異比）
  - Support Error Rate（支援度誤差率）
- ✅ 測試 5 個抽樣比例：10%, 20%, 30%, 40%, 50%

#### 輸出說明：

**終端機輸出：**

```
============================================================
資料集: Retail
交易數: 88162
MinSup Ratio: 0.02
============================================================

計算 Ground Truth...
  - Frequent itemsets: 55
Sampling Rate: 10% (k=8817)
  - nonuni1_std: ratio=0.1091, error=0.0529
  - nonuni1_sys: ratio=0.0727, error=0.0320
  - nonuni2_std: ratio=0.1273, error=0.0315
  - nonuni2_sys: ratio=0.1273, error=0.0311
...
```

**圖表輸出：**

圖表會自動儲存至 `systematic_results_{MIN_SUP}/` 資料夾：

```
systematic_results_0.02/
├── Retail_minsup0.02_ratio.png      # Non-common Output Ratio
├── Retail_minsup0.02_error.png      # Support Error Rate
├── BMS1_minsup0.02_ratio.png
├── BMS1_minsup0.02_error.png
├── BMS2_minsup0.02_ratio.png
├── BMS2_minsup0.02_error.png
├── Chainstore_minsup0.01_ratio.png
└── Chainstore_minsup0.01_error.png
```

> **注意**：資料夾名稱會根據 MinSup 設定自動變化（如 `systematic_results_0.005/`）

---

## ⚙️ 自訂實驗參數

編輯 `src/experiment_systematic.py` 中的設定：

```python
# 全域參數
MAX_LEN = 7                # itemset 最大長度（3-7 建議）
MAX_TRANSACTIONS = 1500000 # 最大交易數（None = 全部）

# 資料集配置
DATASETS = {
    "Retail": {
        "path": "data/retail.txt",
        "min_sup_ratio": 0.02,              # 最小支援度
        "sample_rates": [0.1, 0.2, 0.3, 0.4, 0.5],  # 抽樣比例
        "max_len": MAX_LEN,
        "max_transactions": MAX_TRANSACTIONS,
    },
    # ... 其他資料集
}
```

### 常用參數調整：

| 參數               | 說明             | 建議值                       |
| ------------------ | ---------------- | ---------------------------- |
| `min_sup_ratio`    | 最小支援度門檻   | 0.005 ~ 0.02                 |
| `sample_rates`     | 抽樣比例列表     | `[0.1, 0.2, 0.3, 0.4, 0.5]`  |
| `MAX_LEN`          | itemset 最大長度 | 3（快速）/ 7（完整）         |
| `MAX_TRANSACTIONS` | 限制交易數量     | 150 萬（平衡）/ None（完整） |

---

## 📊 如何解讀結果

### 圖表說明

每張圖包含 4 條線：

- **虛線（--）**：標準有放回抽樣
- **實線（─）**：系統抽樣（改進版）
- **藍色**：Non-Uni 1（基於交易長度）
- **紅色**：Non-Uni 2（基於高頻項目數）

### 評估標準

**越低越好的指標：**

1. **Non-common Output Ratio**

   - 衡量找到的 itemsets 與真實結果的差異
   - 0 = 完美匹配

2. **Support Error Rate**
   - 衡量 support 估計的準確度
   - 0 = 完美估計

### 結果解讀範例

```
如果看到：
✅ 實線在虛線下方 → 系統抽樣表現更好
⚠️ 實線在虛線上方 → 標準抽樣表現更好
≈ 兩線重疊 → 兩者效果相近
```

**關鍵發現：**

- 🏆 **Non-Uni 2 + 系統抽樣**：在兩個指標上都表現最佳
- ⚠️ **Non-Uni 1 + 系統抽樣**：Error Rate 改善，但 Output Ratio 可能較差
- 📊 詳細分析請參考 [RESULT_ANALYSIS.md](RESULT_ANALYSIS.md)

---

## 📁 專案結構

```
dm_final/
├── data/                              # 資料集
│   ├── retail.txt
│   ├── BMS1_itemset_mining.txt
│   ├── BMS2_itemset_mining.txt
│   └── chainstoreFIM.txt
├── src/                               # 原始碼
│   ├── sampling.py                    # 抽樣機率計算
│   ├── miner.py                       # Weighted Eclat 挖掘演算法
│   ├── metrics.py                     # 評估指標
│   ├── experiment.py                  # 原始實驗程式
│   └── experiment_systematic.py       # 🔥 系統抽樣實驗（主程式）
├── systematic_results_{MIN_SUP}/      # 實驗結果圖表
├── README.md                          # 📖 本文件（操作指南）
├── README_SYS.md                      # 📚 系統抽樣原理與技術細節
├── RESULT_ANALYSIS.md                 # 📊 實驗結果分析
└── requirements.txt                   # Python 套件依賴
```

---

## 🎯 核心貢獻

### 1. 實作系統抽樣方法

傳統方法使用**有放回隨機抽樣**，本專案實作了**系統抽樣**：

```python
# 系統抽樣核心概念
累積機率分布 → 固定間隔取樣 → 降低方差
```

### 2. 全面比較實驗

- ✅ 4 個標準資料集
- ✅ 2 種非均勻抽樣機率
- ✅ 2 種取樣方法（標準 vs 系統）
- ✅ 5 個抽樣比例
- ✅ 2 個評估指標

### 3. 發現最佳組合

**實驗結論：Non-Uni 2 + 系統抽樣**

- ✅ Support Error Rate 降低 50%+
- ✅ Non-common Output Ratio 保持穩定
- ✅ 計算成本幾乎相同

---

## 📖 延伸閱讀

### 想了解系統抽樣原理？

👉 閱讀 [README_SYS.md](README_SYS.md)

內容包含：

- 系統抽樣 vs 標準抽樣的詳細比較
- 數學原理與理論保證
- 實作技術細節
- 時間/空間複雜度分析

### 想了解實驗結果為何如此？

👉 閱讀 [RESULT_ANALYSIS.md](RESULT_ANALYSIS.md)

內容包含：

- 為什麼 Error Rate 改善？
- 為什麼 Output Ratio 不穩定？
- 為什麼 nonuni1_sys 表現較差？
- 為什麼 nonuni2_sys 是最佳組合？
- 方差-偏差權衡（Variance-Bias Tradeoff）分析

---

## 💻 系統需求

- **Python**: 3.8+
- **必要套件**: numpy, matplotlib, pandas
- **建議記憶體**: 8GB+（處理大型資料集）
- **建議硬碟空間**: 2GB+（儲存資料集與結果）

---

## 🔧 常見問題

### Q1: 執行時間很長怎麼辦？

**A:** 調整參數以加速測試：

```python
MAX_LEN = 3                # 降低 itemset 長度
MAX_TRANSACTIONS = 50000   # 限制交易數量
sample_rates = [0.1, 0.3]  # 減少測試的抽樣比例
```

### Q2: 記憶體不足怎麼辦？

**A:** 降低 `MAX_TRANSACTIONS`：

```python
MAX_TRANSACTIONS = 100000  # 或更小
```

### Q3: 想測試不同的 MinSup？

**A:** 修改 `DATASETS` 中的 `min_sup_ratio`：

```python
"min_sup_ratio": 0.005,  # 從 0.02 改為 0.005
```

結果會自動儲存到對應的資料夾（如 `systematic_results_0.005/`）

### Q4: 如何只測試單一資料集？

**A:** 在 `main()` 函數中註解掉不需要的資料集：

```python
DATASETS = {
    "Retail": { ... },
    # "BMS1": { ... },    # 註解掉
    # "BMS2": { ... },    # 註解掉
    # "Chainstore": { ... },  # 註解掉
}
```

---

## 📝 引用

如果本專案對您的研究有幫助，歡迎引用：

```
系統抽樣於頻繁項目集挖掘之研究
基於 Non-uniform Sampling 方法的改進與比較實驗
```

---

## 📧 聯絡資訊

如有問題或建議，歡迎透過 GitHub Issues 提出。

---

## 🙏 致謝

本專案基於以下研究：

- **原始論文**: Mining Frequent Itemsets using Non-uniform Sampling
- **系統抽樣理論**: Cochran, W. G. (1977). Sampling Techniques
- **Horvitz-Thompson Estimator**: 用於不等機率抽樣的無偏估計

---

**最後更新**: 2025-12-01
