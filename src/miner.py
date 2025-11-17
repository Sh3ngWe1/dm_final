# # miner.py
# """
# 提供兩種挖掘方式：
# - brute_force_frequent_itemsets：在完整 DB 上暴力挖 frequent itemsets（ground truth）
# - approximate_itemset_miner    ：實作論文 Algorithm 1（抽樣 + 加權 support）
# """

# from collections import defaultdict
# import itertools
# import random
# from typing import Dict, List, Tuple, FrozenSet

# from sampling import get_sampling_probs  # 如果檔案同層就用 from sampling import ...


# Transaction = FrozenSet[str]
# Itemset = Tuple[str, ...]


# # ---------- 完整 DB：暴力 frequent itemset mining（當作真實值） ----------

# def brute_force_frequent_itemsets(
#     transactions: List[Transaction],
#     min_sup_abs: int,
#     max_len: int | None = None,
# ) -> Dict[Itemset, int]:
#     """
#     在完整 DB 上暴力列舉所有 itemset，計算真實 support。
#     回傳：{itemset(tuple): support_count}
#     """
#     if not transactions:
#         return {}

#     if max_len is None:
#         max_len = max(len(t) for t in transactions)

#     itemsets_support: Dict[Itemset, int] = defaultdict(int)

#     # 所有 item
#     all_items = sorted({it for t in transactions for it in t})

#     # 逐層增加長度 1..max_len
#     for L in range(1, max_len + 1):
#         for comb in itertools.combinations(all_items, L):
#             comb_set = frozenset(comb)
#             cnt = sum(1 for t in transactions if comb_set.issubset(t))
#             if cnt >= min_sup_abs:
#                 itemsets_support[comb] = cnt

#     return itemsets_support


# # ---------- 抽樣版：論文 Algorithm 1 ----------

# def approximate_itemset_miner(
#     transactions: List[Transaction],
#     min_sup_abs: int,
#     k: int,
#     sampling: str = "uniform",
#     max_len: int | None = None,
#     random_seed: int | None = None,
# ):
#     """
#     根據論文 Algorithm 1：
#     1. 依 sampling 方法計算每筆交易的 p(T)
#     2. 依 p(T) 進行有放回抽樣 k 次
#     3. 對樣本中的 itemset 計算加權 support：
#        每次出現加 1/p(T)，最後除以 k
#     4. 回傳：
#        - approx_frequent: {itemset: estimated_support}，已過 min_sup_abs 門檻
#        - estimated_support: {itemset: estimated_support}，所有有計算到的 itemset
#     """
#     if random_seed is not None:
#         random.seed(random_seed)

#     n = len(transactions)
#     if n == 0 or k <= 0:
#         return {}, {}

#     if max_len is None:
#         max_len = max(len(t) for t in transactions)

#     # 1. 計算各交易 p(T)
#     probs = get_sampling_probs(transactions, sampling=sampling, min_sup_abs=min_sup_abs)

#     # 2. 依 p(T) 抽樣（with replacement）
#     indices = list(range(n))
#     sampled_indices = random.choices(indices, weights=probs, k=k)

#     # 3. 加權 support：每次看到 itemset I 在 T 中出現，就加 1/p(T)
#     weighted_support: Dict[Itemset, float] = defaultdict(float)

#     for idx in sampled_indices:
#         T = transactions[idx]
#         pT = probs[idx]
#         if pT <= 0.0:
#             # 理論上不會抽到 p(T)=0 的交易；保險檢查
#             continue

#         weight = 1.0 / pT

#         # 列舉 T 的所有子集合（1..max_len）
#         for L in range(1, min(len(T), max_len) + 1):
#             for comb in itertools.combinations(T, L):
#                 itemset = tuple(sorted(comb))
#                 weighted_support[itemset] += weight

#     # 4. 除以 k 得到估計 support
#     estimated_support: Dict[Itemset, float] = {
#         I: ws / k for I, ws in weighted_support.items()
#     }

#     # 5. 篩出 estimated_support >= min_sup_abs 當作 frequent
#     approx_frequent: Dict[Itemset, float] = {
#         I: sup for I, sup in estimated_support.items() if sup >= min_sup_abs
#     }

#     return approx_frequent, estimated_support




# miner.py
"""
挖 frequent itemsets 的核心模組（論文版）

- 使用 Eclat 演算法當作「exact large itemset mining」工具
- exact 部分：在完整 DB 上跑 Eclat（權重全部 = 1）
- approximate 部分：依 Algorithm 1 抽樣並加上權重，再用 Eclat
  * 每筆被抽到的交易權重 = 1 / (p(T) * k)
  * Eclat 的 support = sum(權重)，直接就是估計的 sup(DB, I)

提供兩個主要函式：
- brute_force_frequent_itemsets(transactions, min_sup_abs, max_len)
    → 其實是 Eclat，但沿用舊名字，當 ground truth 用
- approximate_itemset_miner(transactions, min_sup_abs, k, sampling, max_len, random_seed)
    → 論文的 sampling + Eclat 版
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, FrozenSet, Iterable, Set
import random

from sampling import get_sampling_probs  # 和之前一樣，從 sampling.py 引入


Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]


# ---------- Eclat 核心：支援「權重 support」 ----------

def _eclat_mine(
    transactions: List[Transaction],
    weights: List[float],
    min_sup: float,
    max_len: int | None = None,
) -> Dict[Itemset, float]:
    """
    Weighted Eclat：

    transactions : list of frozenset(items)
    weights     : 每筆交易的權重（同長度）
    min_sup     : support 門檻（用「加總權重」判斷）
    max_len     : itemset 最大長度限制（避免組合爆炸）

    回傳：
        { itemset(tuple): support(加總權重) }
        只包含 support >= min_sup 的 frequent itemsets
    """
    n = len(transactions)
    assert n == len(weights)
    if n == 0:
        return {}

    if max_len is None:
        max_len = max(len(t) for t in transactions) if transactions else 0

    # 垂直格式：item -> set(tid)
    item_to_tids: Dict[str, Set[int]] = defaultdict(set)
    for tid, T in enumerate(transactions):
        for it in T:
            item_to_tids[it].add(tid)

    # 單一 item 的 frequent 清單 (L1)
    def sup_from_tids(tids: Iterable[int]) -> float:
        return sum(weights[tid] for tid in tids)

    exts: List[Tuple[str, Set[int]]] = []
    for it in sorted(item_to_tids.keys()):
        tids = item_to_tids[it]
        sup = sup_from_tids(tids)
        if sup >= min_sup:
            exts.append((it, tids))

    results: Dict[Itemset, float] = {}

    # Eclat 遞迴
    def recurse(prefix: Itemset, exts_local: List[Tuple[str, Set[int]]]):
        for i, (it, tids_it) in enumerate(exts_local):
            new_prefix = tuple(sorted(prefix + (it,)))
            sup = sup_from_tids(tids_it)
            if sup >= min_sup:
                results[new_prefix] = sup

                # 若還沒到 max_len，就繼續延伸
                if len(new_prefix) < max_len:
                    new_exts: List[Tuple[str, Set[int]]] = []
                    for j in range(i + 1, len(exts_local)):
                        it2, tids2 = exts_local[j]
                        inter_tids = tids_it & tids2
                        if inter_tids:
                            sup2 = sup_from_tids(inter_tids)
                            if sup2 >= min_sup:
                                new_exts.append((it2, inter_tids))
                    if new_exts:
                        recurse(new_prefix, new_exts)

    # 從空 prefix 開始
    recurse((), exts)
    return results


# ---------- exact：完整 DB 上的 frequent itemsets（Eclat 版） ----------

def brute_force_frequent_itemsets(
    transactions: List[Transaction],
    min_sup_abs: int,
    max_len: int | None = None,
) -> Dict[Itemset, float]:
    """
    exact frequent itemset mining（論文用 Eclat，我們這裡實作 weighted Eclat，權重全為 1）。

    參數：
        transactions : 完整 DB
        min_sup_abs  : 最小支援度（筆數門檻）
        max_len      : itemset 最大長度

    回傳：
        { itemset(tuple): support_count(= 權重加總, 這裡就是實際筆數) }
    """
    if not transactions:
        return {}

    if max_len is None:
        max_len = max(len(t) for t in transactions)

    # 權重都設 1，就變成一般 Eclat
    weights = [1.0] * len(transactions)
    return _eclat_mine(transactions, weights, min_sup=min_sup_abs, max_len=max_len)


# ---------- approximate：Algorithm 1 + Eclat ----------

def approximate_itemset_miner(
    transactions: List[Transaction],
    min_sup_abs: int,
    k: int,
    sampling: str = "uniform",
    max_len: int | None = None,
    random_seed: int | None = None,
):
    """
    根據論文 Algorithm 1 + Eclat：

    1. 依 sampling (uniform / nonuni1 / nonuni2) 計算每筆交易的 p(T)
    2. 依 p(T) 抽樣 k 次（有放回）
    3. 對於被抽到的第 j 筆交易 T_j，給它權重 w_j = 1 / (p(T_j) * k)
       => 對任一 itemset I，Eclat 算出來的 support = sum_j w_j * χ(T_j, I)
          就是論文中的估計值 (1/k) Σ χ / p(T)
    4. 在加權樣本上跑 Eclat，門檻仍然是 min_sup_abs
    5. 回傳：
        approx_frequent   : {I: sup_hat(I)}，sup_hat 已是估計 DB 支援度
        estimated_support : 同上（為了和舊介面相容）
    """
    if random_seed is not None:
        random.seed(random_seed)

    n = len(transactions)
    if n == 0 or k <= 0:
        return {}, {}

    if max_len is None:
        max_len = max(len(t) for t in transactions)

    # 1. 計算 p(T)
    probs = get_sampling_probs(
        transactions=transactions,
        sampling=sampling,
        min_sup_abs=min_sup_abs,
    )

    # 2. 抽樣 k 次
    indices = list(range(n))
    sampled_indices = random.choices(indices, weights=probs, k=k)

    # 3. 構造「加權樣本」
    sample_txns: List[Transaction] = []
    weights: List[float] = []
    for idx in sampled_indices:
        T = transactions[idx]
        pT = probs[idx]
        if pT <= 0.0:
            # 理論上不會抽到 p(T)=0 的交易；這裡只是保險
            continue
        sample_txns.append(T)
        # 權重 = 1 / (p(T) * k)  → support 直接是估計 sup(DB, I)
        weights.append(1.0 / (pT * k))

    if not sample_txns:
        return {}, {}

    # 4. 在加權樣本上跑 Eclat
    est_support = _eclat_mine(
        transactions=sample_txns,
        weights=weights,
        min_sup=min_sup_abs,
        max_len=max_len,
    )

    # 5. est_support 已經只包含 sup_hat >= min_sup_abs 的 itemsets
    approx_frequent = dict(est_support)  # 一樣的 dict，為了語意清楚
    return approx_frequent, est_support
