# sampling.py
"""
定義三種抽樣機率 p(T)：
- uniform  : 每筆交易同樣機率
- nonuni1  : p(T) ∝ |T|（交易長度）
- nonuni2  : p(T) ∝ T 中 frequent items 的個數
"""

from collections import defaultdict
from typing import List, FrozenSet


Transaction = FrozenSet[str]


def compute_probs_uniform(transactions: List[Transaction]) -> List[float]:
    """Uniform：每筆交易機率相同。"""
    n = len(transactions)
    if n == 0:
        return []
    return [1.0 / n] * n


def compute_probs_nonuni1(transactions: List[Transaction]) -> List[float]:
    """
    non-uni1：p(T) ∝ |T|
    交易越長，被抽到的機率越高。
    """
    lens = [len(t) for t in transactions]
    total = sum(lens)
    if total <= 0:
        # 防呆：全部空交易就退回 uniform
        return compute_probs_uniform(transactions)
    return [L / total for L in lens]


def compute_probs_nonuni2(
    transactions: List[Transaction],
    min_sup_abs: int,
) -> List[float]:
    """
    non-uni2：先找 frequent items（單一 item），
    再令 p(T) ∝ T 中 frequent items 的個數。

    min_sup_abs: 最小支援度（絕對筆數）
    """
    # 先統計每個 item 出現次數
    item_cnt = defaultdict(int)
    for t in transactions:
        for it in t:
            item_cnt[it] += 1

    # frequent items（單一 item）
    freq_items = {it for it, c in item_cnt.items() if c >= min_sup_abs}

    # 每筆交易中，frequent items 的數量
    counts = [sum(1 for it in t if it in freq_items) for t in transactions]
    total = sum(counts)
    if total <= 0:
        # 如果完全沒有 frequent item，退回 uniform
        return compute_probs_uniform(transactions)

    return [c / total for c in counts]


def get_sampling_probs(
    transactions: List[Transaction],
    sampling: str,
    min_sup_abs: int,
) -> List[float]:
    """
    封裝好的 helper：
    根據 sampling 參數回傳對應的 p(T) 列表。
    sampling: "uniform" / "nonuni1" / "nonuni2"
    """
    sampling = sampling.lower()
    if sampling == "uniform":
        return compute_probs_uniform(transactions)
    elif sampling == "nonuni1":
        return compute_probs_nonuni1(transactions)
    elif sampling == "nonuni2":
        return compute_probs_nonuni2(transactions, min_sup_abs=min_sup_abs)
    else:
        raise ValueError("sampling must be 'uniform', 'nonuni1', or 'nonuni2'")
