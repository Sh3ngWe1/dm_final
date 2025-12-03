from collections import defaultdict
from typing import List, FrozenSet


Transaction = FrozenSet[str]


def compute_probs_uniform(transactions: List[Transaction]) -> List[float]:
    n = len(transactions)
    if n == 0:
        return []
    return [1.0 / n] * n


def compute_probs_nonuni1(transactions: List[Transaction]) -> List[float]:
    lens = [len(t) for t in transactions]
    total = sum(lens)
    if total <= 0:
        return compute_probs_uniform(transactions)
    return [L / total for L in lens]


def compute_probs_nonuni2(
    transactions: List[Transaction],
    min_sup_abs: int,
) -> List[float]:
    item_cnt = defaultdict(int)
    for t in transactions:
        for it in t:
            item_cnt[it] += 1

    freq_items = {it for it, c in item_cnt.items() if c >= min_sup_abs}
    counts = [sum(1 for it in t if it in freq_items) for t in transactions]
    total = sum(counts)
    if total <= 0:
        return compute_probs_uniform(transactions)

    return [c / total for c in counts]


def get_sampling_probs(
    transactions: List[Transaction],
    sampling: str,
    min_sup_abs: int,
) -> List[float]:
    sampling = sampling.lower()
    if sampling == "uniform":
        return compute_probs_uniform(transactions)
    elif sampling == "nonuni1":
        return compute_probs_nonuni1(transactions)
    elif sampling == "nonuni2":
        return compute_probs_nonuni2(transactions, min_sup_abs=min_sup_abs)
    else:
        raise ValueError("sampling must be 'uniform', 'nonuni1', or 'nonuni2'")
