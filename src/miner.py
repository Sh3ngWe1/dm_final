from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, FrozenSet, Iterable, Set
import random

from sampling import get_sampling_probs


Transaction = FrozenSet[str]
Itemset = Tuple[str, ...]


def _eclat_mine(
    transactions: List[Transaction],
    weights: List[float],
    min_sup: float,
    max_len: int | None = None,
) -> Dict[Itemset, float]:
    n = len(transactions)
    assert n == len(weights)
    if n == 0:
        return {}

    if max_len is None:
        max_len = max(len(t) for t in transactions) if transactions else 0

    item_to_tids: Dict[str, Set[int]] = defaultdict(set)
    for tid, T in enumerate(transactions):
        for it in T:
            item_to_tids[it].add(tid)

    def sup_from_tids(tids: Iterable[int]) -> float:
        return sum(weights[tid] for tid in tids)

    exts: List[Tuple[str, Set[int]]] = []
    for it in sorted(item_to_tids.keys()):
        tids = item_to_tids[it]
        sup = sup_from_tids(tids)
        if sup >= min_sup:
            exts.append((it, tids))

    results: Dict[Itemset, float] = {}

    def recurse(prefix: Itemset, exts_local: List[Tuple[str, Set[int]]]):
        for i, (it, tids_it) in enumerate(exts_local):
            new_prefix = tuple(sorted(prefix + (it,)))
            sup = sup_from_tids(tids_it)
            if sup >= min_sup:
                results[new_prefix] = sup

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

    recurse((), exts)
    return results


def brute_force_frequent_itemsets(
    transactions: List[Transaction],
    min_sup_abs: int,
    max_len: int | None = None,
) -> Dict[Itemset, float]:
    if not transactions:
        return {}

    if max_len is None:
        max_len = max(len(t) for t in transactions)

    weights = [1.0] * len(transactions)
    return _eclat_mine(transactions, weights, min_sup=min_sup_abs, max_len=max_len)


def approximate_itemset_miner(
    transactions: List[Transaction],
    min_sup_abs: int,
    k: int,
    sampling: str = "uniform",
    max_len: int | None = None,
    random_seed: int | None = None,
):
    if random_seed is not None:
        random.seed(random_seed)

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

    indices = list(range(n))
    sampled_indices = random.choices(indices, weights=probs, k=k)

    sample_txns: List[Transaction] = []
    weights: List[float] = []
    for idx in sampled_indices:
        T = transactions[idx]
        pT = probs[idx]
        if pT <= 0.0:
            continue
        sample_txns.append(T)
        weights.append(1.0 / (pT * k))

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
