# metrics.py
"""
實驗評估指標：
- non_common_output_ratio
- support_error_rate
"""

from typing import Dict, Tuple, Set


Itemset = Tuple[str, ...]


def compute_non_common_output_ratio(
    exact_freq: Dict[Itemset, int],
    approx_freq: Dict[Itemset, float],
) -> float:
    """
    non-common output ratio（教學版定義）：
        (|L_s \ L| + |L \ L_s|) / |L|
    其中：
        L   = 真實 frequent itemsets 集合
        L_s = 抽樣挖掘出的 frequent itemsets 集合

    回傳：值越小代表越接近真實集合。
    """
    L: Set[Itemset] = set(exact_freq.keys())
    Ls: Set[Itemset] = set(approx_freq.keys())

    if len(L) == 0:
        return float("nan")

    extra = len(Ls - L)   # 假陽性
    missing = len(L - Ls) # 假陰性

    return (extra + missing) / len(L)


def compute_support_error_rate(
    exact_freq: Dict[Itemset, int],
    estimated_support: Dict[Itemset, float],
) -> float:
    """
    support error rate（教學版定義）：
        在共同 itemsets 上的平均相對誤差：
            mean_{I ∈ L ∩ L_s} |sup_hat(I) - sup(I)| / sup(I)

    exact_freq       : ground truth support（完整 DB）
    estimated_support: 抽樣後估計的 support（所有 itemset）
    """
    L: Set[Itemset] = set(exact_freq.keys())
    Ls: Set[Itemset] = set(estimated_support.keys())
    common = L & Ls

    if not common:
        return float("nan")

    errors = []
    for I in common:
        true_sup = exact_freq[I]
        est_sup = estimated_support[I]
        if true_sup > 0:
            errors.append(abs(est_sup - true_sup) / true_sup)

    if not errors:
        return float("nan")

    return sum(errors) / len(errors)
