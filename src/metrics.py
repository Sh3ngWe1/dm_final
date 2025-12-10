from typing import Dict, Tuple, Set

Itemset = Tuple[str, ...]

def compute_non_common_output_ratio(
    exact_freq: Dict[Itemset, int],
    approx_freq: Dict[Itemset, float],
) -> float:
    L: Set[Itemset] = set(exact_freq.keys())
    Ls: Set[Itemset] = set(approx_freq.keys())

    if len(L) == 0:
        return float("nan")

    extra = len(Ls - L)   # false positive
    missing = len(L - Ls) # false negative

    return (extra + missing) / len(L)


def compute_support_error_rate(
    exact_freq: Dict[Itemset, int],
    estimated_support: Dict[Itemset, float],
) -> float:
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
