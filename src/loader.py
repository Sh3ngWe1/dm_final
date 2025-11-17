def load_transactions(path, sep=" "):
    txns = []
    with open(path, "r") as f:
        for line in f:
            items = line.strip().split(sep)
            if items:
                txns.append(frozenset(items))
    return txns
