import itertools
import math
import random
import numpy as np
import pandas as pd


## e[log_2 |c(t)|] over k-subsets
def ambiguity_curve(f, a, idx, n_samples=200, seed=0):
    rng = random.Random(seed)
    m = len(f)
    tokens = list(idx.keys())

    def c_size(t, s):
        ii = [i for i in idx[t] if i in s]
        if not ii:
            return None
        r = set(a[ii[0]])
        for i in ii[1:]:
            r &= a[i]
        return len(r)

    rows = []
    for k in range(1, m + 1):
        n_total = math.comb(m, k)
        if n_total <= n_samples:
            subsets = list(itertools.combinations(range(m), k))
        else:
            subsets = [tuple(sorted(rng.sample(range(m), k))) for _ in range(n_samples)]

        h = []
        for s in subsets:
            ss = set(s)
            sizes = [max(1, c) for t in tokens if (c := c_size(t, ss)) is not None]
            if sizes:
                h.append(np.mean([math.log2(x) for x in sizes]))

        rows.append({"k": k, "mean_log2_c": np.mean(h) if h else float("nan"), "n": len(h)})

    df = pd.DataFrame(rows)
    auc = float(np.nanmean(df["mean_log2_c"]))
    return df, auc
