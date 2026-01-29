import numpy as np
import pandas as pd

from . import entailment, entropy, constraints
from .data import Dataset


class Result:
    def __init__(self, tokens, curve, summary):
        self.tokens = tokens
        self.curve = curve
        self.summary = summary


def analyze(dataset, n_samples=200):
    f = dataset.f
    a = dataset.a

    c, idx = entailment.compute_c(f, a)

    rows = []
    for t, ctx in c.items():
        c_t = entailment.intersect(ctx)
        u_t = entailment.union(ctx)
        rows.append({
            "t": t,
            "n": len(ctx),
            "|c|": len(c_t),
            "|u|": len(u_t),
            "id": len(c_t) == 1,
            "c": ", ".join(sorted(c_t)),
            "u": ", ".join(sorted(u_t)),
        })

    df = pd.DataFrame(rows).sort_values(["|c|", "n", "t"]).reset_index(drop=True)

    cert_k, cert_s, atom = {}, {}, {}
    for t, ctx in c.items():
        c_t = entailment.intersect(ctx)
        if len(c_t) == 1:
            at = next(iter(c_t))
            k, s = entailment.certificate(ctx, at)
            atom[t] = at
            cert_k[t] = k
            if s:
                cert_s[t] = tuple(sorted(idx[t][j] + 1 for j in s))

    df["atom"] = df["t"].map(atom)
    df["cert_k"] = df["t"].map(cert_k)
    df["cert_s"] = df["t"].map(cert_s)

    curve, auc = entropy.ambiguity_curve(f, a, idx, n_samples=n_samples)
    g = constraints.graph_stats(f)

    n_v = len(c)
    n_id = df["id"].sum()
    n_empty = (df["|c|"] == 0).sum()
    all_a = set.union(*a) if a else set()

    summary = {
        "key": dataset.key,
        "m": len(f),
        "v": n_v,
        "a": len(all_a),
        "id%": 100 * n_id / n_v if n_v else 0,
        "empty%": 100 * n_empty / n_v if n_v else 0,
        "e|c|": df["|c|"].mean() if n_v else 0,
        "e_cert": float(np.nanmean(df["cert_k"])) if n_id else float("nan"),
        "auc": auc,
        **g,
    }

    return Result(df, curve, summary)
