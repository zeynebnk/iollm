from collections import Counter


## sum_{t in f_i} meaning(t) = target_i
def count_models(subset, tokens, f, target, h, limit=None):
    subset = tuple(sorted(subset))

    for t in tokens:
        if not h.get(t):
            return 0

    token_counts = [Counter(toks) for toks in f]
    constrained = sorted({t for i in subset for t in f[i]})
    unconstrained = [t for t in tokens if t not in constrained]

    factor = 1
    for t in unconstrained:
        factor *= len(h[t])

    residuals = [Counter(target[i]) for i in subset]
    counts = [token_counts[i] for i in subset]

    max_per = {t: max(sum(opt.values()) for opt in h[t]) for t in tokens}
    remaining_max = [sum(max_per[t] * cnt for t, cnt in c.items()) for c in counts]

    token_to_constraints = {t: [] for t in constrained}
    for local_idx, i in enumerate(subset):
        for t in f[i]:
            token_to_constraints[t].append(local_idx)

    order = sorted(constrained, key=lambda t: (len(h[t]), -len(token_to_constraints[t]), t))
    sol = [0]

    def backtrack(pos=0):
        if limit and sol[0] * factor >= limit:
            return
        if pos == len(order):
            if all(sum(r.values()) == 0 for r in residuals):
                sol[0] += 1
            return

        t = order[pos]
        occs = [c.get(t, 0) for c in counts]
        max_t = max_per[t]

        for opt in h[t]:
            ok = True
            changed = []

            for si, cnt in enumerate(occs):
                if cnt == 0:
                    continue
                for feat, v in opt.items():
                    residuals[si][feat] -= v * cnt
                changed.append(si)
                if any(v < 0 for v in residuals[si].values()):
                    ok = False
                    break

            old_remaining = None
            if ok:
                old_remaining = remaining_max.copy()
                for si, cnt in enumerate(occs):
                    if cnt > 0:
                        remaining_max[si] -= max_t * cnt
                for si in range(len(residuals)):
                    if sum(residuals[si].values()) > remaining_max[si]:
                        ok = False
                        break

            if ok:
                backtrack(pos + 1)

            if old_remaining:
                remaining_max[:] = old_remaining
            for si in changed:
                cnt = occs[si]
                for feat, v in opt.items():
                    residuals[si][feat] += v * cnt
                for key in list(residuals[si].keys()):
                    if residuals[si][key] == 0:
                        del residuals[si][key]

            if limit and sol[0] * factor >= limit:
                return

    backtrack()
    return sol[0] * factor


def satisfiable(subset, tokens, f, target, h):
    return count_models(subset, tokens, f, target, h, limit=1) >= 1


def leave_one_out(tokens, f, target, h, limit=1000):
    m = len(f)
    full = count_models(range(m), tokens, f, target, h, limit=limit)
    loo = []
    for i in range(m):
        subset = [j for j in range(m) if j != i]
        loo.append(count_models(subset, tokens, f, target, h, limit=limit))
    return {"full": full, "loo": loo}
