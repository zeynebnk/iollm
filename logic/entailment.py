from collections import defaultdict
import itertools


## c(t) = \cap_{i : t \in f_i} a_i
def compute_c(f, a):
    c = defaultdict(list)
    idx = defaultdict(list)
    for i, (tokens, atoms) in enumerate(zip(f, a)):
        for t in set(tokens):
            c[t].append(atoms)
            idx[t].append(i)
    return dict(c), dict(idx)


def intersect(sets):
    if not sets:
        return set()
    return set.intersection(*sets)


def union(sets):
    if not sets:
        return set()
    return set.union(*sets)


## min |s| : \cap_{i \in s} ctx_i = {target}
def exact_cert(contexts, target):
    n = len(contexts)
    for i, s in enumerate(contexts):
        if s == {target}:
            return 1, (i,)
    for k in range(2, n + 1):
        for s in itertools.combinations(range(n), k):
            if set.intersection(*[contexts[j] for j in s]) == {target}:
                return k, s
    return None, None


def greedy_cert(contexts, target):
    remaining = list(range(len(contexts)))
    current = set(contexts[remaining[0]])
    s = [remaining.pop(0)]
    while current != {target} and remaining:
        j = min(remaining, key=lambda j: len(current & contexts[j]))
        s.append(j)
        current &= contexts[j]
        remaining.remove(j)
    if current == {target}:
        return len(s), tuple(s)
    return None, None


def certificate(contexts, target, threshold=12):
    if len(contexts) <= threshold:
        return exact_cert(contexts, target)
    return greedy_cert(contexts, target)
