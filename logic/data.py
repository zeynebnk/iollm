import json
from pathlib import Path


class Dataset:
    def __init__(self, key, f, a):
        self.key = key
        self.f = f  ## f_i = foreign tokens
        self.a = a  ## a_i = atom set


## sentences -> pairs (f_i, a_i), tokens -> f, atoms -> a
def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset(
        key=data.get("dataset_key", Path(path).stem),
        f=[s["tokens"] for s in data["sentences"]],
        a=[set(s["atoms"]) for s in data["sentences"]],
    )


def save(dataset, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_key": dataset.key,
        "sentences": [
            {"tokens": toks, "atoms": sorted(atoms)}
            for toks, atoms in zip(dataset.f, dataset.a)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
