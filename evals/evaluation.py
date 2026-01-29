import asyncio
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from sacrebleu.metrics import CHRF

from .api import (
    async_client, pid, parse_json, save_json, load_json,
    load_problems, DATA_DIR, RESULTS_DIR, IOLLM_DATASET
)
from .prompts import GRADER

QUERIES_FILE = DATA_DIR / "extracted_queries.json"

chrf = CHRF()

_queries = None
def get_queries():
    global _queries
    if _queries is None:
        _queries = load_json(QUERIES_FILE) if QUERIES_FILE.exists() else {}
    return _queries


def load_solutions(dataset=IOLLM_DATASET):
    return {pid(p): dict(p) for p in load_problems(dataset)}


def load_results(path):
    data = load_json(path)
    return data.get("results", data) if isinstance(data, dict) else data


def extract_answer(text):
    if not text:
        return ""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return "\n\n".join(m.strip() for m in matches) if matches else text


def compute_chrf(pairs):
    refs, hyps = [], []
    for p in (pairs or {}).values():
        if p.get("expected") and p.get("predicted"):
            refs.append(p["expected"])
            hyps.append(p["predicted"])
    return chrf.corpus_score(hyps, [refs]).score if refs else 0.0


def fix_scores(scores, pairs):
    if not scores or not pairs:
        return scores

    fixed = dict(scores)
    for item_id, score in scores.items():
        pair = pairs.get(item_id, {})
        exp = str(pair.get("expected", "")).strip()
        pred = str(pair.get("predicted", "")).strip()
        if not exp or not pred:
            continue

        exp_n = unicodedata.normalize('NFKC', exp.lower())
        pred_n = unicodedata.normalize('NFKC', pred.lower())

        # false negative: exact or near-exact match marked wrong
        if score == 0 and (exp_n == pred_n or (exp_n in pred_n and len(exp_n) > 5)):
            fixed[item_id] = 1
            continue

        # false positive checks
        if score == 1 and exp_n != pred_n:
            # ascii vs non-ascii mismatch (likely wrong script)
            exp_ascii = exp.replace(" ", "").replace(".", "").replace(",", "").isascii()
            pred_ascii = pred.replace(" ", "").replace(".", "").replace(",", "").isascii()
            if len(exp) > 5 and len(pred) > 5 and exp_ascii != pred_ascii:
                fixed[item_id] = 0
                continue

            # no word overlap at all
            exp_words, pred_words = set(exp_n.split()), set(pred_n.split())
            if len(exp_words) >= 2 and len(pred_words) >= 2 and not (exp_words & pred_words):
                fixed[item_id] = 0
                continue

            # different leading numbers
            exp_nums, pred_nums = re.findall(r'\d+', exp), re.findall(r'\d+', pred)
            if exp_nums and pred_nums and len(exp_nums[0]) > 1 and exp_nums[0] != pred_nums[0]:
                fixed[item_id] = 0
                continue

            # tiny prediction for long expected
            if len(pred) <= 3 and len(exp) > 10:
                fixed[item_id] = 0

    return fixed


def majority_vote(score_lists):
    if not score_lists:
        return {}
    all_keys = set()
    for scores in score_lists:
        all_keys.update(scores.keys())
    
    voted = {}
    for key in all_keys:
        votes = [s.get(key) for s in score_lists if key in s]
        if votes:
            voted[key] = Counter(votes).most_common(1)[0][0]
    return voted


async def grade_once(prompt, grader):
    kwargs = {"model": grader, "messages": [{"role": "user", "content": prompt}]}
    if not grader.startswith("o"):
        kwargs["temperature"] = 0.7  # need variance for multiple runs
    r = await async_client.chat.completions.create(**kwargs)
    return parse_json(r.choices[0].message.content)


async def grade(result, problem, grader, queries=None, k=1):
    problem_id = result["problem_id"]

    solution = problem.get("solution", "")
    if not solution:
        return {"problem_id": problem_id, "scores": {}, "n_correct": 0, "n_total": 0, "chrf": 0.0, "error": "no solution"}

    if queries is None:
        queries = get_queries()
    items = queries.get(problem_id, {})
    items_text = json.dumps(items, indent=2, ensure_ascii=False) if items else "(see solution below)"
    prompt = GRADER.format(items=items_text, solution=solution, answer=extract_answer(result.get("response", "")))

    try:
        if k == 1:
            # single run with temperature=0
            kwargs = {"model": grader, "messages": [{"role": "user", "content": prompt}]}
            if not grader.startswith("o"):
                kwargs["temperature"] = 0
            r = await async_client.chat.completions.create(**kwargs)
            data = parse_json(r.choices[0].message.content)
            scores = data.get("scores", data)
            pairs = data.get("answer_pairs", {})
        else:
            # k runs with majority vote
            runs = await asyncio.gather(*[grade_once(prompt, grader) for _ in range(k)])
            score_lists = [fix_scores(r.get("scores", r), r.get("answer_pairs", {})) for r in runs]
            scores = majority_vote(score_lists)
            # use pairs from first successful run
            pairs = next((r.get("answer_pairs", {}) for r in runs if r.get("answer_pairs")), {})

        scores = fix_scores(scores, pairs)
        return {
            "problem_id": problem_id,
            "scores": scores,
            "answer_pairs": pairs,
            "n_correct": sum(1 for v in scores.values() if v == 1),
            "n_total": len(scores),
            "chrf": compute_chrf(pairs),
            "k": k,
            "error": None
        }
    except Exception as e:
        return {"problem_id": problem_id, "scores": {}, "n_correct": 0, "n_total": 0, "chrf": 0.0, "error": str(e)}


async def grade_all(results, problems, grader, workers, queries=None, k=1):
    sem = asyncio.Semaphore(workers)

    async def go(r, i):
        async with sem:
            prob = problems.get(r["problem_id"])
            if not prob:
                return {"problem_id": r["problem_id"], "error": "not found"}
            out = await grade(r, prob, grader, queries, k)
            print(f"[{i+1}/{len(results)}] {r['problem_id']}")
            return out

    return await asyncio.gather(*[go(r, i) for i, r in enumerate(results)])


def evaluate(results_path, grader="gpt-5-mini", dataset=IOLLM_DATASET,
             output=None, workers=10, use_queries=True, k=1):
    results = [r for r in load_results(results_path) if not r.get("error")]
    problems = load_solutions(dataset)
    queries = get_queries() if use_queries else None

    if not results:
        print("nothing to evaluate")
        return

    k_str = f" (k={k} majority vote)" if k > 1 else ""
    print(f"grading {len(results)} results with {grader}{k_str}")
    evals = asyncio.run(grade_all(results, problems, grader, workers, queries, k))
    ok = [e for e in evals if not e.get("error")]

    for e in evals:
        if not e.get("error"):
            e["score_20"] = round(20 * e["n_correct"] / e["n_total"], 2) if e["n_total"] else 0
            e["full_credit"] = e["n_correct"] == e["n_total"] and e["n_total"] > 0

    # summary
    correct = sum(e["n_correct"] for e in ok)
    total = sum(e["n_total"] for e in ok)
    avg_chrf = sum(e["chrf"] for e in ok) / len(ok) if ok else 0
    print(f"\n{correct}/{total} correct ({correct/total*100:.0f}%)" if total else "\nno items graded")
    print(f"avg chrf: {avg_chrf:.1f}")

    summary = {
        "grader": grader,
        "k": k,
        "correct": correct,
        "total": total,
        "accuracy": round(correct/total*100, 1) if total else 0,
        "chrf": round(avg_chrf, 1),
        "avg_score_20": round(sum(e.get("score_20", 0) for e in ok) / len(ok), 2) if ok else 0,
        "full_problem_rate": round(100 * sum(1 for e in ok if e.get("full_credit")) / len(ok), 1) if ok else 0,
        "n_graded": len(ok),
        "n_failed": len(evals) - len(ok)
    }

    out = {"summary": summary, "results": evals}
    if output:
        save_json(out, output)
        print(f"saved: {output}")
    return out


def merge_results(inference_path, eval_path, output=None):
    inf = load_json(inference_path)
    ev = load_json(eval_path)
    inf_results = inf.get("results", inf) if isinstance(inf, dict) else inf
    ev_results = ev.get("results", ev) if isinstance(ev, dict) else ev
    ev_by_id = {e["problem_id"]: e for e in ev_results}

    merged = [{**r, "evaluation": ev_by_id.get(r["problem_id"])} for r in inf_results]
    out = {
        "metadata": {
            **(inf.get("metadata", {}) if isinstance(inf, dict) else {}),
            "evaluation": ev.get("summary", {}) if isinstance(ev, dict) else {}
        },
        "results": merged
    }
    if output:
        save_json(out, output)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--grader", default="gpt-5-mini")
    p.add_argument("--dataset", "-d", default=IOLLM_DATASET)
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--no-queries", action="store_true")
    p.add_argument("-k", type=int, default=1, help="grade k times and take majority vote")
    a = p.parse_args()

    output = a.output or (RESULTS_DIR / "evaluations" / f"{a.results.stem}_eval.json")
    evaluate(a.results, a.grader, a.dataset, output, a.workers, not a.no_queries, a.k)

# experiments:
#   representation: eval vs encoded solutions (--dataset username/iollm-encoded)
#   contamination: eval vs original solutions (--dataset username/iollm)
