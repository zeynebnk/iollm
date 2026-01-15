"""grading"""
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
from sacrebleu.metrics import CHRF

load_dotenv()

try:
    from .api import client, async_client, save_json, load_json, load_problems, QUERIES, IOLLM_DATASET
    from .prompts import GRADER, RULES_GRADER
except ImportError:
    from api import client, async_client, save_json, load_json, load_problems, QUERIES, IOLLM_DATASET
    from prompts import GRADER, RULES_GRADER

RESULTS_DIR = Path(__file__).parent.parent / "results"

chrf = CHRF()

_queries = None
def get_queries():
    global _queries
    if _queries is None:
        _queries = load_json(QUERIES) if QUERIES.exists() else {}
    return _queries


def load_solutions(dataset=IOLLM_DATASET):
    return {f"{p['year']}_p{p['problem_number']}": dict(p) for p in load_problems(dataset)}

def load_results(path):
    data = load_json(path)
    return data.get("results", data) if isinstance(data, dict) else data

def extract_answer(text):
    if not text:
        return ""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return "\n\n".join(m.strip() for m in matches) if matches else text

def parse_json(text):
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {}

def compute_chrf(pairs):
    refs, hyps = [], []
    for p in (pairs or {}).values():
        if p.get("expected") and p.get("predicted"):
            refs.append(p["expected"])
            hyps.append(p["predicted"])
    return chrf.corpus_score(hyps, [refs]).score if refs else 0.0


def fix_scores(scores, pairs):
    """catch obvious grading mistakes"""
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

        # false negative: exact match marked wrong
        if score == 0 and (exp_n == pred_n or (exp_n in pred_n and len(exp_n) > 5)):
            fixed[item_id] = 1
            continue

        # false positive checks
        if score == 1 and exp_n != pred_n:
            # ascii vs non-ascii (wrong direction)
            exp_ascii = exp.replace(" ", "").replace(".", "").replace(",", "").isascii()
            pred_ascii = pred.replace(" ", "").replace(".", "").replace(",", "").isascii()
            if len(exp) > 5 and len(pred) > 5 and exp_ascii != pred_ascii:
                fixed[item_id] = 0
                continue

            # no word overlap
            exp_words, pred_words = set(exp_n.split()), set(pred_n.split())
            if len(exp_words) >= 2 and len(pred_words) >= 2 and not (exp_words & pred_words):
                fixed[item_id] = 0
                continue

            # different numbers
            exp_nums, pred_nums = re.findall(r'\d+', exp), re.findall(r'\d+', pred)
            if exp_nums and pred_nums and len(exp_nums[0]) > 1 and exp_nums[0] != pred_nums[0]:
                fixed[item_id] = 0
                continue

            # tiny prediction for long expected
            if len(pred) <= 3 and len(exp) > 10:
                fixed[item_id] = 0

    return fixed


async def grade(result, problem, grader, rules_mode, queries=None):
    pid = result["problem_id"]

    if rules_mode:
        rules = problem.get("solution_rules", "")
        if not rules:
            return {"problem_id": pid, "error": "no rules", "found": 0, "total": 0}
        prompt = RULES_GRADER.format(rules=rules, response=result.get("response", ""))
    else:
        solution = problem.get("solution", "")
        if not solution:
            return {"problem_id": pid, "scores": {}, "n_correct": 0, "n_total": 0, "chrf": 0.0, "error": "no solution"}

        if queries is None:
            queries = get_queries()
        items = queries.get(pid, {})
        items_text = json.dumps(items, indent=2, ensure_ascii=False) if items else "(see solution below)"
        prompt = GRADER.format(items=items_text, solution=solution, answer=extract_answer(result.get("response", "")))

    try:
        if grader == "gpt-5-mini-reasoning":
            r = client.responses.create(
                model="gpt-5-mini", reasoning={"effort": "low"},
                input=[{"role": "user", "content": prompt}]
            )
            output_text = ""
            for item in r.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            output_text = c.text
            data = parse_json(output_text)
        else:
            kwargs = {"model": grader, "messages": [{"role": "user", "content": prompt}]}
            if not grader.startswith("o") and "5-mini" not in grader:
                kwargs["temperature"] = 0
            r = await async_client.chat.completions.create(**kwargs)
            data = parse_json(r.choices[0].message.content)

        if rules_mode:
            return {
                "problem_id": pid,
                "rules": data.get("rules", {}),
                "found": data.get("found", 0),
                "total": data.get("total", 0),
                "error": None
            }
        else:
            scores = data.get("scores", data)
            pairs = data.get("answer_pairs", {})
            scores = fix_scores(scores, pairs)
            return {
                "problem_id": pid,
                "scores": scores,
                "answer_pairs": pairs,
                "n_correct": sum(1 for v in scores.values() if v == 1),
                "n_total": len(scores),
                "chrf": compute_chrf(pairs),
                "error": None
            }
    except Exception as e:
        if rules_mode:
            return {"problem_id": pid, "error": str(e), "found": 0, "total": 0}
        return {"problem_id": pid, "scores": {}, "n_correct": 0, "n_total": 0, "chrf": 0.0, "error": str(e)}


async def grade_all(results, problems, grader, workers, rules_mode, queries=None):
    sem = asyncio.Semaphore(workers)

    async def go(r, i):
        async with sem:
            prob = problems.get(r["problem_id"])
            if not prob:
                return {"problem_id": r["problem_id"], "error": "not found"}
            out = await grade(r, prob, grader, rules_mode, queries)
            print(f"[{i+1}/{len(results)}] {r['problem_id']}")
            return out

    return await asyncio.gather(*[go(r, i) for i, r in enumerate(results)])


def evaluate(results_path, grader="gpt-4o", dataset=IOLLM_DATASET,
             output=None, workers=10, use_queries=True, rules_mode=False):
    results = [r for r in load_results(results_path) if not r.get("error")]
    problems = load_solutions(dataset)

    queries = get_queries() if use_queries and not rules_mode else None
    if rules_mode:
        results = [r for r in results if problems.get(r["problem_id"], {}).get("solution_rules")]

    if not results:
        print("nothing to evaluate")
        return

    print(f"grading {len(results)} results with {grader}")
    evals = asyncio.run(grade_all(results, problems, grader, workers, rules_mode, queries))
    ok = [e for e in evals if not e.get("error")]

    # add per-problem scores
    for e in evals:
        if not e.get("error"):
            if rules_mode:
                e["score_20"] = round(20 * e["found"] / e["total"], 2) if e["total"] else 0
                e["full_credit"] = e["found"] == e["total"] and e["total"] > 0
            else:
                e["score_20"] = round(20 * e["n_correct"] / e["n_total"], 2) if e["n_total"] else 0
                e["full_credit"] = e["n_correct"] == e["n_total"] and e["n_total"] > 0

    # summary
    if rules_mode:
        found = sum(e["found"] for e in ok)
        total = sum(e["total"] for e in ok)
        print(f"\nrules: {found}/{total} ({found/total*100:.0f}%)" if total else "\nno rules graded")
        summary = {"mode": "rules", "grader": grader, "found": found, "total": total}
    else:
        correct = sum(e["n_correct"] for e in ok)
        total = sum(e["n_total"] for e in ok)
        avg_chrf = sum(e["chrf"] for e in ok) / len(ok) if ok else 0
        print(f"\n{correct}/{total} correct ({correct/total*100:.0f}%)" if total else "\nno items graded")
        print(f"avg chrf: {avg_chrf:.1f}")
        summary = {"mode": "items", "grader": grader, "correct": correct, "total": total, "chrf": round(avg_chrf, 1)}

    summary["avg_score_20"] = round(sum(e.get("score_20", 0) for e in ok) / len(ok), 2) if ok else 0
    summary["full_problem_rate"] = round(100 * sum(1 for e in ok if e.get("full_credit")) / len(ok), 1) if ok else 0
    summary["n_graded"] = len(ok)
    summary["n_failed"] = len(evals) - len(ok)

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
    p.add_argument("--grader", default="gpt-4o")
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--rules", action="store_true")
    p.add_argument("--no-queries", action="store_true")
    a = p.parse_args()

    output = a.output or (RESULTS_DIR / "evaluations" / f"{a.results.stem}_eval.json")
    evaluate(a.results, a.grader, IOLLM_DATASET, output, a.workers, not a.no_queries, a.rules)
