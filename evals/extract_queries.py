"""query-answer formatting"""
import asyncio
import json
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from .api import async_client, save_json, load_json, load_problems, QUERIES, IOLLM_DATASET
    from .prompts import EXTRACT_QUERIES
except ImportError:
    from api import async_client, save_json, load_json, load_problems, QUERIES, IOLLM_DATASET
    from prompts import EXTRACT_QUERIES


def pid(p):
    return f"{p['year']}_p{p['problem_number']}"

def parse_json(text):
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {}


async def extract_one(sem, prob, i, total):
    async with sem:
        problem_id = pid(prob)
        prompt = EXTRACT_QUERIES.format(problem=prob["problem_text"], solution=prob["solution"])
        try:
            r = await async_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            items = parse_json(r.choices[0].message.content).get("items", {})
            print(f"[{i+1}/{total}] {problem_id}: {len(items)} items")
            return {"problem_id": problem_id, "items": items}
        except Exception as e:
            print(f"[{i+1}/{total}] {problem_id} error: {e}")
            return {"problem_id": problem_id, "items": {}, "error": str(e)}


def extract_queries(dataset=IOLLM_DATASET, output=None, workers=10):
    problems = load_problems(dataset)
    print(f"extracting from {len(problems)} problems...\n")

    async def go():
        sem = asyncio.Semaphore(workers)
        return await asyncio.gather(*[extract_one(sem, p, i, len(problems)) for i, p in enumerate(problems)])

    results = asyncio.run(go())
    extracted = {r["problem_id"]: r["items"] for r in results}
    print(f"\ntotal items: {sum(len(v) for v in extracted.values())}")

    save_json(extracted, output or QUERIES)
    print(f"saved: {output or QUERIES}")
    return extracted


def update_queries(problem_ids, dataset=IOLLM_DATASET, queries_path=QUERIES):
    existing = load_json(queries_path) if queries_path.exists() else {}
    problems = [p for p in load_problems(dataset) if pid(p) in problem_ids]
    if not problems:
        print(f"no problems match: {problem_ids}")
        return

    print(f"updating {len(problems)} problems...\n")

    async def go():
        sem = asyncio.Semaphore(5)
        return await asyncio.gather(*[extract_one(sem, p, i, len(problems)) for i, p in enumerate(problems)])

    for r in asyncio.run(go()):
        if not r.get("error"):
            existing[r["problem_id"]] = r["items"]
    save_json(existing, queries_path)
    print(f"saved: {queries_path}")


def validate_queries(queries_path=QUERIES, dataset=IOLLM_DATASET):
    if not queries_path.exists():
        print(f"not found: {queries_path}")
        return

    q = load_json(queries_path)
    problem_ids = {pid(p) for p in load_problems(dataset)}
    missing = problem_ids - set(q.keys())
    empty = [k for k, v in q.items() if not v]

    print(f"problems: {len(problem_ids)}, queries: {len(q)}, items: {sum(len(v) for v in q.values())}")
    if missing:
        print(f"missing ({len(missing)}): {', '.join(sorted(missing)[:10])}{'...' if len(missing) > 10 else ''}")
    if empty:
        print(f"empty ({len(empty)}): {', '.join(empty[:10])}")
    print(f"\n{'✓ valid' if not missing else '✗ missing problems'}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", type=Path, default=QUERIES)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--update", nargs="+")
    a = p.parse_args()

    if a.validate:
        validate_queries(a.output)
    elif a.update:
        update_queries(a.update, queries_path=a.output)
    else:
        extract_queries(output=a.output)
