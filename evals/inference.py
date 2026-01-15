"""inference"""
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from .api import client, async_client, is_gpt5, save_json, load_json, load_problems, IOLLM_DATASET
    from .prompts import SOLVER
except ImportError:
    from api import client, async_client, is_gpt5, save_json, load_json, load_problems, IOLLM_DATASET
    from prompts import SOLVER

RESULTS_DIR = Path(__file__).parent.parent / "results"
TRACKING_FILE = RESULTS_DIR / "tracking.json"


def pid(p):
    return f"{p['year']}_p{p['problem_number']}"

def prompt(p):
    return f"{SOLVER}\n\n{p['problem_text']}"

def filter_problems(problems, years=None, nums=None):
    if years:
        problems = [p for p in problems if int(p["year"]) in years]
    if nums:
        problems = [p for p in problems if p["problem_number"] in nums]
    return problems


# tracking

def load_tracking():
    return load_json(TRACKING_FILE) if TRACKING_FILE.exists() else {"jobs": {}, "responses": {}}

def save_tracking(t):
    save_json(t, TRACKING_FILE)

def track(resp_id=None, batch_id=None, **meta):
    t = load_tracking()
    meta["created"] = datetime.now().isoformat()
    if resp_id:
        t["responses"][resp_id] = meta
    if batch_id:
        t["jobs"][batch_id] = meta
    save_tracking(t)


# realtime

async def infer_one(prob, model, reasoning, dataset):
    t0 = time.time()
    resp_id, usage = None, {}
    problem_id = pid(prob)

    try:
        if is_gpt5(model):
            r = await async_client.responses.create(
                model=model, input=prompt(prob),
                reasoning={"effort": reasoning}, store=True, background=True
            )
            resp_id = r.id
            track(resp_id=resp_id, problem_id=problem_id, model=model,
                  reasoning=reasoning, dataset=dataset, status="pending")

            while r.status in ("queued", "in_progress"):
                await asyncio.sleep(5)
                r = client.responses.retrieve(resp_id)

            if r.status != "completed":
                raise Exception(f"status: {r.status}")

            text = r.output_text
            if r.usage:
                usage = {
                    "input": r.usage.input_tokens,
                    "output": r.usage.output_tokens,
                    "reasoning": getattr(r.usage.output_tokens_details, 'reasoning_tokens', 0)
                                 if r.usage.output_tokens_details else 0
                }
        else:
            r = await async_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt(prob)}]
            )
            resp_id = getattr(r, 'id', None)
            text = r.choices[0].message.content
            if r.usage:
                usage = {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}

        return {
            "problem_id": problem_id, "response": text, "response_id": resp_id,
            "model": model, "reasoning": reasoning, "dataset": dataset,
            "usage": usage, "latency": time.time() - t0, "error": None
        }
    except Exception as e:
        return {
            "problem_id": problem_id, "response": "", "response_id": resp_id,
            "model": model, "reasoning": reasoning, "dataset": dataset,
            "usage": usage, "latency": time.time() - t0, "error": str(e)
        }


async def infer_all(probs, model, reasoning, dataset, workers=4, output=None):
    sem = asyncio.Semaphore(workers)
    results = []

    async def go(p, i):
        async with sem:
            r = await infer_one(p, model, reasoning, dataset)
            results.append(r)
            ok = "✓" if not r["error"] else "✗"
            print(f"[{len(results)}/{len(probs)}] {ok} {r['problem_id']} ({r['latency']:.0f}s)")
            if output and len(results) % 5 == 0:
                save_json({"metadata": {}, "results": results}, output)

    await asyncio.gather(*[go(p, i) for i, p in enumerate(probs)])
    return results


def run_inference(model="gpt-5.2", reasoning="high", dataset=IOLLM_DATASET,
                  output=None, workers=4, years=None, problems=None):
    probs = filter_problems(load_problems(dataset), years, problems)
    if not probs:
        print("no problems match filters")
        return

    print(f"running {len(probs)} problems | {model} | {reasoning}")
    start = datetime.now()
    results = asyncio.run(infer_all(probs, model, reasoning, dataset, workers, output))

    n_ok = sum(1 for r in results if not r["error"])
    print(f"done: {n_ok}/{len(results)}")

    out = {
        "metadata": {
            "model": model, "reasoning": reasoning, "dataset": dataset,
            "n_problems": len(probs), "n_success": n_ok, "n_failed": len(results) - n_ok,
            "started": start.isoformat(), "finished": datetime.now().isoformat()
        },
        "results": results
    }
    if output:
        save_json(out, output)
    return out


# batch

def submit_batch(model="gpt-5.2", reasoning="high", dataset=IOLLM_DATASET,
                 output=None, years=None, problems=None):
    probs = filter_problems(load_problems(dataset), years, problems)
    if not probs:
        print("no problems match filters")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    batch_file = RESULTS_DIR / f"batch_input_{int(time.time())}.jsonl"
    endpoint = "/v1/responses" if is_gpt5(model) else "/v1/chat/completions"
    problem_ids = []

    with open(batch_file, "w") as f:
        for p in probs:
            problem_ids.append(pid(p))
            if is_gpt5(model):
                body = {"model": model, "input": prompt(p), "reasoning": {"effort": reasoning}}
            else:
                body = {"model": model, "messages": [{"role": "user", "content": prompt(p)}]}
            f.write(json.dumps({"custom_id": pid(p), "method": "POST", "url": endpoint, "body": body}) + "\n")

    file = client.files.create(file=open(batch_file, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=file.id, endpoint=endpoint, completion_window="24h")

    track(batch_id=batch.id, status=batch.status, model=model, reasoning=reasoning,
          dataset=dataset, n_problems=len(probs), problem_ids=problem_ids,
          output=str(output) if output else None, input_file_id=file.id, years=years)

    print(f"submitted: {batch.id}")
    print(f"  {len(probs)} problems | {model} | {reasoning}")
    return {"batch_id": batch.id, "n_problems": len(probs)}


def download_batch(batch_id, output, job_info=None):
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        return None, b.status

    raw = RESULTS_DIR / f"{batch_id}.jsonl"
    raw.write_bytes(client.files.content(b.output_file_id).read())

    results = []
    for line in open(raw):
        item = json.loads(line)
        body = item.get("response", {}).get("body", {})
        text = body.get("output_text") or body.get("choices", [{}])[0].get("message", {}).get("content", "")

        usage = body.get("usage", {})
        usage_out = {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)}
        if usage.get("output_tokens_details"):
            usage_out["reasoning"] = usage["output_tokens_details"].get("reasoning_tokens", 0)

        result = {
            "problem_id": item["custom_id"],
            "response": text,
            "response_id": body.get("id"),
            "model": body.get("model") or (job_info.get("model") if job_info else None),
            "reasoning": (body.get("reasoning", {}) or {}).get("effort") or (job_info.get("reasoning") if job_info else None),
            "dataset": job_info.get("dataset") if job_info else None,
            "batch_id": batch_id,
            "usage": usage_out,
            "error": str(item["error"]) if item.get("error") else None
        }
        results.append(result)
        if result["response_id"]:
            track(resp_id=result["response_id"], problem_id=result["problem_id"],
                  model=result["model"], reasoning=result["reasoning"],
                  dataset=result["dataset"], status="completed", batch_id=batch_id)

    n_ok = sum(1 for r in results if not r["error"])
    save_json({"metadata": {"batch_id": batch_id, "n_problems": len(results), "n_success": n_ok}, "results": results}, output)
    return results, output


# job management

def check_batches(download=True):
    t = load_tracking()
    jobs = t.get("jobs", {})
    if not jobs:
        print("no tracked jobs")
        return

    print(f"checking {len(jobs)} jobs...\n")
    for batch_id, info in jobs.items():
        try:
            b = client.batches.retrieve(batch_id)
            old = info.get("status")
            info["status"] = b.status

            progress = f"({b.request_counts.completed}/{b.request_counts.total})" if b.request_counts else ""
            icon = {"completed": "✓", "failed": "✗", "expired": "⏰"}.get(b.status, "⏳")
            print(f"{icon} {batch_id[:30]}... {b.status} {progress} | {info.get('model')} | {info.get('n_problems')} problems")

            if download and b.status == "completed" and old != "completed":
                output = info.get("output") or str(RESULTS_DIR / f"{batch_id}.json")
                results, path = download_batch(batch_id, Path(output), info)
                if results:
                    n_ok = sum(1 for r in results if not r["error"])
                    print(f"   → downloaded: {path} ({n_ok}/{len(results)} ok)")
                    info["downloaded"] = datetime.now().isoformat()
        except Exception as e:
            print(f"? {batch_id[:30]}... error: {e}")

    save_tracking(t)
    statuses = [j.get("status") for j in jobs.values()]
    print(f"\n{statuses.count('completed')} done, {statuses.count('in_progress')} running, {statuses.count('failed')} failed")


def check_responses():
    t = load_tracking()
    pending = {k: v for k, v in t.get("responses", {}).items() if v.get("status") != "recovered"}
    if not pending:
        print("no pending responses")
        return

    print(f"checking {len(pending)} responses...\n")
    recovered, to_remove = [], []

    for resp_id, info in list(pending.items()):
        try:
            r = client.responses.retrieve(resp_id)
            if r.status == "completed":
                usage = {}
                if r.usage:
                    usage = {
                        "input": r.usage.input_tokens,
                        "output": r.usage.output_tokens,
                        "reasoning": getattr(r.usage.output_tokens_details, 'reasoning_tokens', 0)
                                     if r.usage.output_tokens_details else 0
                    }
                recovered.append({
                    "problem_id": info.get("problem_id", "unknown"),
                    "response": r.output_text,
                    "response_id": resp_id,
                    "model": info.get("model") or r.model,
                    "reasoning": info.get("reasoning"),
                    "dataset": info.get("dataset"),
                    "usage": usage,
                    "error": None
                })
                to_remove.append(resp_id)
                print(f"✓ {info.get('problem_id', resp_id[:20])} recovered")
            elif r.status in ("queued", "in_progress"):
                print(f"⏳ {info.get('problem_id', resp_id[:20])} {r.status}")
            else:
                to_remove.append(resp_id)
                print(f"✗ {info.get('problem_id', resp_id[:20])} {r.status}")
        except Exception as e:
            print(f"? {info.get('problem_id', resp_id[:20])} {e}")

    for rid in to_remove:
        t["responses"].pop(rid, None)
    save_tracking(t)

    if recovered:
        output = RESULTS_DIR / f"recovered_{int(time.time())}.json"
        save_json({"metadata": {"type": "recovered"}, "results": recovered}, output)
        print(f"\n→ saved {len(recovered)} to {output}")


def list_jobs():
    jobs = load_tracking().get("jobs", {})
    if not jobs:
        print("no tracked jobs")
        return
    for bid, info in jobs.items():
        icon = {"completed": "✓", "failed": "✗"}.get(info.get("status"), "⏳")
        print(f"{icon} {bid} | {info.get('model')} | {info.get('reasoning')} | {info.get('n_problems')} problems")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5.2")
    p.add_argument("--reasoning", default="high", choices=["none", "low", "medium", "high"])
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--years", type=int, nargs="+")
    p.add_argument("--problems", type=int, nargs="+")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--jobs", action="store_true")
    a = p.parse_args()

    if a.check:
        check_batches()
        print()
        check_responses()
    elif a.jobs:
        list_jobs()
    elif a.batch:
        output = a.output or (RESULTS_DIR / f"{a.model.replace('.', '-')}_{a.reasoning}.json")
        submit_batch(a.model, a.reasoning, IOLLM_DATASET, output, a.years, a.problems)
    else:
        output = a.output or (RESULTS_DIR / f"{a.model.replace('.', '-')}_{a.reasoning}.json")
        run_inference(a.model, a.reasoning, IOLLM_DATASET, output, a.workers, a.years, a.problems)
