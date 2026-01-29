import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

from .api import (
    client, async_client, pid, save_json, load_json,
    load_problems, RESULTS_DIR, IOLLM_DATASET
)
from .prompts import SOLVER

TRACKING_FILE = RESULTS_DIR / "tracking.json"

REASONING_EFFORTS = ["none", "low", "medium", "high"]


def make_prompt(problem):
    return f"{SOLVER}\n\n{problem['problem_text']}"


def is_reasoning_model(model):
    return model.startswith("o") or "5" in model


def filter_problems(problems, years=None, nums=None):
    if years:
        problems = [p for p in problems if int(p["year"]) in years]
    if nums:
        problems = [p for p in problems if p["problem_number"] in nums]
    return problems


# tracking

def load_tracking():
    return load_json(TRACKING_FILE) if TRACKING_FILE.exists() else {"jobs": {}}


def save_tracking(t):
    save_json(t, TRACKING_FILE)


def track_job(batch_id, **meta):
    t = load_tracking()
    meta["created"] = datetime.now().isoformat()
    t["jobs"][batch_id] = meta
    save_tracking(t)


# realtime inference

async def infer_one(prob, model, reasoning, dataset):
    t0 = time.time()
    problem_id = pid(prob)

    try:
        if is_reasoning_model(model) and reasoning != "none":
            # reasoning model with effort control
            r = await async_client.responses.create(
                model=model,
                input=make_prompt(prob),
                reasoning={"effort": reasoning}
            )
            text = r.output_text
            usage = {}
            if r.usage:
                usage = {
                    "input": r.usage.input_tokens,
                    "output": r.usage.output_tokens,
                    "reasoning": getattr(r.usage.output_tokens_details, 'reasoning_tokens', 0)
                                 if r.usage.output_tokens_details else 0
                }
        else:
            # standard chat completion
            r = await async_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": make_prompt(prob)}]
            )
            text = r.choices[0].message.content
            usage = {}
            if r.usage:
                usage = {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}
                if hasattr(r.usage, 'completion_tokens_details') and r.usage.completion_tokens_details:
                    usage["reasoning"] = getattr(r.usage.completion_tokens_details, 'reasoning_tokens', 0)

        return {
            "problem_id": problem_id,
            "response": text,
            "model": model,
            "reasoning": reasoning,
            "dataset": dataset,
            "usage": usage,
            "latency": round(time.time() - t0, 1),
            "error": None
        }
    except Exception as e:
        return {
            "problem_id": problem_id,
            "response": "",
            "model": model,
            "reasoning": reasoning,
            "dataset": dataset,
            "usage": {},
            "latency": round(time.time() - t0, 1),
            "error": str(e)
        }


async def infer_all(probs, model, reasoning, dataset, workers, output=None):
    sem = asyncio.Semaphore(workers)
    results = []

    async def go(p):
        async with sem:
            r = await infer_one(p, model, reasoning, dataset)
            results.append(r)
            status = "ok" if not r["error"] else "err"
            print(f"[{len(results)}/{len(probs)}] {status} {r['problem_id']} ({r['latency']}s)")
            if output and len(results) % 5 == 0:
                save_json({"metadata": {"model": model, "reasoning": reasoning}, "results": results}, output)

    await asyncio.gather(*[go(p) for p in probs])
    return results


def run_inference(model="gpt-5.2", reasoning="high", dataset=IOLLM_DATASET,
                  output=None, workers=4, years=None, problems=None):
    probs = filter_problems(load_problems(dataset), years, problems)
    if not probs:
        print("no problems match filters")
        return

    print(f"running {len(probs)} problems with {model} (reasoning={reasoning})")
    start = datetime.now()
    results = asyncio.run(infer_all(probs, model, reasoning, dataset, workers, output))

    n_ok = sum(1 for r in results if not r["error"])
    print(f"done: {n_ok}/{len(results)}")

    out = {
        "metadata": {
            "model": model,
            "reasoning": reasoning,
            "dataset": dataset,
            "n_problems": len(probs),
            "n_success": n_ok,
            "n_failed": len(results) - n_ok,
            "started": start.isoformat(),
            "finished": datetime.now().isoformat()
        },
        "results": results
    }
    if output:
        save_json(out, output)
        print(f"saved: {output}")
    return out


# batch inference

def submit_batch(model="gpt-5.2", reasoning="high", dataset=IOLLM_DATASET,
                 output=None, years=None, problems=None):
    probs = filter_problems(load_problems(dataset), years, problems)
    if not probs:
        print("no problems match filters")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    batch_file = RESULTS_DIR / f"batch_input_{int(time.time())}.jsonl"

    use_responses = is_reasoning_model(model) and reasoning != "none"
    endpoint = "/v1/responses" if use_responses else "/v1/chat/completions"

    with open(batch_file, "w") as f:
        for p in probs:
            if use_responses:
                body = {
                    "model": model,
                    "input": make_prompt(p),
                    "reasoning": {"effort": reasoning}
                }
            else:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": make_prompt(p)}]
                }
            req = {"custom_id": pid(p), "method": "POST", "url": endpoint, "body": body}
            f.write(json.dumps(req) + "\n")

    with open(batch_file, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=endpoint,
        completion_window="24h"
    )

    track_job(
        batch_id=batch.id,
        status=batch.status,
        model=model,
        reasoning=reasoning,
        dataset=dataset,
        n_problems=len(probs),
        output=str(output) if output else None,
        input_file=str(batch_file),
        years=years
    )

    print(f"submitted: {batch.id}")
    print(f"  {len(probs)} problems | {model} | reasoning={reasoning}")
    return {"batch_id": batch.id, "n_problems": len(probs)}


def check_batch(batch_id):
    b = client.batches.retrieve(batch_id)
    counts = b.request_counts
    progress = f"{counts.completed}/{counts.total}" if counts else "?"
    print(f"{batch_id}: {b.status} ({progress})")
    return b


def download_batch(batch_id, output=None):
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        print(f"batch not ready: {b.status}")
        return None

    t = load_tracking()
    job_info = t.get("jobs", {}).get(batch_id, {})

    raw_file = RESULTS_DIR / f"{batch_id}.jsonl"
    raw_file.write_bytes(client.files.content(b.output_file_id).read())

    results = []
    with open(raw_file) as f:
        for line in f:
            item = json.loads(line)
            body = item.get("response", {}).get("body", {})

            # handle both responses API and chat completions API
            text = body.get("output_text") or body.get("choices", [{}])[0].get("message", {}).get("content", "")

            usage = body.get("usage", {})
            usage_out = {
                "input": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
                "output": usage.get("output_tokens") or usage.get("completion_tokens", 0)
            }
            if usage.get("output_tokens_details"):
                usage_out["reasoning"] = usage["output_tokens_details"].get("reasoning_tokens", 0)
            elif usage.get("completion_tokens_details"):
                usage_out["reasoning"] = usage["completion_tokens_details"].get("reasoning_tokens", 0)

            results.append({
                "problem_id": item["custom_id"],
                "response": text,
                "model": body.get("model") or job_info.get("model"),
                "reasoning": job_info.get("reasoning"),
                "dataset": job_info.get("dataset"),
                "batch_id": batch_id,
                "usage": usage_out,
                "error": str(item["error"]) if item.get("error") else None
            })

    n_ok = sum(1 for r in results if not r["error"])
    out = {
        "metadata": {
            "batch_id": batch_id,
            "model": job_info.get("model"),
            "reasoning": job_info.get("reasoning"),
            "dataset": job_info.get("dataset"),
            "n_problems": len(results),
            "n_success": n_ok,
            "n_failed": len(results) - n_ok
        },
        "results": results
    }

    output = output or job_info.get("output") or (RESULTS_DIR / f"{batch_id}.json")
    save_json(out, output)
    print(f"downloaded: {output} ({n_ok}/{len(results)} ok)")

    if batch_id in t.get("jobs", {}):
        t["jobs"][batch_id]["status"] = "completed"
        t["jobs"][batch_id]["downloaded"] = datetime.now().isoformat()
        save_tracking(t)

    return out


def check_batches(auto_download=True):
    t = load_tracking()
    jobs = t.get("jobs", {})
    if not jobs:
        print("no tracked jobs")
        return

    print(f"checking {len(jobs)} jobs...\n")
    for batch_id, info in jobs.items():
        try:
            b = client.batches.retrieve(batch_id)
            was_pending = info.get("status") != "completed"
            info["status"] = b.status
            counts = b.request_counts
            progress = f"({counts.completed}/{counts.total})" if counts else ""
            print(f"  {batch_id[:24]}... {b.status} {progress} | {info.get('model')} | {info.get('reasoning')}")

            if auto_download and b.status == "completed" and was_pending and not info.get("downloaded"):
                save_tracking(t)
                download_batch(batch_id)
                t = load_tracking()
        except Exception as e:
            print(f"  {batch_id[:24]}... error: {e}")

    save_tracking(t)
    statuses = [j.get("status") for j in jobs.values()]
    completed = statuses.count("completed")
    running = statuses.count("in_progress") + statuses.count("validating") + statuses.count("finalizing")
    print(f"\n{completed} done, {running} running")


def list_jobs():
    jobs = load_tracking().get("jobs", {})
    if not jobs:
        print("no tracked jobs")
        return
    for bid, info in jobs.items():
        print(f"  {bid} | {info.get('status', '?')} | {info.get('model')} | {info.get('reasoning')} | {info.get('n_problems')} problems")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", default="gpt-5.2")
    p.add_argument("--reasoning", "-r", default="high", choices=REASONING_EFFORTS)
    p.add_argument("--dataset", "-d", default=IOLLM_DATASET)
    p.add_argument("--output", "-o", type=Path)
    p.add_argument("--workers", "-w", type=int, default=4)
    p.add_argument("--years", type=int, nargs="+")
    p.add_argument("--problems", type=int, nargs="+")

    p.add_argument("--batch", action="store_true", help="submit batch job")
    p.add_argument("--check", action="store_true", help="check jobs and download if ready")
    p.add_argument("--download", type=str, metavar="BATCH_ID", help="manually download a batch")
    p.add_argument("--jobs", action="store_true", help="list tracked jobs")

    a = p.parse_args()

    if a.jobs:
        list_jobs()
    elif a.check:
        check_batches()
    elif a.download:
        download_batch(a.download, a.output)
    elif a.batch:
        output = a.output or (RESULTS_DIR / f"{a.model.replace('.', '-')}_{a.reasoning}_batch.json")
        submit_batch(a.model, a.reasoning, a.dataset, output, a.years, a.problems)
    else:
        output = a.output or (RESULTS_DIR / f"{a.model.replace('.', '-')}_{a.reasoning}.json")
        run_inference(a.model, a.reasoning, a.dataset, output, a.workers, a.years, a.problems)
