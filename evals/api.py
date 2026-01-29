import json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

client = OpenAI()
async_client = AsyncOpenAI()

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
IOLLM_DATASET = "zeynebnk/iollm"


def pid(problem):
    return f"{problem['year']}_p{problem['problem_number']}"


def parse_json(text):
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    return json.loads(m.group()) if m else {}


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_problems(dataset=IOLLM_DATASET, split="test"):
    from datasets import load_dataset
    return list(load_dataset(dataset, split=split))
