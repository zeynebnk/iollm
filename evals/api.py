"""api/inference"""
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

client = OpenAI()
async_client = AsyncOpenAI()

IOLLM_DATASET = "zeynebnk/iollm"
QUERIES = Path(__file__).parent.parent / "data" / "extracted_queries.json"

def is_gpt5(model):
    return model.startswith("gpt-5")

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
