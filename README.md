# IOLLM
## Language, Reasoning, and Latent Learning

The rapid progress of models on many benchmarks, including medal-level performance in international olympiads such as computing and mathematics, has brought them recognition as 'general reasoners'. But there's still a gap between how we talk about reasoning and how we measure it. Rather than asking whether models reason, IOLLM aims to isolate the mechanisms of reasoning in language models, and where that process breaks down, and use that to drive their reasoning further.

IOLLM is a project on rule induction as reasoning: how do language models infer a latent system from a tiny set of examples, and then generalize it correctly to new queries? We turn to the Linguistics Olympiad, the olympiad perhaps most aligned for language models. With self-contained problems requiring multi-step and compositional reasoning, it serves as an abstractable way to examine reasoning in language models. 

This repo contains code used to evaluate reasoning performance of language models on IOLLM. We isolate the impact of 'reasoning' on performance: we isolate representation (mis)alignment, contamination and prior knowledge, and formalize identifiability and information flow to connect forms of logical complexity to identify reasoning failures. This challenging frontier eval measures broadly relevant capabilities in hierarchical multi-step reasoning and efficient in-context deduction and is a well-grounded and useful framework.  

## setup

```bash
pip install -r requirements.txt
```

## usage

```bash
cd iollm/evals

# inference
python inference.py --model gpt-5.2 --reasoning high
python inference.py --model gpt-5.2 --batch
python inference.py --check

# evaluation
python evaluation.py results/gpt-5-2_high.json
python evaluation.py results/gpt-5-2_high.json -k 3  # majority vote
```

## options

```
inference.py
  --model, -m      model name (default: gpt-5.2)
  --reasoning, -r  effort: none, low, medium, high (default: high)
  --dataset, -d    dataset
  --batch          use batch API
  --check          check batch status & download
  --years          filter by year
  --problems       filter by problem number

evaluation.py
  --grader         grader model (default: gpt-5-mini)
  --dataset, -d    HuggingFace dataset for solutions
  -k               grade k times, majority vote
```

## dataset
Default: `zeynebnk/iollm`
Fields: `year`, `problem_number`, `problem_text`, `solution`

## logic
tools for analyzing logical structure of problems:

```python
from iollm.logic import analyze
from iollm.logic.data import Dataset

ds = Dataset.from_file("problem.json")
result = analyze(ds)

result.summary    # identifiability, entropy, graph stats
result.tokens     # per-token analysis
result.curve      # ambiguity reduction curve
```

computes:
- **identifiability**: can each token be uniquely determined?
- **certificates**: minimal evidence needed for identification  
- **ambiguity curves**: how ambiguity decreases with more examples
- **constraint graph**: co-occurrence structure, treewidth
