# iollm


## setup

```bash
pip install -r requirements.txt
```

## eval

```bash
cd iollm/evals

# inference
python inference.py --model gpt-5.2 --reasoning high

# batch (cheaper)
python inference.py --batch
python inference.py --check

# eval
python evaluation.py results.json
```
