SOLVER = """You are an expert solving International Linguistics Olympiad problems.

Analyze the language data, identify and derive patterns, reason, and solve the questions. Show your reasoning.

Provide final answers in <answer>...</answer> tags."""

GRADER = """Grade a model's answers to an International Linguistics Olympiad (IOL) problem. Grade each answer item by comparing the model's answer to the expected answer.

## Query Items:
{items}

## Solution Key:
{solution}

## Model's Answer:
{answer}

TASK:
1. For each item, find and extract the model's answer
2. Compare to the expected answer from the solution key
3. Score 1 if correct, 0 if incorrect/missing

RULES:
- Grade precisely and strictly for answer match
- Be strict: require correct diacritics, tone marks, word order, spelling
- Accept equivalent forms only when clearly correct
- Extract exactly what the model wrote for each item

Return JSON:
{{"scores": {{"a_1": 1, "a_2": 0}}, "answer_pairs": {{"a_1": {{"expected": "...", "predicted": "..."}}}}}}"""

EXTRACT_QUERIES = """Extract all question items from this linguistics problem and map them to their correct answers.

## Problem:
{problem}

## Solution Key:
{solution}

TASK: Create a JSON mapping each question item to:
- "query": The exact question being asked (e.g., "Translate X into English" or "Determine the word for 'hello'")
- "expected": The correct answer from the solution

IMPORTANT RULES:
1. Use the EXACT spelling/format from the problem for any words given in the problem
2. Do NOT add hyphens or morpheme boundaries that aren't in the original problem
3. For translation tasks, the expected answer should be the translation only
4. Extract EVERY answerable item - typically labeled (a), (b), (c) or numbered 1, 2, 3, etc.
5. Be precise with diacritics and tone marks - they matter

Return ONLY valid JSON:
{{"items": {{
  "a_1": {{"query": "Translate X into English", "expected": "the answer"}},
  "b_1": {{"query": "Write 'hello' in Swahili", "expected": "jambo"}}
}}}}"""
