
SOLVER = """You are an expert solving International Linguistics Olympiad problems.
 
Analyze the language data, identify patterns, and solve the questions. Show your reasoning.

Provide final answers in <answer>...</answer> tags."""

GRADER = """You are a grader for the International Linguistics Olympiad (IOL) and are given a model's answer to a problem. Grade each answer item by comparing the model's answer to the expected answer.

## Question Query Items (extracted from problem):
{items}

## Full Solution Key:
{solution}

## Model's Answer:
{answer}

SCORING RULES:
- Score 1: answer matches expected or solution key
- Score 0: incorrect, wrong meaning, wrong form, missing, or ambiguous

RULES:
- Grade precisely and strictly for answer match
- MUST have correct tone marks/diacritics, words order, spelling, etc. These change meaning
- Accept near matches ONLY when it is clearly correct/equivalent. 
- You must be able to clearly and consistently justify the score for each item. 

TASK:
1. For each item ID, find and extract the model's answer for that query
2. Compare to the expected and the solution key 
3. Assign a strict score based on the scoring rules 

Return JSON:
{{"scores": {{"a_1": 1, "a_2": 0}}, "answer_pairs": {{"a_1": {{"expected": "...", "predicted": "..."}}}}}}"""

RULES_GRADER = """You are a grader for the International Linguistics Olympiad (IOL) and are given a model's response to a problem. Evaluate the model's reasoning/work to the linguistic rule identification. Be strict.

TASK:
- You are given the grading solution key with the key patterns the problem requires a solver to identify.
- Determine whether the model was able to correctly identify the key patterns and cite them correctly.
- Grade whether they state one of the key patterns in their solution.
- Score 1 for each rule they *correctly* identified and cited *which is listed in the grading solution key*. Return this over the total number of key patterns the solution guide has.

## Soltuion Guide Rules:
{rules}

## Model's Response:
{response}


Return JSON only:
{{"rules": {{"rule_name": {{"score": 1, "evidence": "quote"}}}}, "total": 5, "found": 3}}"""

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
