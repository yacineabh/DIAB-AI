# Step 2 — Raw Frontier Model Batches

## What's in this folder

42 raw text files (`batch_01.txt` → `batch_42.txt`), each containing the direct output of a frontier model (Claude Opus / GPT-4) in response to a structured prompting request.

## Batch format

Each batch file contains approximately 30 JSON records separated by newlines. Each record follows this schema:

```json
{
  "profile": "Age: 45, Sex: M, Diabetes Type: Type 2, Weight: 98 kg",
  "input": "D1 07:30 G=148 | 08:00 M=toast C=25 | ...",
  "reasoning": "The patient shows a consistent pattern of ...",
  "output": "{\"feature\": \"RECOMMENDATIONS\", \"message\": \"...\", \"recommendations\": [...]}",
  "category": "analysis",
  "diabetes_type": "Type 2"
}
```

## Why raw outputs need processing

The frontier model occasionally:
- Wraps JSON in markdown code blocks (` ```json `)
- Adds numbered prefixes (`1. { ... }`)
- Produces reasoning that is too short (< 150 chars)
- Outputs invalid JSON in the `output` field
- Duplicates a patient profile across batches

All of these are handled by `../step3_data_processing/parser.py`.

## Frontier model used

These batches were generated using Claude Opus with structured batch prompts designed to maximize JSON adherence. Each prompt included:
- A strict JSON schema specification
- 2-3 few-shot examples
- An explicit instruction to never invent glucose values not present in the provided logs

## Processing

Run `parser.py` from `step3_data_processing/` to parse these into the clean training dataset:

```bash
cd ../step3_data_processing
python parser.py
```

Output: `../step4_training_dataset/diabetix_training_data.jsonl`
