# Step 4 — Training Dataset

## `diabetix_training_data.jsonl`

This is the final, clean training dataset produced by running `parser.py` over all 42 raw batches.

## Statistics

| Metric | Value |
|--------|-------|
| Total examples | ~600 |
| Format | JSON Lines (`.jsonl`) |
| Category — `analysis` | ~400 examples |
| Category — `advisory` | ~200 examples |
| Diabetes Type 1 | ~50% |
| Diabetes Type 2 | ~50% |
| Avg reasoning length | ~800 characters |
| Duplicates | **0** (verified) |

## Schema

```json
{
  "profile":       "Age: 34, Sex: F, Diabetes Type: Type 1, Weight: 62 kg",
  "input":         "D1 08:30 G=142 | 09:00 M=oatmeal C=55 I=5R | ...",
  "reasoning":     "<step-by-step clinical reasoning from the frontier model>",
  "output":        "{\"feature\": \"POST_MEAL_SPIKE\", \"patterns\": [...]}",
  "category":      "analysis",
  "diabetes_type": "Type 1"
}
```

## Reproduce this file

```bash
cd ../step3_data_processing
python parser.py
# Output is written to ../step4_training_dataset/diabetix_training_data.jsonl
```

## Usage in training

The LoRA fine-tuning notebooks in `../step5_lora_finetuning/` expect this file to be uploaded to Kaggle as a dataset. The notebooks reference it at:

```
/kaggle/input/datasets/<username>/diabetic-ai-data/diabetix_training_data.jsonl
```

For the V1 notebook, a pre-split version was used (train/val/test: 1818/100/100).
