# Pillar 1 — Knowledge Distillation + LoRA Fine-Tuning

## Overview

This pillar implements **knowledge distillation**: we use a large, expensive frontier model (Claude / GPT-4) as a *teacher*, generate expert-level reasoning on synthetic diabetes cases, then train a compact **Qwen2.5-3B-Instruct** student model to replicate that reasoning at a fraction of the cost.

The student model powers three real-time features in the DiabAI app:
1. **Pattern detection** — identify recurring glucose patterns from 10-day logs
2. **Personalised recommendations** — actionable advice grounded in the patient's own data
3. **Insulin estimation** — mathematical dose suggestion with full reasoning chain

---

## Pipeline at a Glance

```
master_seeds.json
      │
      ▼
step1_synthetic_data_generation/
  Generate synthetic patient profiles + 10-day logs
  Send to frontier model for expert annotations
      │
      ▼
step2_raw_batches/
  Raw frontier model outputs (42 batches)
      │
      ▼
step3_data_processing/
  parser.py — validate, clean, deduplicate
  → diabetix_training_data.jsonl
      │
      ▼
step4_training_dataset/
  ~600 high-quality (profile, input, reasoning, output) examples
      │
      ▼
step5_lora_finetuning/
  LoRA fine-tune Qwen2.5-3B on Kaggle 2×T4
      │
      ▼
step6_inference_demo/
  Test the model on the three app features
```

---

## Step-by-Step Details

### Step 1 — Synthetic Data Generation

**`master_seeds.json`** contains the seed patient profiles: demographic combinations (age, sex, weight, diabetes type, insulin regimen) that define the diversity of the synthetic cohort.

**`01_generate_synthetic_data.ipynb`** shows how these seeds are expanded into full 10-day glucose logs following a compact log format:

```
D1 09:00 G=138 | 09:30 M=cereal C=50 I=6R | 11:15 G=250 | 13:00 G=145
```

Where `G=` glucose (mg/dL), `M=` meal, `C=` carbs (g), `I=` insulin dose.

Each synthetic patient log was then sent to a frontier model with a structured prompt asking for:
- `feature`: the dominant glycemic pattern (e.g., "POST_MEAL_SPIKE", "DAWN_PHENOMENON")
- `patterns`: list of specific observations with evidence from the log
- `recommendations`: actionable, grounded advice

---

### Step 2 — Raw Batches

**`step2_raw_batches/`** contains 42 raw `.txt` files — direct outputs from the frontier model, one batch per file, each containing ~30 JSON examples.

These are unprocessed: some have JSON formatting errors, some have markdown artifacts (` ```json ` wrappers), some have fields with insufficient detail. All of this is handled in step 3.

**Prompting strategy:** Each batch request asked the frontier model to produce `N` complete JSON records in a single response, using a strict schema. Batching was necessary to generate enough volume efficiently within API rate limits.

---

### Step 3 — Data Processing & Deduplication

**`parser.py`** is the heart of the data quality pipeline. It:

1. **Fixes common JSON formatting errors** — strips markdown artifacts, removes trailing commas, handles numbered lists
2. **Validates required fields** — `profile`, `input`, `reasoning`, `output`, `category`, `diabetes_type`
3. **Validates content quality**:
   - Reasoning must be ≥ 150 characters (rejects lazy/stub outputs)
   - `input` must contain "Glucose" (confirms it's actually a log, not a placeholder)
   - `output` must be valid JSON with the correct schema keys
4. **Deduplicates** using the first 120 characters of the `input` field as a fingerprint — catches exact-duplicate cases where the same patient profile was accidentally submitted twice
5. **Reports readiness** — checks target thresholds before declaring the dataset training-ready

**`02_data_processing_analysis.ipynb`** explores the processed data: distribution of categories, diabetes types, meal diversity (368 unique meal names), exercise patterns, and token length audit.

---

### Step 4 — Training Dataset

**`diabetix_training_data.jsonl`** — the final training-ready dataset.

| Metric | Value |
|--------|-------|
| Total examples | ~600 |
| Categories | analysis: ~400, advisory: ~200 |
| Diabetes types | Type 1: ~50%, Type 2: ~50% |
| Reasoning length | min 150 chars, avg ~800 chars |
| Duplicates | 0 (verified by parser) |

Each record has 6 fields:
```json
{
  "profile": "Age: 34, Sex: F, Diabetes Type: Type 1, Weight: 62 kg",
  "input": "D1 08:30 G=142 | 09:00 M=oatmeal C=55 I=5R | ...",
  "reasoning": "<step-by-step clinical reasoning>",
  "output": "{\"feature\": \"POST_MEAL_SPIKE\", \"patterns\": [...], ...}",
  "category": "analysis",
  "diabetes_type": "Type 1"
}
```

---

### Step 5 — LoRA Fine-Tuning

Two training runs are preserved here:

#### V1: `03_lora_finetuning_v1.ipynb` (Exploration)
- **Model:** `Qwen2.5-7B-Instruct` (4-bit quantized)
- **Config:** LoRA r=32, α=64, rslora=True, multi-GPU torchrun
- **Dataset:** 360 train / 40 val (analysis category only)
- **Result:** Training loss 0.91 → 0.31 over 3 epochs. Validated the approach.

#### Final: `04_lora_finetuning_final.ipynb` (Production)
- **Model:** `Qwen2.5-3B-Instruct` (full precision, not quantized — 32 GB VRAM available)
- **Config:** LoRA r=16, α=32, dropout=0 (Unsloth recommendation)
- **Dataset:** 1,818 train / 100 val / 100 test (full dataset with chat template formatting)
- **Training:** 3 epochs, effective batch size 32, LR 2e-4 cosine, ~28 min on 2×T4
- **Eval:** JSON validity check + field accuracy on held-out test set

**Why Qwen2.5-3B for production?** Smaller = faster inference = lower cost at scale. The 3B model is small enough to run on a single T4 at serving time, while the 7B exploration confirmed the task is learnable.

---

### Step 6 — Inference Demo

**`05_model_inference_demo.ipynb`** demonstrates the three app features:

| Feature | Input | Output |
|---------|-------|--------|
| **Estimated Insulin Need** | Current glucose + meal carbs + 10-day log | Recommended dose + ICR/ISF derivation |
| **Meal Impact Estimator** | Meal description + 10-day log | Predicted glucose curve + personalised advice |
| **Health Insight** | Current glucose reading + timestamp + log | Contextualised assessment + one action |

> **Setup:** Set `GEMINI_API_KEY` environment variable, or for the fine-tuned model, load from the Hugging Face Hub adapter path.

---

## Requirements

```bash
# For data processing (local)
pip install python-json-logger

# For fine-tuning (Kaggle/Colab GPU required)
pip install unsloth trl transformers accelerate datasets peft bitsandbytes

# For inference demo
pip install google-genai
```

---

## Disclaimer

All insulin dose suggestions produced by this model are **mathematical estimations** for research purposes only — not medical prescriptions. The model must never be deployed without appropriate clinical oversight.
