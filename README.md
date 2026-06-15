# DiabAI 🩺

> **AI-powered diabetes management** — three machine learning pillars designed to run at the heart of a diabetes self-management app.

---

## Architecture Overview

```
DiabAI
├── 1_knowledge_distillation/   ← Fine-tune a small LLM to reason about glucose logs
├── 2_glucose_forecasting/      ← Predict blood glucose 2 hours ahead
└── 3_meal_scan/                ← Classify food photos to estimate carbs
```

Each pillar is self-contained with its own data, models, and notebooks.

---

## Pillar 1 — Knowledge Distillation + LoRA Fine-Tuning

**Goal:** Distil the clinical reasoning ability of a large frontier model (GPT-4 / Claude) into a compact **Qwen2.5-3B-Instruct** that fits on-device or on a cheap inference server.

**Full pipeline:**

| Step | What happens |
|------|-------------|
| `step1_synthetic_data_generation/` | Generate thousands of synthetic diabetic patient profiles and 10-day glucose logs, then send them to a frontier model to produce expert analyses |
| `step2_raw_batches/` | Raw frontier-model outputs (42 batches × ~30 examples each) |
| `step3_data_processing/` | Parse, validate, and deduplicate the raw outputs using `parser.py` |
| `step4_training_dataset/` | Final clean `diabetix_training_data.jsonl` (~600 expert Q&A pairs) |
| `step5_lora_finetuning/` | LoRA fine-tune on Kaggle 2×T4 (Unsloth + TRL SFTTrainer) |
| `step6_inference_demo/` | Test the fine-tuned model on the three app features |

**Model:** `Qwen2.5-3B-Instruct` + LoRA (r=16, α=32) → 0.96% trainable params  
**Hardware:** Kaggle 2×T4 (32 GB VRAM total)  
**Training time:** ~28 minutes for 3 epochs on 1,818 examples

📖 [Full pipeline details →](1_knowledge_distillation/README.md)

---

## Pillar 2 — Glucose Forecasting

**Goal:** Predict blood glucose **2 hours ahead** from sparse fingerstick data (6–8 readings/day) — far harder than CGM-based forecasting.

**Approach:**
- **Target:** Δglucose (delta), not absolute value → avoids mean-reversion collapse
- **Features:** ~72 engineered features (IOB, COB, acceleration, AUC, circadian ISF, per-user sensitivities)
- **Models:** LightGBM + XGBoost ensemble, with asymmetric sample weights (hypo ×5, severe ×15)
- **Physics safety:** hard clip [40,400] + monotonic constraints (insulin↑→glucose↓, carbs↑→glucose↑)
- **Data:** 200 synthetic patients × 180 days, 5 clinical sub-types (T1 MDI, T1 pump, T2 oral, T2 basal, T2 intensive)

**Results:** Zero impossible predictions, Zone D held low, hypo-detection ≈ 90%

📖 [Forecasting notebook →](2_glucose_forecasting/glucose_prediction_vF.ipynb)

---

## Pillar 3 — Meal Scan

**Goal:** Classify a food photo into one of 11 categories and return the predicted carbohydrate content.

**Model:** EfficientNet fine-tuned on Food-101 (restricted to 11 relevant classes)  
**Export formats:** PyTorch `.pth` → ONNX → TFLite (float32 + fp16)  
**Classes:** baklawa, chourba, couscous, egg, french fries, pizza, sfenje, spaghetti, sushi, tajin_zitoun, tiramisu

📖 [Inference demo →](3_meal_scan/meal_classifier_inference.ipynb)

---

## Setup

```bash
# Clone
git clone https://github.com/yacineabh/DiabAI.git
cd DiabAI

# Each pillar has its own requirements — see the pillar README
# Example for glucose forecasting:
pip install lightgbm xgboost pandas numpy scikit-learn matplotlib seaborn scipy

# For LoRA fine-tuning (run on Kaggle or Colab with GPU):
pip install unsloth trl transformers accelerate datasets peft bitsandbytes
```

**Environment variable required for LLM-based features:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Project Context

This project was developed as part of a 2CS (2nd year Computer Science) final project to build an intelligent diabetes companion app. The ML components described here power the app's backend inference.

---

## Disclaimer

All insulin-related outputs from this system are **mathematical estimations based on historical patterns — not medical prescriptions**. Always consult a qualified healthcare provider before making changes to any diabetes treatment plan.
