# Pillar 2 — Glucose Forecasting

## Goal

Predict blood glucose **2 hours ahead** from sparse patient timeline logs. Unlike continuous glucose monitor (CGM) forecasting, sparse logging (6–8 fingersticks/day) is highly dynamic and challenging because there is no continuous signal.

---

## Directory Structure

```
2_glucose_forecasting/
├── README.md                      ← This file
├── glucose_prediction_vF.ipynb    ← Main forecasting notebook (model training & evaluation)
│
├── data/                          ← Generated datasets
│   ├── patient_profiles.csv       ← Patient attributes (diabetes type, regimen, ISF, ICR)
│   ├── glucose_readings.csv       ← Raw fingerstick glucose values
│   ├── insulin_doses.csv          ← Rapid and long-acting dose timestamps
│   ├── meals.csv                  ← Carbohydrate intake events
│   ├── activities.csv             ← Exercise duration and intensity
│   └── features.csv               ← Large pre-compiled features file
│
└── models/                        ← Trained models and stats
    ├── glucose_lgbm.txt           ← Final LightGBM model weights
    ├── glucose_xgb.json           ← Final XGBoost model weights
    ├── model_config.json          ← Hyperparameters and configuration
    ├── feature_cols.pkl           ← Pickled feature column names
    └── feature_stats.pkl          ← Feature statistics (mean/std) for normalization
```

---

## Machine Learning Pipeline

### 1. Advanced Feature Engineering
Since raw logs are discrete events, the notebook constructs continuous physiological states from time-series logs:
- **Insulin on Board (IOB)**: Decaying exponential profile modeling active insulin remaining in the body (using a 4-hour clearance half-life).
- **Carbs on Board (COB)**: Decaying exponential profile modeling glucose absorption rates.
- **Glycemic Trajectory**: Multi-timepoint acceleration, velocity, and area under the curve (AUC).
- **Circadian Rhythms**: Time-of-day feature sine/cosine projections mapping diurnal variations in insulin sensitivity (ISF).

### 2. Clinical Safety Guardrails
Standard regression models collapse to the mean value when faced with high noise, predicting safe "average" values and completely missing hypoglycemic drops. To solve this, the pipeline enforces:
- **Asymmetric Sample Weights**: Penalizes errors during low glucose events (Hypo weighted x5, Severe Hypo weighted x15) to force the models to accurately fit safety-critical drops.
- **Monotonic Physics Constraints**: Enforces physiological boundaries (e.g., higher insulin must mathematically drive predicted glucose down, higher carbs must drive it up).
- **Output Clipping**: Safety-clips final predictions to range `[40, 400]` mg/dL.

### 3. Ensemble Modeling
Uses a weighted ensemble of **LightGBM** (highly robust to noise) and **XGBoost** (highly sensitive to non-linear interaction terms) to output the final forecast.

---

## Results and Evaluation

- **Hypoglycemia Detection Rate**: ~90% of hypoglycemic events predicted 2 hours in advance.
- **Zone D Minimize**: Zero clinically dangerous errors (predicting high when the patient is actually crashing).
- **Feature Importance**: Features like `IOB`, recent glucose trend velocity, and meal carb sizes are shown to drive the predictions.

---

## How to Run

1. Install the forecasting dependencies:
   ```bash
   pip install lightgbm xgboost pandas numpy scikit-learn matplotlib seaborn scipy
   ```
2. Open the main Jupyter notebook and execute the cells:
   ```bash
   jupyter notebook glucose_prediction_vF.ipynb
   ```
