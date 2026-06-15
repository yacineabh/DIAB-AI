# Pillar 3 — Meal Scan

## Goal

Automatically classify a food photo into one of **11 specific classes** and estimate the corresponding carbohydrate content to assist in rapid insulin dose calculation.

---

## Directory Structure

```
3_meal_scan/
├── README.md                          ← This file
├── mealsScanTraining.ipynb            ← Main training notebook (PyTorch)
├── meal_classifier_inference.ipynb    ← Clean inference demo (ONNX & TFLite)
│
└── models/                            ← Trained models and figures
    ├── best_model.pth                 ← Raw PyTorch weights
    ├── class_names.json               ← List of canonical class labels
    ├── food_classifier.onnx           ← Exported ONNX Runtime model
    ├── food_classifier.tflite         ← Exported TFLite float32 model
    ├── food_classifier_fp16.tflite    ← Exported TFLite float16 model
    ├── confusion_matrix.png           ← Evaluation confusion matrix
    └── training_curves.png            ← Training vs Validation loss/accuracy curves
```

---

## Model Architecture and Training

- **Base Network**: EfficientNet-B0 (pretrained on ImageNet)
- **Dataset**: Food-101 dataset restricted to 11 classes, featuring local/diabetic-frequent foods:
  1. `baklawa`
  2. `chourba`
  3. `couscous`
  4. `egg`
  5. `french fries`
  6. `pizza`
  7. `sfenje`
  8. `spaghetti`
  9. `sushi`
  10. `tajin_zitoun`
  11. `tiramisu`
- **Output Layer**: Fully connected classification head mapping to the 11 classes.

---

## Model Formats and Deployment

To run models efficiently on both mobile backends and devices:
1. **PyTorch (`best_model.pth`)**: Used for the core training code and testing in research environment.
2. **ONNX (`food_classifier.onnx`)**: Exported from PyTorch. Ideal for server-side deployments using `onnxruntime` for fast CPU/GPU inference.
3. **TFLite (`food_classifier.tflite` / `food_classifier_fp16.tflite`)**: Formatted for mobile application backends or directly on Android/iOS devices. The `fp16` model offers half-precision quantization, reducing model size by 50% (~8 MB) with negligible impact on accuracy.

---

## How to Run Inference

Inference is fully detailed in [meal_classifier_inference.ipynb](meal_classifier_inference.ipynb).

### Requirements

```bash
pip install onnxruntime tensorflow numpy pillow
```

### Running ONNX prediction:
```python
import onnxruntime as ort
from PIL import Image
import numpy as np

session = ort.InferenceSession('models/food_classifier.onnx')
# Preprocess image to 224x224 and normalize using ImageNet stats, then:
# logits = session.run(None, {'input': processed_image_array})
```

### Running TFLite prediction:
```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='models/food_classifier.tflite')
interpreter.allocate_tensors()
# Set input tensor and invoke:
# interpreter.set_tensor(input_details[0]['index'], processed_image_array)
# interpreter.invoke()
```
