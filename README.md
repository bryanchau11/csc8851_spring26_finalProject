# CSC8851 Spring 2026 Final Project - Food Nutrition Estimator 🍱

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.50.2-FF7A59)
![YOLOv8](https://img.shields.io/badge/YOLOv8-seg-111111)
![Course](https://img.shields.io/badge/CSC8851-Spring%202026-6A5ACD)

> Team: Bryan Chau, Katelyn Truong, Loc Giang  
> Course: CSC8851 Deep Learning  
> Semester: Spring 2026

This repository contains our end-to-end system for estimating nutrition from one food image.
At the current stage, the app supports the full multi-phase pipeline (Phase 3 + 4 + 5 + 6), with fallback logic when some checkpoints are missing.

If you just want to run the demo locally, go to [Quick Start](#quick-start-run-the-app-current-setup).

---

## 🚀 Current Project Stage

The app is no longer Phase-4-only. It now supports these runtime modes:

- `full`: Phase 6 (weight-first) + Phase 4 (direct regression) + Phase 5 (food classifier), with Phase 3 loaded for baseline comparison
- `phase6`: weight-first path only
- `phase4`: direct regression path only
- `phase3`: ResNet-50 fallback baseline

| Mode | Status | Description |
|---|---|---|
| `full` | 🟢 Best | Ensemble + classifier + priors |
| `phase6` | 🟡 Good | Weight-first with density constants |
| `phase4` | 🟡 Good | Direct nutrition regression |
| `phase3` | 🔵 Baseline | ResNet fallback |

In `full` mode, the app combines:

1. YOLOv8 segmentation + MiDaS depth -> 9 geometric features
2. WeightMLP (Phase 6) with MC Dropout
3. NutritionMLP (Phase 4) with MC Dropout
4. EfficientNet-B0 food classification (Phase 5, optional but recommended)
5. USDA/restaurant serving priors for food-aware scaling and ingredient edits

---

## 🗂️ Project Layout

```
csc8851_spring26_finalProject/
|
|- app/
|  |- app.py
|  |- core_models.py
|  |- pipeline.py
|  |- ui.py
|  `- usda.py
|
|- notebooks/
|  |- 01_data_exploration.ipynb
|  |- 02_dataset_and_dataloader.ipynb
|  |- 03_model.ipynb
|  |- 04_yolo_depth_pipeline.ipynb
|  |- 05_food_classifier.ipynb
|  |- 06_weight_prediction.ipynb
|  `- 07_kaggle_train_ingredient_yolov8_seg.ipynb
|
|- models/   (all checkpoints and metadata go here)
|- report/
|- data/
`- yolov8n-seg.pt
```

---

## ⚡ Quick Start (Run the App, Current Setup)

### 1) Environment

```bash
git clone https://github.com/bryanchau11/csc8851_spring26_finalProject.git
cd csc8851_spring26_finalProject

python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install packages

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics timm opencv-python pillow
pip install gradio==3.50.2 gradio_client==0.6.1 huggingface_hub==0.20.3
pip install numpy
```

> [!TIP]
> On Apple Silicon (M1/M2/M3), PyTorch will use MPS when available.

> [!NOTE]
> On CUDA machines, install the CUDA build of PyTorch from pytorch.org.

### 3) Put checkpoints in `models/`

The app will auto-detect what it can load.

| File | Used by | Required for mode |
|---|---|---|
| `best_weight_mlp.pt` | Phase 6 WeightMLP | `full` / `phase6` |
| `nutrition_constants.json` | Phase 6 nutrient conversion | `full` / `phase6` |
| `weight_feat_stats.npz` | Phase 6 feature normalization | `full` / `phase6` |
| `best_mlp.pt` | Phase 4 NutritionMLP | `full` / `phase4` |
| `mlp_feat_stats.npz` | Phase 4 feature normalization | `full` / `phase4` |
| `best_model.pt` | Phase 3 ResNet baseline | optional fallback/comparison |
| `best_food_classifier.pt` | Phase 5 dish classifier | optional (food naming + priors) |
| `food101_labels.json` | Phase 5 class names | optional fallback labels |
| `best_ingredient_yolov8_seg.pt` | ingredient/component segmentation | optional |
| `ingredient_labels.json` | ingredient label map | optional |

Minimum recommended to run the current best path:

```text
models/best_weight_mlp.pt
models/nutrition_constants.json
models/weight_feat_stats.npz
models/best_mlp.pt
models/mlp_feat_stats.npz
```

### 4) (Optional) USDA API key

The app works without this, but you can improve lookup quality and reduce misses:

```bash
export USDA_API_KEY=your_key_here
```

If not set, the code uses `DEMO_KEY` with cache/fallback logic.

> [!TIP]
> For repeated testing, keeping `models/usda_cache.json` helps reduce API calls and speeds up future runs.

### 5) Launch

```bash
python app/app.py
```

Open `http://127.0.0.1:7860` and test with a meal image.

---

## 👀 What You Should See in the App

- Active pipeline badge (`full`, `phase6`, `phase4`, or `phase3`)
- Calories + macros with uncertainty
- Dish label when the classifier is available/confident
- Ingredient table (with editable portions/units)
- Weight prediction details accordion
- GPU/compute stats accordion

---

## 📓 Notebook Phases (Current)

| Phase | Notebook | Purpose |
|---|---|---|
| 1 | `01_data_exploration.ipynb` | Nutrition5K exploration |
| 2 | `02_dataset_and_dataloader.ipynb` | Dataset/DataLoader prep |
| 3 | `03_model.ipynb` | ResNet-50 nutrition baseline |
| 4 | `04_yolo_depth_pipeline.ipynb` | YOLO + MiDaS + NutritionMLP |
| 5 | `05_food_classifier.ipynb` | Food-101 EfficientNet-B0 classifier |
| 6 | `06_weight_prediction.ipynb` | Weight-first model + constants |
| 7 | `07_kaggle_train_ingredient_yolov8_seg.ipynb` | Ingredient/component segmentation |

Suggested Kaggle training order for current app behavior:

1. `03_model.ipynb`
2. `04_yolo_depth_pipeline.ipynb`
3. `05_food_classifier.ipynb`
4. `06_weight_prediction.ipynb`
5. `07_kaggle_train_ingredient_yolov8_seg.ipynb` (optional)

---

## 🧩 Dependencies Used by the App

```text
torch
torchvision
ultralytics
timm
opencv-python
pillow
gradio==3.50.2
gradio_client==0.6.1
huggingface_hub==0.20.3
numpy
```

Training notebooks may additionally use common analysis packages (for example matplotlib/scikit-learn/pandas).

---

## 🛠️ Quick Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: timm` | `pip install timm` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `No model checkpoint found` at startup | Add at least one valid checkpoint set in `models/` (Phase 6 or Phase 4 or Phase 3) |
| Food label missing | `best_food_classifier.pt` is missing, failed to load, or confidence is too low |
| USDA lookup weak/missing | set `USDA_API_KEY`, then retry (cache is stored in `models/usda_cache.json`) |
| Gradio/version conflicts | keep `gradio==3.50.2`, `gradio_client==0.6.1`, `huggingface_hub==0.20.3` |

> [!WARNING]
> If the app starts but predictions look flat/off, the most common cause is a stats mismatch (`mlp_feat_stats.npz` or `weight_feat_stats.npz` from a different training run).

---

## ⚠️ Notes and Limitations

- Nutrition5K images are controlled overhead captures, so real-world photos can still cause domain shift.
- MiDaS depth is relative, not metric depth; Phase 6 handles this with serving anchors and conservative scaling.
- Food-101 has 101 classes, so long-tail regional foods can still be mislabeled.

---

Built for CSC8851 Spring 2026 🍜
