# CSC8851 Spring 2026 — Final Project: Food Nutrition Estimator 🍕

> **Team**: Bryan Chau  
> **Course**: CSC8851 — Deep Learning  
> **Semester**: Spring 2026

Hey! This README walks through everything I built for the final project — from data exploration all the way to a live demo app. If you just want to **run the app** without re-training anything, jump straight to [Quick Start](#quick-start-run-the-app-without-retraining).

---

## What This Project Does

We built a system that takes a **photo of a food dish** and predicts:
- 🔥 Calories (kcal)
- 🥑 Fat (g)
- 🍗 Protein (g)
- 🍞 Carbohydrates (g)

The final pipeline stacks four models together:

```
Input photo
    │
    ├─► YOLOv8-seg          → segments the food region (mask)
    ├─► EfficientNet-B0     → identifies what food it is (Phase 5, optional)
    ├─► MiDaS_small         → estimates depth (how much food is on the plate)
    └─► NutritionMLP        → predicts calories/fat/protein/carbs
              + MC Dropout  → gives uncertainty estimate (±)
```

---

## Project Phases Overview

| Phase | Notebook | What I did |
|---|---|---|
| **1** | `01_data_exploration.ipynb` | Explored Nutrition5K — checked class distributions, plotted calorie histograms, found the dataset covers ~5K overhead dish photos |
| **2** | `02_dataset_and_dataloader.ipynb` | Built a custom PyTorch `Dataset` + `DataLoader` for Nutrition5K with train/val/test splits |
| **3** | `03_model.ipynb` | Fine-tuned **ResNet-50** end-to-end for nutrition regression — this is the baseline |
| **4** | `04_yolo_depth_pipeline.ipynb` | Built the full pipeline: YOLOv8-seg → MiDaS depth → 9-feature MLP with MC Dropout uncertainty |
| **5** | `05_food_classifier.ipynb` | Fine-tuned **EfficientNet-B0** on Food-101 (101 food categories) so the app can actually name what food it sees |

Phases 3–5 were trained on **Kaggle** (T4 GPU). Phases 1–2 can run locally.

---

## Project Structure

```
csc8851_spring26_finalProject/
│
├── app/
│   └── app.py                  ← Gradio demo (the thing you run!)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_dataset_and_dataloader.ipynb
│   ├── 03_model.ipynb           ← ResNet-50 baseline (Phase 3)
│   ├── 04_yolo_depth_pipeline.ipynb  ← Main pipeline (Phase 4)
│   └── 05_food_classifier.ipynb      ← Food type detection (Phase 5)
│
├── models/                      ← Put downloaded checkpoints here!
│   ├── best_mlp.pt              ← Phase 4 MLP weights (from Kaggle)
│   ├── best_model.pt            ← Phase 3 ResNet-50 weights (from Kaggle)
│   ├── mlp_feat_stats.npz       ← Feature normalisation stats (from Kaggle)
│   ├── best_food_classifier.pt  ← Phase 5 EfficientNet weights (optional)
│   └── food101_labels.json      ← Food-101 class names (optional)
│
├── data/                        ← Nutrition5K dataset goes here (local only)
├── src/                         ← Helper utilities
└── yolov8n-seg.pt               ← YOLOv8n-seg weights (auto-downloaded by ultralytics)
```

---

## Quick Start — Run the App Without Retraining

### Step 1: Clone & set up environment

```bash
git clone https://github.com/bryanchau11/csc8851_spring26_finalProject.git
cd csc8851_spring26_finalProject

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics timm pillow opencv-python
pip install gradio==3.50.2 gradio_client==0.6.1 huggingface_hub==0.20.3
```

> **Note for Mac M1/M2/M3**: The `torch` install above works fine. MPS acceleration is enabled automatically.  
> **Note for GPU machines**: Replace the torch install URL with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 3: Download the model checkpoints from Kaggle

You need these files from the Kaggle notebook outputs:

| File | From which Kaggle notebook | Required? |
|---|---|---|
| `best_mlp.pt` | `04_yolo_depth_pipeline` output | **Yes** |
| `mlp_feat_stats.npz` | `04_yolo_depth_pipeline` output | **Yes** |
| `best_model.pt` | `03_model` output | Optional (Phase 3 fallback) |
| `best_food_classifier.pt` | `05_food_classifier` output | Optional (food naming) |
| `food101_labels.json` | `05_food_classifier` output | Optional (food naming) |

Place all downloaded files inside the `models/` folder. The minimum to run Phase 4 is:
```
models/best_mlp.pt
models/mlp_feat_stats.npz
```

### Step 4: Run the app

```bash
source .venv/bin/activate    # if not already active
python app/app.py
```

You'll see something like:
```
✓ YOLO food classes: ['apple', 'banana', 'bowl', ...]
✓ Phase 4 pipeline loaded  (targets: ['calories', 'fat', 'protein', 'carbs'])
✓ Phase 5 food classifier loaded  (101 classes)
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx.gradio.live
```

Open `http://127.0.0.1:7860` in your browser, upload any food photo, click **Predict**.

---

## What the App Shows

- **Calorie / macro prediction** with uncertainty (e.g. `calories: 412.3 ± 48.2`)
- **Food type label** if Phase 5 classifier is loaded (e.g. `🍕 Detected food: Pizza (87% confidence)`)
- **Top-3 food alternatives** in the ingredient table
- Which pipeline is active (Phase 4 preferred, falls back to Phase 3 ResNet if MLP not found)

---

## How to Retrain (if you want to)

All notebooks are designed to run on **Kaggle** (free T4 GPU). Upload the notebook, connect the Nutrition5K dataset, and run all cells. Each notebook has a **checkpoint-skip guard** — if the output file already exists it won't re-train from scratch.

### Kaggle dataset needed
- **Nutrition5K** — search "Nutrition5K" in Kaggle datasets, or use the official one from Google Research.
- **Food-101** — `torchvision.datasets.Food101` auto-downloads it (Phase 5 only, ~5 GB).

### Recommended training order
```
03_model.ipynb          → ~1-2 hours on T4
04_yolo_depth_pipeline.ipynb  → ~30 min on T4 (features are cached)
05_food_classifier.ipynb      → ~3-4 hours on T4 (20 epochs, Food-101)
```

---

## Model Details

### Phase 4 — NutritionMLP

```
Input: 9 features
  - mask_area (fraction of image covered by food)
  - depth mean/std/median/max (full image)
  - masked depth mean/std/median/max (food region only)

Architecture: Linear(9→128) → BN → ReLU → Dropout(0.2)
              Linear(128→64) → BN → ReLU → Dropout(0.2)
              Linear(64→32)  → BN → ReLU → Dropout(0.2)
              Linear(32→4)   → [calories, fat, protein, carbs]

Uncertainty: MC Dropout (30 forward passes, fixed seed=42)
             Reports mean ± 2σ
```

### Phase 5 — FoodClassifier

```
Backbone: EfficientNet-B0 (pretrained ImageNet, all layers fine-tuned)
Head: Dropout(0.3) → Linear(1280→512) → BN → ReLU → Dropout(0.15) → Linear(512→101)
Dataset: Food-101 (75,750 train / 25,250 test images, 101 classes)
Expected accuracy: ~83-87% top-1, ~96% top-5
Training: AdamW, differential LR (backbone 5e-5, head 3e-4), Cosine LR, Mixup α=0.4
```

---

## Known Limitations

1. **Calories are geometry-based, not food-type-based** — the MLP uses depth + mask area, so it can't tell pizza from salad. The Phase 5 classifier adds the food label but doesn't yet feed back into the calorie prediction.

2. **Nutrition5K is a lab dataset** — all images are Google cafeteria dishes photographed overhead in controlled lighting. Real-world photos (restaurant, home) will be less accurate.

3. **Food-101 has 101 categories** — common foods are well covered (pizza, ramen, sushi, burgers, salads, tacos, etc.) but regional/ethnic foods may not be recognised. See below for dataset upgrades.

---

## Upgrading Food Recognition (Future Work)

If you want to train on more food categories for better real-world coverage:

| Dataset | Classes | Best for | Link |
|---|---|---|---|
| Food-101 *(current)* | 101 | Common Western foods | `torchvision.datasets.Food101` |
| iFood-2019 | 251 | Fine-grained categories | Kaggle: `horizonhardik/ifood-2019-fgvc6` |
| UECFOOD-256 | 256 | Japanese & Asian cuisine | http://foodcam.mobi/dataset256.html |
| Food2K | 2,000 | Maximum variety | Needs a big GPU |

To switch datasets: update `CFG['num_classes']` in `05_food_classifier.ipynb` and re-run.

---

## Dependencies

```
torch>=2.0
torchvision>=0.15
ultralytics          # YOLOv8
timm                 # EfficientNet-B0
opencv-python
pillow
gradio==3.50.2
gradio_client==0.6.1
huggingface_hub==0.20.3
scikit-learn         # top_k_accuracy_score (training only)
numpy
matplotlib
```

---

## Quick Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: timm` | `pip install timm` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `UnpicklingError` when loading `.pt` | Already handled — we use `weights_only=False` in all `torch.load()` calls |
| App says "No model checkpoint found" | Make sure `best_mlp.pt` and `mlp_feat_stats.npz` are in `models/` |
| Food classifier shows "Unknown dish" | Either `best_food_classifier.pt` is missing or the food is not in Food-101's 101 classes |
| Gradio version conflicts | Stick to the exact versions: `gradio==3.50.2 gradio_client==0.6.1 huggingface_hub==0.20.3` |

---

*Built for CSC8851 Spring 2026 — thanks for checking it out!* 🙂
