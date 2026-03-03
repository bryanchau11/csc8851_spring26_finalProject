"""
Nutrition5K — Gradio Demo App  (Phase 4 pipeline)

Two inference modes (auto-selected based on available checkpoints):
  Phase 4 (preferred): YOLOv8-seg → MiDaS depth → MLP regression + MC uncertainty
  Phase 3 (fallback) : ResNet-50 direct image → nutrition regression

Usage:
    pip install gradio torch torchvision ultralytics timm pillow opencv-python
    python app/app.py

Required files in models/:
    best_mlp.pt          ← Phase 4 MLP  (from Kaggle output of 04_yolo_depth_pipeline)
    mlp_feat_stats.npz   ← feature normalisation stats
    best_model.pt        ← Phase 3 ResNet fallback
"""

import os, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import gradio as gr
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
P4_CKPT    = os.path.join(MODELS_DIR, 'best_mlp.pt')
P4_STATS   = os.path.join(MODELS_DIR, 'mlp_feat_stats.npz')
P3_CKPT    = os.path.join(MODELS_DIR, 'best_model.pt')
IMG_SIZE   = 224
MC_SAMPLES = 30

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else
                      'cpu')

# ── Food class filtering — built dynamically from YOLO's own class names ───────
# Keywords that indicate food or food containers; non-food items like
# "dining table", "fork", "knife", "spoon" are excluded.
_FOOD_KEYWORDS = {
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'bowl', 'cup', 'wine glass',
    'bottle', 'salad', 'egg', 'meat', 'rice', 'bread', 'cheese',
    'burger', 'sushi', 'noodle', 'soup', 'fruit', 'vegetable',
}
_NON_FOOD = {'dining table', 'fork', 'knife', 'spoon', 'chopsticks'}

def _build_food_classes(yolo):
    """Return set of class IDs whose names match food keywords."""
    food_ids = set()
    for cid, name in yolo.names.items():
        n = name.lower()
        if n in _NON_FOOD:
            continue
        if any(kw in n for kw in _FOOD_KEYWORDS):
            food_ids.add(int(cid))
    return food_ids

FOOD_CLASSES = set()   # populated after yolo_model is loaded

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

class NutritionMLP(nn.Module):
    def __init__(self, in_feats: int, num_targets: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),       nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_targets),
        )
    def forward(self, x): return self.net(x)


class NutritionEstimator(nn.Module):
    """Phase 3 ResNet-50 model (fallback)."""
    def __init__(self, num_targets: int):
        super().__init__()
        base = tv_models.resnet50(weights=None)
        in_f = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_targets),
        )
    def forward(self, x): return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────────────────────────────────────

pipeline_mode = None   # 'phase4' or 'phase3'
mlp = yolo_model = midas_model = midas_transform = None
feat_mean = feat_std = None
target_cols = ['calories', 'fat', 'protein', 'carbs']

# ── Try Phase 4 ───────────────────────────────────────────────────────────────
if os.path.isfile(P4_CKPT) and os.path.isfile(P4_STATS):
    try:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO('yolov8n-seg.pt')
        FOOD_CLASSES.update(_build_food_classes(yolo_model))
        print(f'✓ YOLO food classes: {[yolo_model.names[i] for i in sorted(FOOD_CLASSES)]}')

        stats = np.load(P4_STATS, allow_pickle=True)
        feat_mean   = stats['feat_mean'].astype(np.float32)
        feat_std    = stats['feat_std'].astype(np.float32)
        target_cols = list(stats['target_cols'])

        ckpt = torch.load(P4_CKPT, map_location=device, weights_only=False)
        target_cols = ckpt.get('target_cols', target_cols)
        mlp = NutritionMLP(in_feats=9, num_targets=len(target_cols)).to(device)
        mlp.load_state_dict(ckpt['model_state_dict'])

        # Lazy MiDaS
        midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        midas_model.to(device).eval()
        _t = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        midas_transform = _t.small_transform

        pipeline_mode = 'phase4'
        print(f'✓ Phase 4 pipeline loaded  (targets: {target_cols})')
    except Exception as e:
        print(f'⚠ Phase 4 load failed ({e}) — trying Phase 3 fallback')

# ── Try Phase 3 fallback ─────────────────────────────────────────────────────
if pipeline_mode is None:
    if os.path.isfile(P3_CKPT):
        ckpt = torch.load(P3_CKPT, map_location=device, weights_only=False)
        target_cols = ckpt.get('target_cols', target_cols)
        p3_model = NutritionEstimator(num_targets=len(target_cols)).to(device)
        p3_model.load_state_dict(ckpt['model_state_dict'])
        p3_model.eval()
        pipeline_mode = 'phase3'
        print(f'✓ Phase 3 ResNet model loaded  (targets: {target_cols})')
    else:
        print('⚠ No checkpoint found. Place best_mlp.pt + mlp_feat_stats.npz '
              '(or best_model.pt) in models/')


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def _get_food_mask(pil_img: Image.Image):
    """Returns (mask np.ndarray, detected_items list[dict{name, area_ratio}])."""
    img_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    # Circular centre fallback
    fallback = np.zeros((H, W), dtype=np.float32)
    cv2.circle(fallback, (W // 2, H // 2), int(min(H, W) * 0.3), 1.0, -1)

    if yolo_model is None:
        return cv2.resize(fallback, (IMG_SIZE, IMG_SIZE)), []

    results  = yolo_model(img_bgr, verbose=False)[0]
    if results.masks is None or len(results.masks) == 0:
        return cv2.resize(fallback, (IMG_SIZE, IMG_SIZE)), []

    classes  = results.boxes.cls.cpu().numpy().astype(int)
    confs    = results.boxes.conf.cpu().numpy()
    food_idx = [i for i, c in enumerate(classes) if c in FOOD_CLASSES]

    # If no recognised food items detected, use centre-crop fallback with no items listed
    if not food_idx:
        return cv2.resize(fallback, (IMG_SIZE, IMG_SIZE)), []

    combined = np.zeros((H, W), dtype=np.float32)
    item_masks = []
    for i in food_idx:
        m = cv2.resize(results.masks[i].data[0].cpu().numpy(), (W, H))
        m_bin = (m > 0.5).astype(np.float32)
        combined = np.maximum(combined, m_bin)
        item_masks.append({
            'name': yolo_model.names.get(int(classes[i]), f'Item {i+1}').replace('_', ' ').title(),
            'conf': float(confs[i]),
            'area': float(m_bin.mean()),
        })

    combined = (combined > 0.5).astype(np.float32)
    if combined.sum() < 100:
        combined = fallback
        item_masks = []

    return cv2.resize(combined, (IMG_SIZE, IMG_SIZE)), item_masks


def _get_depth_map(pil_img: Image.Image) -> np.ndarray:
    if midas_model is None:
        return np.full((IMG_SIZE, IMG_SIZE), 0.5, dtype=np.float32)
    try:
        img_rgb = np.array(pil_img.convert('RGB'))
        inp     = midas_transform(img_rgb).to(device)
        with torch.no_grad():
            pred = midas_model(inp).squeeze().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return cv2.resize(pred, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    except Exception:
        return np.full((IMG_SIZE, IMG_SIZE), 0.5, dtype=np.float32)


def _extract_features(pil_img: Image.Image):
    """Returns (feature_vector np.ndarray shape (9,), detected_items list)."""
    mask, items = _get_food_mask(pil_img)
    depth = _get_depth_map(pil_img)

    mask_area = mask.mean()
    d_mean, d_std = depth.mean(), depth.std()
    d_med,  d_max = float(np.median(depth)), depth.max()

    masked = depth[mask > 0.5]
    if len(masked) == 0:
        masked = depth.flatten()
    md_mean, md_std = masked.mean(), masked.std()
    md_med,  md_max = float(np.median(masked)), masked.max()

    feat = np.array([mask_area,
                     d_mean, d_std, d_med, d_max,
                     md_mean, md_std, md_med, md_max], dtype=np.float32)
    return ((feat - feat_mean) / feat_std).flatten(), items  # shape (9,)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    if image is None:
        return 'Please upload an image.', None, []

    if pipeline_mode is None:
        return ('No model checkpoint found. Download best_mlp.pt + mlp_feat_stats.npz '
                'from Kaggle output and place them in models/.'), None, []

    if pipeline_mode == 'phase4':
        feat, items = _extract_features(image)
        x    = torch.tensor(feat).unsqueeze(0).to(device)

        # MC Dropout uncertainty — keep Dropout active but BatchNorm in eval
        # Use fixed seed so same image always gives same result
        mlp.eval()
        for m in mlp.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        torch.manual_seed(42)
        preds_mc = []
        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                preds_mc.append(mlp(x).squeeze(0).cpu().numpy())
        # Reset back to full eval so next call starts clean
        mlp.eval()
        preds_mc = np.stack(preds_mc)
        mean_pred = preds_mc.mean(0)
        std_pred  = preds_mc.std(0)

        # Find calories index for per-item breakdown
        cal_idx = next((i for i, c in enumerate(target_cols) if 'cal' in c.lower()), 0)
        total_cal = float(mean_pred[cal_idx])

        # Build ingredient table (proportional calorie split by mask area)
        if items:
            total_area = sum(it['area'] for it in items) or 1.0
            ingredient_rows = []
            for it in items:
                frac = it['area'] / total_area
                item_cal = total_cal * frac
                ingredient_rows.append([it['name'], f"{item_cal:.0f}", f"{it['conf']*100:.0f}%"])
        else:
            ingredient_rows = [['Unknown dish (YOLO-COCO has no matching class)', f"{total_cal:.0f}", '—']]

        result_json = {col: f'{mu:.1f} ± {sig*2:.1f}' for col, mu, sig in zip(target_cols, mean_pred, std_pred)}
        mode_note = '_Pipeline: YOLOv8-seg → MiDaS depth → MLP + MC Dropout_'

    else:  # phase3
        img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            mean_pred = p3_model(img_t).squeeze(0).cpu().numpy()
        result_json = {col: f'{v:.1f}' for col, v in zip(target_cols, mean_pred)}
        mode_note  = '_Pipeline: ResNet-50 direct regression (Phase 3 baseline)_'
        ingredient_rows = [['N/A (ResNet baseline — no segmentation)', '—', '—']]

    label_txt = '\n'.join([f'{col}: {result_json[col]}' for col in target_cols])
    label_txt += f'\n\n{mode_note}\n> Units: kcal for calories, grams for macros'
    return label_txt, result_json, ingredient_rows


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

mode_label = {
    'phase4': '🔬 Phase 4 — YOLO + Depth + MLP (with uncertainty)',
    'phase3': '📊 Phase 3 — ResNet-50 baseline',
    None:     '⚠ No model loaded',
}

with gr.Blocks(title='Nutrition5K Estimator') as demo:
    gr.Markdown(
        f'# 🥗 Nutrition5K — Dish Nutrition Estimator\n'
        f'Upload a photo of a food dish to predict **calories, fat, protein, and carbohydrates**.\n\n'
        f'**Active pipeline**: {mode_label.get(pipeline_mode, "unknown")}'
    )
    with gr.Row():
        img_input = gr.Image(type='pil', label='Upload Dish Image')
        with gr.Column():
            text_out = gr.Markdown(label='Nutrition Prediction')
            json_out = gr.JSON(label='Raw values')

    gr.Markdown('### 🥦 Detected Ingredients & Calorie Breakdown')
    ingredient_table = gr.Dataframe(
        headers=['Ingredient', 'Est. Calories (kcal)', 'Confidence'],
        datatype=['str', 'str', 'str'],
        label='YOLO-detected items (only COCO classes: bowl, apple, banana, sandwich, pizza, etc. — most Nutrition5K dishes show as unknown)',
        interactive=False,
    )

    predict_btn = gr.Button('Predict', variant='primary')
    predict_btn.click(fn=predict, inputs=img_input, outputs=[text_out, json_out, ingredient_table])

    gr.Markdown(
        '> **Note**: Best results with overhead-style plated dish photos '
        '(matching the Nutrition5K training distribution).'
    )

if __name__ == '__main__':
    demo.launch(share=True)
