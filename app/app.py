"""
Nutrition5K — Gradio Demo App  (Phase 4 + 5 + 6 pipeline)

Inference priority (auto-selected based on available checkpoints):
  Phase 6 (preferred): weight-first — predict weight (g) → × dataset constants → nutrition
  Phase 4 (fallback) : YOLO + MiDaS depth → MLP direct regression + MC uncertainty
  Phase 3 (fallback) : ResNet-50 direct image → nutrition regression

Usage:
    pip install gradio torch torchvision ultralytics timm pillow opencv-python
    python app/app.py

Required files in models/:
    best_weight_mlp.pt       ← Phase 6 WeightMLP  (06_weight_prediction.ipynb)
    nutrition_constants.json ← calorie density constants (kcal/g, fat/g, ...)
    weight_feat_stats.npz    ← Phase 6 feature normalisation stats
    best_mlp.pt              ← Phase 4 MLP  (fallback)
    mlp_feat_stats.npz       ← Phase 4 feature normalisation stats
    best_model.pt            ← Phase 3 ResNet fallback
"""

import os, json, warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import gradio as gr
warnings.filterwarnings('ignore')

MODELS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
# Phase 6 — weight-first (preferred)
P6_CKPT       = os.path.join(MODELS_DIR, 'best_weight_mlp.pt')
P6_CONST      = os.path.join(MODELS_DIR, 'nutrition_constants.json')
P6_STATS      = os.path.join(MODELS_DIR, 'weight_feat_stats.npz')
# Phase 4 — direct regression (fallback)
P4_CKPT       = os.path.join(MODELS_DIR, 'best_mlp.pt')
P4_STATS      = os.path.join(MODELS_DIR, 'mlp_feat_stats.npz')
# Phase 3 — ResNet baseline (fallback)
P3_CKPT       = os.path.join(MODELS_DIR, 'best_model.pt')
# Phase 5 — food classifier (optional, adds food-type label)
FOOD_CLF_CKPT = os.path.join(MODELS_DIR, 'best_food_classifier.pt')
FOOD_CLF_LBLS = os.path.join(MODELS_DIR, 'food101_labels.json')
IMG_SIZE      = 224
MC_SAMPLES    = 30

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
    """Phase 4 — predicts [calories, fat, protein, carbs] directly."""
    def __init__(self, in_feats: int, num_targets: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),       nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, num_targets),
        )
    def forward(self, x): return self.net(x)


class WeightMLP(nn.Module):
    """Phase 6 — predicts dish weight in grams (1 output).
    Nutrition = predicted_weight × dataset_constants."""
    def __init__(self, in_feats: int = 9, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),       nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),       nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)   # (B,) scalar


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


class FoodClassifier(nn.Module):
    """Phase 5 — EfficientNet-B0 fine-tuned on Food-101 (101 classes)."""
    def __init__(self, num_classes: int = 101):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False,
                                          num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features   # 1280
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────────────────────────────────────

pipeline_mode = None   # 'phase6', 'phase4', or 'phase3'
mlp = weight_mlp_model = yolo_model = midas_model = midas_transform = None
feat_mean = feat_std = None
nutrition_constants = {}   # Phase 6 calorie-density constants
target_cols = ['calories', 'fat', 'protein', 'carbs']

# Phase 5 food classifier (optional — loaded if checkpoint exists)
food_clf        = None
food_clf_labels = []   # list[str] index → class name

# ── Try Phase 6 (weight-first approach) ──────────────────────────────────────
if os.path.isfile(P6_CKPT) and os.path.isfile(P6_CONST) and os.path.isfile(P6_STATS):
    try:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO('yolov8n-seg.pt')
        FOOD_CLASSES.update(_build_food_classes(yolo_model))
        print(f'✓ YOLO food classes: {[yolo_model.names[i] for i in sorted(FOOD_CLASSES)]}')

        stats = np.load(P6_STATS, allow_pickle=True)
        feat_mean = stats['feat_mean'].astype(np.float32)
        feat_std  = stats['feat_std'].astype(np.float32)

        with open(P6_CONST) as _f:
            nutrition_constants = json.load(_f)
        # Derive target_cols order from constant keys (strip _per_g)
        target_cols = [k.replace('_per_g', '') for k in nutrition_constants]

        ckpt = torch.load(P6_CKPT, map_location=device, weights_only=False)
        weight_mlp_model = WeightMLP(in_feats=9).to(device)
        weight_mlp_model.load_state_dict(ckpt['model_state_dict'])
        weight_mlp_model.eval()

        midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        midas_model.to(device).eval()
        _t = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        midas_transform = _t.small_transform

        pipeline_mode = 'phase6'
        print(f'✓ Phase 6 pipeline loaded  (weight-first, targets: {target_cols})')
        print(f'  Constants: { {k: round(v,4) for k,v in nutrition_constants.items()} }')
    except Exception as e:
        print(f'⚠ Phase 6 load failed ({e}) — trying Phase 4 fallback')

# ── Try Phase 4 ───────────────────────────────────────────────────────────────
if pipeline_mode is None and os.path.isfile(P4_CKPT) and os.path.isfile(P4_STATS):
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

# ── Try Phase 5 food classifier (optional — runs alongside Phase 4) ───────────
if os.path.isfile(FOOD_CLF_CKPT):
    try:
        import timm  # noqa: F401  (ensure timm is importable before instantiating)
        _clf_ckpt = torch.load(FOOD_CLF_CKPT, map_location=device, weights_only=False)
        _num_cls  = len(_clf_ckpt.get('class_names', []))
        if _num_cls == 0:
            _num_cls = 101
        food_clf = FoodClassifier(num_classes=_num_cls).to(device)
        food_clf.load_state_dict(_clf_ckpt['model_state_dict'])
        food_clf.eval()
        # Use class_names from checkpoint, or fall back to food101_labels.json
        if 'class_names' in _clf_ckpt:
            food_clf_labels = _clf_ckpt['class_names']
        elif os.path.isfile(FOOD_CLF_LBLS):
            with open(FOOD_CLF_LBLS) as _f:
                _lbl_map = json.load(_f)
            food_clf_labels = [_lbl_map[str(i)] for i in range(len(_lbl_map))]
        print(f'✓ Phase 5 food classifier loaded  ({_num_cls} classes)')
    except Exception as _e:
        print(f'⚠ Phase 5 classifier load failed ({_e}) — food type detection disabled')
else:
    print('ℹ Phase 5 food classifier not found — train 05_food_classifier.ipynb on Kaggle '
          'then download best_food_classifier.pt into models/')


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


def _classify_food_type(pil_img: Image.Image, mask: np.ndarray, top_k: int = 3):
    """
    Phase 5: Crop the food region and classify using EfficientNet-B0.
    Returns list of (display_name, confidence_float) sorted best-first.
    Returns [] if food_clf is not loaded.
    """
    if food_clf is None or not food_clf_labels:
        return []

    # ── Crop bounding box of the mask region ─────────────────────────────────
    H_img, W_img = np.array(pil_img.convert('RGB')).shape[:2]
    # Resize mask to match original image dimensions (mask may be 224×224)
    mask_full = cv2.resize(mask, (W_img, H_img))
    ys, xs    = np.where(mask_full > 0.5)
    if len(ys) > 0:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad_y  = max(4, int((y1 - y0) * 0.05))
        pad_x  = max(4, int((x1 - x0) * 0.05))
        y0 = max(0, y0 - pad_y);  y1 = min(H_img, y1 + pad_y)
        x0 = max(0, x0 - pad_x);  x1 = min(W_img, x1 + pad_x)
        crop = pil_img.crop((x0, y0, x1, y1))
    else:
        crop = pil_img  # no mask → use full image

    _clf_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = _clf_transform(crop.convert('RGB')).unsqueeze(0).to(device)

    food_clf.eval()
    with torch.no_grad():
        logits = food_clf(x).squeeze(0)
    probs  = F.softmax(logits, dim=0).cpu().numpy()
    topk_i = probs.argsort()[-top_k:][::-1]

    return [
        (food_clf_labels[int(i)].replace('_', ' ').title(), float(probs[int(i)]))
        for i in topk_i
    ]


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
    norm_feat = ((feat - feat_mean) / feat_std).flatten()  # shape (9,)
    return norm_feat, items, mask  # also return raw mask for food classification


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

    if pipeline_mode in ('phase6', 'phase4'):
        feat, items, raw_mask = _extract_features(image)
        x    = torch.tensor(feat).unsqueeze(0).to(device)

        if pipeline_mode == 'phase6':
            # ── Phase 6: predict weight → multiply by constants ────────────────
            weight_mlp_model.eval()
            for m in weight_mlp_model.modules():
                if isinstance(m, nn.Dropout): m.train()
            torch.manual_seed(42)
            weight_samples = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    weight_samples.append(weight_mlp_model(x).item())
            weight_mlp_model.eval()

            w_raw  = float(np.mean(weight_samples))
            w_std  = float(np.std(weight_samples))

            # Fallback: if model is very uncertain (domain shift from real photos),
            # use the Nutrition5K mean dish weight (~280g) instead of a near-zero prediction
            DATASET_MEAN_WEIGHT = 280.0   # kcal-weighted mean across Nutrition5K
            MIN_WEIGHT          = 50.0    # absolute floor — no recognisable dish is <50g
            _used_fallback = w_raw < MIN_WEIGHT or (w_std > abs(w_raw) and w_raw < DATASET_MEAN_WEIGHT)
            if _used_fallback:
                w_mean = DATASET_MEAN_WEIGHT
            else:
                w_mean = max(w_raw, MIN_WEIGHT)

            # Apply dataset constants: nutrition = weight × constant
            const_keys = list(nutrition_constants.keys())  # e.g. calories_per_g, ...
            mean_pred = np.array([w_mean * nutrition_constants[k] for k in const_keys],
                                 dtype=np.float32)
            std_pred  = np.array([w_std  * nutrition_constants[k] for k in const_keys],
                                 dtype=np.float32)
            # target_cols was set from const keys at load time
            _fallback_note = ' — low-confidence, used dataset mean' if _used_fallback else ''
            mode_extra = f'\n_Predicted weight: **{w_mean:.0f} ± {w_std*2:.0f}g** (MC Dropout, 30 samples{_fallback_note})_'
        else:
            # ── Phase 4: direct regression ─────────────────────────────────────
            mlp.eval()
            for m in mlp.modules():
                if isinstance(m, nn.Dropout): m.train()
            torch.manual_seed(42)
            preds_mc = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    preds_mc.append(mlp(x).squeeze(0).cpu().numpy())
            mlp.eval()
            preds_mc = np.stack(preds_mc)
            mean_pred = preds_mc.mean(0)
            std_pred  = preds_mc.std(0)
            mode_extra = ''

        # Find calories index for per-item breakdown
        cal_idx = next((i for i, c in enumerate(target_cols) if 'cal' in c.lower()), 0)
        total_cal = float(mean_pred[cal_idx])

        # Phase 5: classify food type from the image crop
        food_type_preds = _classify_food_type(image, raw_mask, top_k=3)
        detected_food   = food_type_preds[0][0] if food_type_preds else None
        food_conf       = food_type_preds[0][1] if food_type_preds else None

        # Build ingredient table (proportional calorie split by mask area)
        if items:
            total_area = sum(it['area'] for it in items) or 1.0
            ingredient_rows = []
            for it in items:
                frac = it['area'] / total_area
                item_cal = total_cal * frac
                # Override YOLO class name with food classifier result if available
                display_name = detected_food if detected_food else it['name']
                conf_str     = f"{food_conf*100:.0f}%" if food_conf else f"{it['conf']*100:.0f}%"
                ingredient_rows.append([display_name, f"{item_cal:.0f}", conf_str])
        else:
            if detected_food:
                ingredient_rows = [[detected_food, f"{total_cal:.0f}", f"{food_conf*100:.0f}%"]]
                # Show top-3 alternatives in extra rows
                for alt_name, alt_conf in food_type_preds[1:]:
                    ingredient_rows.append([f'  (alt) {alt_name}', '—', f"{alt_conf*100:.0f}%"])
            else:
                ingredient_rows = [['Unknown dish (YOLO-COCO has no matching class)', f"{total_cal:.0f}", '—']]

        result_json = {col: f'{mu:.1f} ± {sig*2:.1f}' for col, mu, sig in zip(target_cols, mean_pred, std_pred)}
        clf_note  = f'🍽 Detected food: **{detected_food}** ({food_conf*100:.0f}% confidence)\n\n' if detected_food else ''
        if pipeline_mode == 'phase6':
            mode_note = (f'{clf_note}'
                         f'_Pipeline: Phase 6 — YOLOv8-seg + MiDaS depth → WeightMLP → nutrition × constants_'
                         f'{mode_extra}')
        else:
            mode_note = f'{clf_note}_Pipeline: YOLOv8-seg → EfficientNet-B0 food type → MiDaS depth → MLP + MC Dropout_'

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
    'phase6': '🎯 Phase 6 — Weight-First: YOLO + MiDaS → WeightMLP → × dataset constants (professor spec)',
    'phase4': '🔬 Phase 4+5 — YOLO + EfficientNet food type + Depth + MLP direct regression',
    'phase3': '📊 Phase 3 — ResNet-50 baseline',
    None:     '⚠ No model loaded',
}

clf_label = '🍕 Phase 5 food classifier active' if food_clf is not None else '⚠ No food classifier (run 05_food_classifier.ipynb to enable food-type detection)'

with gr.Blocks(title='Nutrition5K Estimator') as demo:
    p6_note = ('> **Phase 6 active**: Dish weight is predicted first, then converted to calories/macros '
               'using calorie-density constants derived from Nutrition5K. '
               '(Professor spec §4.3)\n\n') if pipeline_mode == 'phase6' else ''
    gr.Markdown(
        f'# 🥗 Nutrition5K — Dish Nutrition Estimator\n'
        f'Upload **any real-world food photo** to get calories, fat, protein, and carbs.\n\n'
        f'{p6_note}'
        f'**Active pipeline**: {mode_label.get(pipeline_mode, "unknown")}  \n'
        f'**Food recognition**: {clf_label}'
    )
    with gr.Row():
        img_input = gr.Image(type='pil', label='Upload Dish Image')
        with gr.Column():
            text_out = gr.Markdown(label='Nutrition Prediction')
            json_out = gr.JSON(label='Raw values')

    gr.Markdown('### 🍽 Detected Food Type & Calorie Breakdown')
    ingredient_table = gr.Dataframe(
        headers=['Detected Food / Ingredient', 'Est. Calories (kcal)', 'Confidence'],
        datatype=['str', 'str', 'str'],
        label='Phase 5 food classifier (101 Food-101 classes) + YOLO segmentation',
        interactive=False,
    )

    predict_btn = gr.Button('Predict 🔍', variant='primary')
    predict_btn.click(fn=predict, inputs=img_input, outputs=[text_out, json_out, ingredient_table])

    gr.Markdown(
        '> **Tips**: Works best with a single dish centred in frame.  \n'
        '> Food-101 covers pizza, sushi, ramen, burgers, salads, pasta, tacos, and 94 more categories.  \n'
        '> For Asian & regional foods train on UECFOOD-256 or iFood-2019 (see notebook 05).  \n'
        '> **Phase 6**: Train `06_weight_prediction.ipynb` on Kaggle, download '
        '`best_weight_mlp.pt + nutrition_constants.json + weight_feat_stats.npz`, place in `models/`.'
    )

if __name__ == '__main__':
    demo.launch(share=True)
