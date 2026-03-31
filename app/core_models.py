import os, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    from .usda import _load_usda_cache  # type: ignore
except Exception:
    from usda import _load_usda_cache  # type: ignore
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
ING_LBLS_JSON = os.path.join(MODELS_DIR, 'ingredient_labels.json')
IMG_SIZE      = 224
MC_SAMPLES    = 30

# If the dish classifier is unsure, don’t present a confidently-wrong label.
DISH_LABEL_MIN_CONF = 0.35

# FoodSeg103 contains many fruit categories. If the ingredient model says a
# single fruit dominates the image, prefer it over the Food-101 dish label.
FOODSEG103_FRUITS = {
    'apple', 'date', 'apricot', 'avocado', 'banana', 'strawberry', 'cherry',
    'blueberry', 'raspberry', 'mango', 'olives', 'peach', 'lemon', 'pear',
    'fig', 'pineapple', 'grape', 'kiwi', 'melon', 'orange', 'watermelon',
}


def _maybe_foodseg103_single_fruit_override(ingredient_items,
                                           min_dom_frac: float = 0.75,
                                           min_top_area: float = 0.12,
                                           min_conf: float = 0.15):
    """Return (name_title, conf_float) if a single FoodSeg103 fruit dominates."""
    if not ingredient_items:
        return None
    try:
        items = [dict(x) for x in ingredient_items if x]
        if not items:
            return None
        items.sort(key=lambda x: float(x.get('area', 0.0)), reverse=True)
        top = items[0]
        name = str(top.get('name', '')).strip().lower()
        if name not in FOODSEG103_FRUITS:
            return None
        top_area = float(top.get('area', 0.0))
        top_conf = float(top.get('conf', 0.0))
        if top_area < min_top_area or top_conf < min_conf:
            return None
        total_area = float(sum(float(it.get('area', 0.0)) for it in items) or 0.0)
        if total_area <= 0:
            return None
        if (top_area / total_area) < min_dom_frac:
            return None
        return name.title(), top_conf
    except Exception:
        return None
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else
                      'cpu')


def _load_ingredient_labels():
    """Load optional ingredient/component label mapping.

    Expected format in models/ingredient_labels.json:
      {"0": "rice", "1": "egg", ...}
    or
      ["rice", "egg", ...]  (index = class id)
    """
    if not os.path.isfile(ING_LBLS_JSON):
        return None
    try:
        with open(ING_LBLS_JSON) as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return {int(i): str(v) for i, v in enumerate(obj)}
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out or None
        return None
    except Exception:
        return None


INGREDIENT_LABELS = _load_ingredient_labels()


def _gpu_stats_md(timing=None) -> str:
    """Return a Markdown string with GPU/device info, per-stage timing, and training history."""
    lines = ['### ⚡ GPU / Compute Stats', '']

    # Device info
    if torch.cuda.is_available():
        idx   = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / 1024**3
        alloc = torch.cuda.memory_allocated(idx) / 1024**3
        resvd = torch.cuda.memory_reserved(idx)  / 1024**3
        free  = total - resvd
        lines += [
            f'**Device**: CUDA — {props.name}',
            f'**CUDA version**: {torch.version.cuda}  ·  '
            f'**cuDNN**: {torch.backends.cudnn.version()}',
            f'**VRAM**: {alloc:.2f} GB allocated · {resvd:.2f} GB reserved · '
            f'{free:.2f} GB free · {total:.2f} GB total',
            f'**SM count**: {props.multi_processor_count}  ·  '
            f'**Compute cap**: {props.major}.{props.minor}',
        ]
    elif torch.backends.mps.is_available():
        lines += [
            '**Device**: Apple Silicon MPS (Metal Performance Shaders)',
            '_VRAM is shared with system RAM — not separately reported by PyTorch._',
        ]
    else:
        lines += ['**Device**: CPU only (no GPU detected)']

    lines += [
        '',
        f'**PyTorch**: {torch.__version__}',
        '',
    ]

    # Training compute history (from training_stats.json saved by notebooks)
    _stats_path = os.path.join(MODELS_DIR, 'training_stats.json')
    if os.path.isfile(_stats_path):
        try:
            with open(_stats_path) as _sf:
                _ts = json.load(_sf)

            lines += [
                '---',
                '',
                '### 📊 Training Compute History (Kaggle)',
                '',
                '| Phase | Model | GPU | Params | Epochs | Best Metric | GPU Train Time |',
                '|---|---|---|---|---|---|---|',
            ]

            _phase_order = ['phase3', 'phase4', 'phase5', 'phase6']
            for _pk in _phase_order:
                if _pk not in _ts:
                    continue
                _p = _ts[_pk]
                _params_m = (f'{_p["params"]/1e6:.2f}M'
                             if isinstance(_p.get('params'), (int, float)) else '—')
                _ep_run = _p.get('epochs_run', '?')
                _ep_max = _p.get('max_epochs', '?')
                _epochs = (f'{_ep_run} / {_ep_max}'
                           if isinstance(_ep_run, int) else f'ckpt / {_ep_max}')
                # Best metric — phase-specific
                if _pk == 'phase3':
                    _mae = _p.get('val_cal_mae_kcal')
                    _metric = f'Cal MAE = {_mae} kcal' if _mae else f'val_loss = {_p.get("best_val_loss","—")}'
                elif _pk == 'phase4':
                    _metric = f'val_loss = {_p.get("best_val_loss","—")}'
                elif _pk == 'phase5':
                    _t1 = _p.get('best_val_top1_pct')
                    _t5 = _p.get('best_val_top5_pct')
                    _metric = (f'Top-1 = {_t1}%, Top-5 = {_t5}%'
                               if _t1 else f'val_. = {_p.get("best_val_loss","—")}')
                else:  # phase6
                    _mae = _p.get('best_val_mae_g')
                    _metric = f'Weight MAE = {_mae} g' if _mae else f'Huber = {_p.get("best_val_loss_huber","—")}'

                _gpu     = _p.get('gpu_name', _p.get('device', '—'))
                _t_hms   = _p.get('train_time_hms', '—')
                _t_s     = _p.get('train_time_s', 0)
                _t_str   = f'{_t_hms} ({_t_s:.0f}s)' if isinstance(_t_s, float) and _t_s > 5 else '—'
                _pname   = _p.get('phase_name', _pk)
                _model   = _p.get('model', '—')
                lines.append(
                    f'| {_pname} | {_model} | {_gpu} | {_params_m} | {_epochs} | {_metric} | {_t_str} |'
                )

            # Footer with when each phase was trained
            _dates = [f'**{_pk}**: {_ts[_pk]["trained_at"]}'
                      for _pk in _phase_order if _pk in _ts and 'trained_at' in _ts[_pk]]
            if _dates:
                lines += ['', '_Trained at: ' + ' · '.join(_dates) + '_']
        except Exception as _e:
            lines += [f'_Could not parse training_stats.json: {_e}_']
    else:
        lines += [
            '---',
            '',
            '### 📊 Training Compute History',
            '',
            '_`training_stats.json` not found in `models/`._  ',
            '_Run each training notebook on Kaggle, then download and place `training_stats.json` in `models/`._',
        ]

    lines += ['']

    # Inference timing
    if timing:
        lines += ['---', '', '**Inference timing (wall-clock):**', '']
        lines += ['| Stage | Time (ms) |', '|---|---|']
        for stage, ms in timing.items():
            lines.append(f'| {stage} | {ms:.1f} |')

    return '\n'.join(lines)


# Food class filtering — built dynamically from YOLO's own class names
# Keywords that indicate food or food containers; non-food items like
# "dining table", "fork", "knife", "spoon" are excluded.
_FOOD_KEYWORDS = {
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'bowl', 'cup', 'wine glass',
    'bottle', 'salad', 'egg', 'meat', 'rice', 'bread', 'cheese',
    'burger', 'sushi', 'noodle', 'soup', 'fruit', 'vegetable',
}
_NON_FOOD = {'dining table', 'fork', 'knife', 'spoon', 'chopsticks'}

# USDA FoodData Central API is implemented in app/usda.py


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


# Model definitions

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


# Load models at startup

pipeline_mode = None   # 'full', 'phase6', 'phase4', or 'phase3'
                       # 'full' = Phase 6 + Phase 4 + Phase 5 all running together
mlp = weight_mlp_model = yolo_model = midas_model = midas_transform = None
ingredient_yolo_model = None
_weight_log_target = False   # True when checkpoint was trained on log(w+1)
_weight_log_offset = 1.0    # additive offset used in log transform
p3_model = None
# Phase 6 feature normalisation stats
feat_mean = feat_std = None
# Phase 4 feature normalisation stats (separate — trained independently)
p4_feat_mean = p4_feat_std = None
nutrition_constants = {}   # Phase 6 calorie-density constants
target_cols = ['calories', 'fat', 'protein', 'carbs']

# Phase 5 food classifier (optional — loaded if checkpoint exists)
food_clf        = None
food_clf_labels = []   # list[str] index → class name

# Load YOLO + MiDaS (shared by Phase 4 and 6)
_geo_models_loaded = False
try:
    from ultralytics import YOLO as _YOLO
    yolo_model = _YOLO('yolov8n-seg.pt')
    FOOD_CLASSES.update(_build_food_classes(yolo_model))
    midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    midas_model.to(device).eval()
    _t = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    midas_transform = _t.small_transform
    _geo_models_loaded = True
    print(f'✓ YOLO + MiDaS loaded')
except Exception as _e:
    print(f'⚠ YOLO/MiDaS load failed ({_e})')


def _find_ingredient_seg_weights():
    """Find a likely ingredient/component YOLOv8-seg weight file under models/.

    We avoid hardcoding a single filename so the user can drop in the Kaggle
    output without renaming.
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models_dir = os.path.abspath(models_dir)
    if not os.path.isdir(models_dir):
        return None
    candidates = []
    for fn in os.listdir(models_dir):
        if not fn.lower().endswith('.pt'):
            continue
        key = fn.lower()
        # Heuristic: prefer files that look like ingredient segmentation exports
        score = 0
        if 'ingredient' in key:
            score += 3
        if 'seg' in key or 'segment' in key:
            score += 2
        if 'yolo' in key or 'yolov8' in key:
            score += 1
        if score <= 0:
            continue
        candidates.append((score, os.path.join(models_dir, fn)))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], os.path.getmtime(t[1])), reverse=True)
    return candidates[0][1]


_ingredient_ckpt = None
try:
    if _geo_models_loaded:
        _ingredient_ckpt = _find_ingredient_seg_weights()
        if _ingredient_ckpt:
            ingredient_yolo_model = _YOLO(_ingredient_ckpt)
            print(f'✓ Ingredient seg model loaded: {os.path.basename(_ingredient_ckpt)}')
        else:
            print('ℹ No ingredient-seg weights found under models/ (optional)')
except Exception as _e:
    ingredient_yolo_model = None
    print(f'ℹ Ingredient seg model not available ({_e})')

# Load Phase 6 (weight-first)
_p6_loaded = False
if _geo_models_loaded and os.path.isfile(P6_CKPT) and os.path.isfile(P6_CONST) and os.path.isfile(P6_STATS):
    try:
        stats = np.load(P6_STATS, allow_pickle=True)
        feat_mean = stats['feat_mean'].astype(np.float32)
        feat_std  = stats['feat_std'].astype(np.float32)

        with open(P6_CONST) as _f:
            nutrition_constants = json.load(_f)
        target_cols = [k.replace('_per_g', '') for k in nutrition_constants]

        ckpt = torch.load(P6_CKPT, map_location=device, weights_only=False)
        weight_mlp_model = WeightMLP(in_feats=9).to(device)
        weight_mlp_model.load_state_dict(ckpt['model_state_dict'])
        weight_mlp_model.eval()
        _weight_log_target = ckpt.get('use_log_target', False)
        _weight_log_offset = float(ckpt.get('log_offset', 1.0))
        _p6_loaded = True
        print(f'✓ Phase 6 loaded  (weight-first, targets: {target_cols})')
        print(f'  Constants: { {k: round(v,4) for k,v in nutrition_constants.items()} }')
    except Exception as e:
        print(f'⚠ Phase 6 load failed ({e})')

# Load Phase 4 (direct regression) — runs alongside Phase 6
_p4_loaded = False
if _geo_models_loaded and os.path.isfile(P4_CKPT) and os.path.isfile(P4_STATS):
    try:
        _p4_stats = np.load(P4_STATS, allow_pickle=True)
        p4_feat_mean = _p4_stats['feat_mean'].astype(np.float32)
        p4_feat_std  = _p4_stats['feat_std'].astype(np.float32)

        _p4_ckpt = torch.load(P4_CKPT, map_location=device, weights_only=False)
        _p4_tcols = list(_p4_ckpt.get('target_cols',
                         _p4_stats.get('target_cols', target_cols)))
        mlp = NutritionMLP(in_feats=9, num_targets=len(_p4_tcols)).to(device)
        mlp.load_state_dict(_p4_ckpt['model_state_dict'])
        mlp.eval()
        # Align Phase 4 target cols to Phase 6 order if both loaded
        if not _p6_loaded:
            target_cols = _p4_tcols
        _p4_loaded = True
        print(f'✓ Phase 4 loaded  (direct regression, targets: {_p4_tcols})')
    except Exception as e:
        print(f'⚠ Phase 4 load failed ({e})')

# Set pipeline mode based on what loaded
if _p6_loaded and _p4_loaded:
    pipeline_mode = 'full'    # All phases running — best mode
elif _p6_loaded:
    pipeline_mode = 'phase6'
elif _p4_loaded:
    pipeline_mode = 'phase4'

if pipeline_mode in ('full', 'phase6', 'phase4'):
    # Use Phase 6 normalisation stats if available, else Phase 4
    if feat_mean is None and p4_feat_mean is not None:
        feat_mean, feat_std = p4_feat_mean, p4_feat_std

# Load Phase 3 (ResNet baseline) — always attempt, used as fallback or comparison
if os.path.isfile(P3_CKPT):
    try:
        _p3_ckpt = torch.load(P3_CKPT, map_location=device, weights_only=False)
        _p3_tcols = _p3_ckpt.get('target_cols', target_cols)
        p3_model = NutritionEstimator(num_targets=len(_p3_tcols)).to(device)
        p3_model.load_state_dict(_p3_ckpt['model_state_dict'])
        p3_model.eval()
        if pipeline_mode is None:
            pipeline_mode = 'phase3'
            target_cols = _p3_tcols
        print(f'✓ Phase 3 ResNet loaded  (baseline, targets: {_p3_tcols})')
    except Exception as _e:
        print(f'⚠ Phase 3 load failed ({_e})')

if pipeline_mode is None:
    print('⚠ No checkpoint found. Place model files in models/')

# Try Phase 5 food classifier (optional — runs alongside Phase 4)
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

# Load USDA cache from disk so first predictions are instant
_load_usda_cache()
