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
import urllib.request, urllib.parse
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

# ── Per-food nutrition database for all 101 Food-101 classes ──────────────────
# Source: USDA FoodData Central (https://fdc.nal.usda.gov/) — typical serving
# Each entry: (kcal/g, fat/g, protein/g, carb/g, typical_serving_g)
# This implements the full "detect → portion estimate → nutrition DB" pipeline:
#   Phase 5 identifies the food → look up serving_g → multiply by per-gram macros
# ── USDA FoodData Central API — dynamic nutrition lookup ──────────────────────
# Set your free API key: export USDA_API_KEY=your_key_here
# Sign up at: https://fdc.nal.usda.gov/api-key-signup.html
# Without a key the app falls back to Nutrition5K global constants.
# With DEMO_KEY you get 30 req/hour (enough for demos).
USDA_API_KEY    = os.environ.get('USDA_API_KEY', 'DEMO_KEY')
USDA_CACHE_FILE = os.path.join(MODELS_DIR, 'usda_cache.json')
_usda_cache: dict = {}   # food_name -> (kcal/g, fat/g, protein/g, carb/g, serving_g)

# ── Restaurant-portion defaults (typical served weight, not USDA lab sample) ──
# Used when WeightMLP is unreliable AND USDA serving_g looks too small (<150g)
_RESTAURANT_PORTIONS = {
    'pasta': 350, 'spaghetti': 350, 'carbonara': 350, 'linguine': 350,
    'fettuccine': 350, 'lasagna': 300, 'gnocchi': 250, 'ravioli': 280,
    'ramen': 450, 'pho': 450, 'noodle': 350, 'pad thai': 300,
    'pizza': 250, 'burger': 250, 'hamburger': 250, 'hot dog': 150,
    'steak': 220, 'pork chop': 200, 'chicken': 220, 'salmon': 180,
    'rice': 300, 'fried rice': 300, 'bibimbap': 350, 'paella': 300,
    'curry': 350, 'soup': 350, 'stew': 350, 'chowder': 300,
    'sandwich': 220, 'club sandwich': 220, 'grilled cheese': 200, 'burrito': 250,
    'sushi': 250, 'sashimi': 150, 'tacos': 220, 'gyoza': 180,
    'pancakes': 180, 'waffles': 180, 'french toast': 180,
    'omelette': 200, 'eggs benedict': 220, 'huevos': 250,
    'cake': 120, 'cheesecake': 120, 'tiramisu': 150, 'ice cream': 150,
    'salad': 250, 'caesar salad': 250, 'greek salad': 200,
    'nachos': 200, 'chicken wings': 250, 'spring rolls': 150,
    'dumplings': 200, 'bibimbap': 350, 'peking duck': 250,
}


def _restaurant_serving_g(food_name: str) -> float:
    """Return a sensible restaurant-portion weight (g) for a given food name."""
    lower = food_name.lower()
    for kw, g in _RESTAURANT_PORTIONS.items():
        if kw in lower:
            return float(g)
    return 280.0   # Nutrition5K dataset mean as final fallback


def _load_usda_cache():
    """Load previously fetched USDA results from disk into memory."""
    global _usda_cache
    if os.path.isfile(USDA_CACHE_FILE):
        with open(USDA_CACHE_FILE) as f:
            raw = json.load(f)
        _usda_cache = {k: tuple(v) for k, v in raw.items()}
        print(f'✓ USDA cache loaded ({len(_usda_cache)} entries) from {USDA_CACHE_FILE}')


def _save_usda_cache():
    """Persist in-memory USDA cache to disk."""
    os.makedirs(os.path.dirname(USDA_CACHE_FILE), exist_ok=True)
    with open(USDA_CACHE_FILE, 'w') as f:
        json.dump({k: list(v) for k, v in _usda_cache.items()}, f, indent=2)


def _lookup_usda(food_name: str):
    """
    Query USDA FoodData Central for per-gram macros + typical serving size.

    Strategy
    ────────
    1. Build query list: full dish name PLUS each individual keyword (>=4 chars).
       e.g. "spaghetti carbonara" → ["spaghetti carbonara", "carbonara", "spaghetti"]
       FNDDS often lacks composite dish names (e.g. "carbonara") so falling back
       to keywords is essential — the keyword "carbonara" hits branded pasta meals
       at the correct ~2 kcal/g instead of the FNDDS "spaghetti sauce" at 0.9.

    2. For every USDA result compute kcal via the ATWATER formula:
           kcal/g = protein×4 + fat×9 + carbs×4  (all in g/g)
       This is more reliable than the survey "Energy" field which varies by
       preparation and can capture sauces-only rather than full plated dishes.

    3. Keep candidates in 0.5–4.5 kcal/g — the realistic range for a plated
       restaurant portion (broth/salad ~0.5, cheese/nut dishes up to ~4.5).

    4. Pick the highest Atwater-consistent candidate across ALL queries.
       Higher kcal/g wins because USDA survey data skews toward light/home-cooked
       versions; restaurant dishes are calorie-denser.

    Returns (kcal/g, fat/g, protein/g, carb/g, serving_g) or None.
    """
    cache_key = food_name.lower().replace('_', ' ').strip()
    if cache_key in _usda_cache:
        return _usda_cache[cache_key]

    if not USDA_API_KEY:
        return None

    # Build list of queries: full name first, then individual distinctive words
    words = [w for w in cache_key.split() if len(w) >= 4]
    words.sort(key=len, reverse=True)   # longer/more-specific words first
    queries_to_try = [cache_key] + [w for w in words if w != cache_key]

    def _fetch_foods(q: str):
        """Fetch up to 8 results for query q (no dataType filter = all sources)."""
        params = urllib.parse.urlencode({
            'query': q, 'pageSize': 8, 'api_key': USDA_API_KEY
        })
        url = f'https://api.nal.usda.gov/fdc/v1/foods/search?{params}'
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=7) as resp:
            return json.loads(resp.read()).get('foods', [])

    def _parse_food(food):
        """
        Return (kcal_g, fat_g, prot_g, carb_g, serving_g).
        kcal is computed with the Atwater formula from macros — not the survey
        Energy field — because USDA macros are more accurately measured than
        per-serving energy values in the FNDDS survey.
        """
        nmap = {n['nutrientName']: float(n.get('value') or 0)
                for n in food.get('foodNutrients', [])}
        fat_100  = nmap.get('Total lipid (fat)', 0.0)
        prot_100 = nmap.get('Protein', 0.0)
        carb_100 = nmap.get('Carbohydrate, by difference', 0.0)
        # Atwater: kcal per 100 g
        kcal_100 = prot_100 * 4.0 + fat_100 * 9.0 + carb_100 * 4.0
        kcal_g = kcal_100 / 100.0
        fat_g  = fat_100  / 100.0
        prot_g = prot_100 / 100.0
        carb_g = carb_100 / 100.0
        # Serving size from USDA input foods, else 0 (caller handles fallback)
        srv = 0.0
        for m in food.get('finalFoodInputFoods', []):
            gw = m.get('gramWeight')
            if gw and float(gw) > 20:
                srv = float(gw)
                break
        return kcal_g, fat_g, prot_g, carb_g, srv

    # Plausible range for a plated restaurant dish (not a pure sauce or raw oil)
    KCAL_MIN, KCAL_MAX = 0.5, 4.5

    def _median_candidate(candidates):
        """Return the median-kcal/g candidate from a list — avoids extremes."""
        candidates.sort(key=lambda t: t[0])
        return candidates[len(candidates) // 2]

    try:
        # ── Step 1: Try the full dish name first ──────────────────────────────
        # If it gives plausible results, use the MEDIAN of those (not max).
        # Taking the median avoids both the diluted low-calorie survey variants
        # AND accidentally grabbing a condiment/sauce entry that skews high.
        full_name_candidates = []
        try:
            for food in _fetch_foods(queries_to_try[0]):
                parsed = _parse_food(food)
                if KCAL_MIN <= parsed[0] <= KCAL_MAX:
                    full_name_candidates.append(parsed)
        except Exception:
            pass

        if full_name_candidates:
            # Full dish name returned plausible entries — use median
            best = _median_candidate(full_name_candidates)
        else:
            # ── Step 2: Full name failed → try individual keywords ────────────
            # e.g. "spaghetti carbonara" → "carbonara" → hits branded pasta meals
            # In this fallback we take the HIGHEST result because keywords are
            # less specific and more likely to match ingredient-only items.
            kw_candidates = []
            for q in queries_to_try[1:]:
                try:
                    for food in _fetch_foods(q):
                        parsed = _parse_food(food)
                        if KCAL_MIN <= parsed[0] <= KCAL_MAX:
                            kw_candidates.append(parsed)
                except Exception:
                    continue
            if not kw_candidates:
                return None
            best = max(kw_candidates, key=lambda t: t[0])

        kcal_g, fat_g, prot_g, carb_g, srv = best

        # Serving size: use USDA value only if it looks like a real meal portion;
        # otherwise ask _restaurant_serving_g for a sensible default.
        if srv < 150:
            srv = _restaurant_serving_g(cache_key)

        entry = (kcal_g, fat_g, prot_g, carb_g, srv)
        _usda_cache[cache_key] = entry
        _save_usda_cache()
        print(f'  USDA "{cache_key}": {kcal_g:.2f} kcal/g  '
              f'F={fat_g:.3f}  P={prot_g:.3f}  C={carb_g:.3f}  srv={srv:.0f}g'
              f'  (full-name:{len(full_name_candidates)} hits)')
        return entry

    except Exception as _e:
        print(f'  ⚠ USDA lookup failed for "{cache_key}": {_e}')
        return None


# Backward-compat alias — populated lazily from USDA cache
FOOD_KCAL_PER_G: dict = {}


# Stub so existing code that calls _food_name_to_key still works
def _food_name_to_key(name: str) -> str:
    """Normalise a food classifier label to a plain search query."""
    return name.lower().replace('_', ' ').replace('-', ' ').strip()


# ── Old hardcoded FOOD_NUTRITION_DB (101 entries) replaced by _lookup_usda() ──
# The USDA FoodData Central API now provides live per-gram macros for any food.


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

pipeline_mode = None   # 'full', 'phase6', 'phase4', or 'phase3'
                       # 'full' = Phase 6 + Phase 4 + Phase 5 all running together
mlp = weight_mlp_model = yolo_model = midas_model = midas_transform = None
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

# ── Load YOLO + MiDaS (shared by Phase 4 and 6) ──────────────────────────────
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

# ── Load Phase 6 (weight-first) ───────────────────────────────────────────────
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
        _p6_loaded = True
        print(f'✓ Phase 6 loaded  (weight-first, targets: {target_cols})')
        print(f'  Constants: { {k: round(v,4) for k,v in nutrition_constants.items()} }')
    except Exception as e:
        print(f'⚠ Phase 6 load failed ({e})')

# ── Load Phase 4 (direct regression) — runs alongside Phase 6 ─────────────────
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

# ── Load Phase 3 (ResNet baseline) — always attempt, used as fallback or comparison ──
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

# Load USDA cache from disk so first predictions are instant
_load_usda_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def _get_food_mask(pil_img: Image.Image):
    """Returns (mask np.ndarray, detected_items list[dict{name, area_ratio}]).

    Fallback strategy (most real-world food photos lack COCO food classes):
      1. YOLO food class match  → use those masks (best)
      2. YOLO found other objects → use union of non-person/non-vehicle masks
         (dining table, utensils, containers are good food-location proxies)
      3. Everything else         → central 60%×60% rectangle
    """
    img_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    H, W    = img_bgr.shape[:2]

    # Central rectangle fallback — covers 60% of H and W, much better than
    # a small circle for centred real-world food photos.
    fallback = np.zeros((H, W), dtype=np.float32)
    y0f, y1f = int(H * 0.2), int(H * 0.8)
    x0f, x1f = int(W * 0.2), int(W * 0.8)
    fallback[y0f:y1f, x0f:x1f] = 1.0

    # Classes to skip when building the second-chance proxy mask
    _SKIP_CLASSES = {
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'chair', 'couch', 'bed', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    }

    if yolo_model is None:
        return cv2.resize(fallback, (IMG_SIZE, IMG_SIZE)), []

    results  = yolo_model(img_bgr, verbose=False)[0]
    if results.masks is None or len(results.masks) == 0:
        return cv2.resize(fallback, (IMG_SIZE, IMG_SIZE)), []

    classes  = results.boxes.cls.cpu().numpy().astype(int)
    confs    = results.boxes.conf.cpu().numpy()
    food_idx = [i for i, c in enumerate(classes) if c in FOOD_CLASSES]

    if not food_idx:
        # Second-chance: union of any non-person/non-vehicle objects (dining
        # table, fork/knife/spoon, potted plant, etc. all reliably co-occur
        # with food in real-world food photos).
        proxy_idx = [
            i for i, c in enumerate(classes)
            if yolo_model.names.get(int(c), '').lower() not in _SKIP_CLASSES
        ]
        if proxy_idx:
            combined = np.zeros((H, W), dtype=np.float32)
            for i in proxy_idx:
                m = cv2.resize(results.masks[i].data[0].cpu().numpy(), (W, H))
                combined = np.maximum(combined, (m > 0.5).astype(np.float32))
            combined = (combined > 0.5).astype(np.float32)
            if combined.sum() > 500:   # at least 500 px must be covered
                return cv2.resize(combined, (IMG_SIZE, IMG_SIZE)), []
        # Nothing useful from YOLO → central-rectangle fallback
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
    if mask is not None:
        mask_full = cv2.resize(mask, (W_img, H_img))
    else:
        mask_full = np.zeros((H_img, W_img), dtype=np.float32)  # no mask → use full image
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

    if pipeline_mode in ('full', 'phase6', 'phase4'):
        # ── Step 1 (Phase 4/6 shared): YOLO mask + MiDaS depth → 9 features ──────
        feat_raw, items, raw_mask = _extract_features(image)
        # raw feat_raw is already normalised by Phase 6 stats;
        # for Phase 4 we need its own normalisation
        x6 = torch.tensor(feat_raw, dtype=torch.float32).unsqueeze(0).to(device)
        # Phase 4 feature vector (re-normalised with p4 stats if available)
        if p4_feat_mean is not None:
            _raw_unnorm = feat_raw * feat_std.flatten() + feat_mean.flatten()
            _p4_norm    = (_raw_unnorm - p4_feat_mean.flatten()) / p4_feat_std.flatten()
            x4 = torch.tensor(_p4_norm, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            x4 = x6

        # ── Step 2 (Phase 5): classify food type ─────────────────────────────────
        food_type_preds = _classify_food_type(image, raw_mask, top_k=3)
        detected_food   = food_type_preds[0][0] if food_type_preds else None
        food_conf       = food_type_preds[0][1] if food_type_preds else None

        # ── Step 3 (Phase 6): predict weight → × food-specific constants ─────────
        p6_mean = p6_std = None
        w_mean  = None
        w_std_mc = 0.0
        _weight_note = ''
        if weight_mlp_model is not None:
            weight_mlp_model.eval()
            for m in weight_mlp_model.modules():
                if isinstance(m, nn.Dropout): m.train()
            torch.manual_seed(42)
            _w_samples = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    _w_samples.append(weight_mlp_model(x6).item())
            weight_mlp_model.eval()
            w_raw = float(np.mean(_w_samples));  w_std_mc = float(np.std(_w_samples))

            DATASET_MEAN_WEIGHT = 280.0
            MIN_WEIGHT          = 50.0

            # ── Nutrition DB lookup (detect → portion → database) ──────────────
            # Do this BEFORE the fallback decision so we can use the USDA serving
            # size as a plausibility check on the WeightMLP prediction.
            _db_entry     = None
            _food_scale_note = ''
            if detected_food and food_conf and food_conf >= 0.15:
                _query    = _food_name_to_key(detected_food)
                _db_entry = _lookup_usda(_query)

            # Fallback criteria (use USDA/restaurant serving size instead of MLP):
            #  1. WeightMLP below hard floor (50g)
            #  2. Very uncertain prediction relative to its own magnitude
            #  3. Prediction is less than 40% of the USDA typical serving size —
            #     this catches cases where MiDaS depth cues mislead the MLP
            #     (e.g. a bowl seen from above looks shallow → low predicted weight)
            #     40% threshold is purely geometric: anything less than ~2/5 of the
            #     expected restaurant portion is almost certainly a depth artefact.
            _usda_serving = float(_db_entry[4]) if _db_entry is not None else DATASET_MEAN_WEIGHT
            _cv_fallback  = (w_std_mc > abs(w_raw) and w_raw < DATASET_MEAN_WEIGHT)
            _size_fallback = (w_raw < 0.40 * _usda_serving)
            _used_fallback = w_raw < MIN_WEIGHT or _cv_fallback or _size_fallback

            # Decide portion weight
            if _used_fallback:
                if _db_entry is not None:
                    w_mean = float(_db_entry[4])
                else:
                    w_mean = _restaurant_serving_g(detected_food or '')
            else:
                w_mean = max(w_raw, MIN_WEIGHT)

            # Build per-target nutrition from DB (all 4 macros) or fall back to
            # global constants scaled by kcal ratio (old behaviour)
            const_keys = list(nutrition_constants.keys())
            if _db_entry is not None:
                # Full DB lookup: each macro has its own per-gram value
                # DB tuple: (kcal/g, fat/g, protein/g, carb/g, serving_g)
                _db_kcal, _db_fat, _db_pro, _db_carb, _db_srv = _db_entry
                _db_macro_per_g = {
                    'calories': _db_kcal,
                    'fat':      _db_fat,
                    'protein':  _db_pro,
                    'carb':     _db_carb,
                    'carbs':    _db_carb,  # alias
                }
                active_constants = {
                    k: _db_macro_per_g.get(k.replace('_per_g', ''),
                       nutrition_constants[k])
                    for k in const_keys
                }
                _food_scale_note = (f' · DB: {_db_kcal:.2f} kcal/g, '
                                    f'{_db_fat:.3f} fat/g, '
                                    f'{_db_pro:.3f} pro/g, '
                                    f'{_db_carb:.3f} carb/g')
            else:
                # No DB match — fall back to global Nutrition5K constants
                active_constants = dict(nutrition_constants)

            p6_mean = np.array([w_mean * active_constants[k] for k in const_keys], dtype=np.float32)
            p6_std  = np.array([w_std_mc * active_constants[k] for k in const_keys], dtype=np.float32)
            _src = 'DB serving' if (_used_fallback and _db_entry) else ('fallback' if _used_fallback else 'WeightMLP')
            _weight_note = f'**{w_mean:.0f}±{w_std_mc*2:.0f}g** ({_src}){_food_scale_note}'

        # ── Step 4 (Phase 4): direct regression via MLP ───────────────────────────
        p4_mean = p4_std = None
        if mlp is not None:
            mlp.eval()
            for m in mlp.modules():
                if isinstance(m, nn.Dropout): m.train()
            torch.manual_seed(42)
            _p4_mc = []
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    _p4_mc.append(mlp(x4).squeeze(0).cpu().numpy())
            mlp.eval()
            _p4_mc = np.stack(_p4_mc)
            p4_mean = _p4_mc.mean(0)
            p4_std  = _p4_mc.std(0)

        # ── Step 5: Ensemble Phase 4 + Phase 6 ───────────────────────────────────
        # When a food-specific USDA entry was found with plausible caloric density
        # (>= 1.0 kcal/g), trust Phase 6 heavily — Phase 4 was trained on lab-
        # controlled Nutrition5K dishes and under-predicts restaurant portions.
        # Without a food-specific match, fall back to inverse-variance weighting.
        if p6_mean is not None and p4_mean is not None:
            _usda_confidence = (_db_entry is not None and _db_entry[0] >= 1.0
                                and food_conf is not None and food_conf >= 0.15)
            if _usda_confidence:
                # Food type known + USDA lookup good → Phase 6 dominates
                _alpha6 = 0.85
                _alpha4 = 0.15
                mean_pred = _alpha6 * p6_mean + _alpha4 * p4_mean
                std_pred  = np.sqrt(_alpha6**2 * p6_std**2 + _alpha4**2 * p4_std**2)
                _blend_note = 'food-specific USDA'
            else:
                # No confident food match — use inverse-variance
                p6_var = np.clip(p6_std ** 2, 1e-6, None)
                p4_var = np.clip(p4_std ** 2, 1e-6, None)
                w6 = 1.0 / p6_var
                w4 = 1.0 / p4_var
                w_total   = w6 + w4
                mean_pred = (w6 * p6_mean + w4 * p4_mean) / w_total
                std_pred  = np.sqrt(1.0 / w_total)
                _alpha6   = float((w6 / w_total).mean())
                _alpha4   = 1.0 - _alpha6
                _blend_note = 'inverse-variance'
            ensemble_note = (f'\n_Ensemble: Phase 6 weight {_alpha6*100:.0f}% · '
                             f'Phase 4 {_alpha4*100:.0f}% ({_blend_note})_'
                             f'\n_Phase 6 weight prediction: {_weight_note}_')
        elif p6_mean is not None:
            mean_pred, std_pred = p6_mean, p6_std
            ensemble_note = f'\n_Phase 6 only (Phase 4 not loaded) · Weight: {_weight_note}_'
        else:
            mean_pred, std_pred = p4_mean, p4_std
            ensemble_note = '\n_Phase 4 only (Phase 6 not loaded)_'

        # ── Step 6 (Phase 3): ResNet baseline comparison ──────────────────────────
        p3_note = ''
        if p3_model is not None:
            _img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                _p3_pred = p3_model(_img_t).squeeze(0).cpu().numpy()
            cal_idx3 = next((i for i, c in enumerate(target_cols) if 'cal' in c.lower()), 0)
            p3_note  = f'\n_Phase 3 baseline (ResNet-50): {_p3_pred[cal_idx3]:.0f} kcal_'

        # ── Build outputs ─────────────────────────────────────────────────────────
        cal_idx  = next((i for i, c in enumerate(target_cols) if 'cal' in c.lower()), 0)
        total_cal = float(mean_pred[cal_idx])

        # Format weight string for the table: "250 g" or "250±66 g"
        _w_str = f'{w_mean:.0f}±{w_std_mc*2:.0f} g' if w_mean else '—'

        if detected_food:
            # Food classifier identified the dish — collapse all YOLO segments
            # into a single row for the whole dish.  Showing one row per YOLO
            # mask segment is misleading (they all say the same food name and
            # the per-segment calorie split has no nutritional meaning).
            ingredient_rows = [[detected_food, _w_str, f"{total_cal:.0f}", f"{food_conf*100:.0f}%"]]
            for alt_name, alt_conf in (food_type_preds[1:3] if food_type_preds else []):
                ingredient_rows.append([f'  (alt) {alt_name}', '—', '—', f"{alt_conf*100:.0f}%"])
        elif items:
            # No food classification — show each YOLO segment with its own name
            total_area = sum(it['area'] for it in items) or 1.0
            ingredient_rows = []
            for it in items:
                frac = it['area'] / total_area
                conf_str = f"{it['conf']*100:.0f}%"
                ingredient_rows.append([it['name'], _w_str, f"{total_cal * frac:.0f}", conf_str])
        else:
            ingredient_rows = [['Unknown dish', _w_str, f"{total_cal:.0f}", '—']]

        result_json = {col: f'{mu:.1f} ± {sig*2:.1f}' for col, mu, sig in zip(target_cols, mean_pred, std_pred)}
        clf_note  = f'🍽 Detected food: **{detected_food}** ({food_conf*100:.0f}% confidence)\n\n' if detected_food else ''
        mode_note = (f'{clf_note}'
                     f'_Pipeline: Phase 6 — YOLOv8-seg + MiDaS → WeightMLP · '
                     f'Phase 4 — YOLO+MiDaS → MLP · Phase 5 — EfficientNet-B0_'
                     f'{ensemble_note}{p3_note}')

    else:  # phase3 only
        img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            mean_pred = p3_model(img_t).squeeze(0).cpu().numpy()
        result_json = {col: f'{v:.1f}' for col, v in zip(target_cols, mean_pred)}
        mode_note  = '_Pipeline: ResNet-50 direct regression (Phase 3 baseline)_'
        ingredient_rows = [['N/A (ResNet baseline — no segmentation)', '—', '—', '—']]

    label_txt = '\n'.join([f'{col}: {result_json[col]}' for col in target_cols])
    label_txt += f'\n\n{mode_note}\n> Units: kcal for calories, grams for macros'
    return label_txt, result_json, ingredient_rows


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

mode_label = {
    'full':   '🚀 Full Pipeline — Phase 3+4+5+6 all active (ensemble Phase4+6, food type from Phase5, ResNet3 baseline)',
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
        headers=['Detected Food / Ingredient', 'Predicted Weight', 'Est. Calories (kcal)', 'Confidence'],
        datatype=['str', 'str', 'str', 'str'],
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
