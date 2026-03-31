import os
import json
import time
import warnings
import re

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

try:
    import open_clip  # type: ignore
except Exception:
    open_clip = None

try:
    from .core_models import (
        MODELS_DIR,
        IMG_SIZE,
        MC_SAMPLES,
        DISH_LABEL_MIN_CONF,
        FOOD_CLASSES,
        FOODSEG103_FRUITS,
        INGREDIENT_LABELS,
        _gpu_stats_md,
        _maybe_foodseg103_single_fruit_override,
        device,
        pipeline_mode,
        yolo_model,
        midas_model,
        midas_transform,
        ingredient_yolo_model,
        mlp,
        weight_mlp_model,
        p3_model,
        feat_mean,
        feat_std,
        p4_feat_mean,
        p4_feat_std,
        nutrition_constants,
        target_cols,
        food_clf,
        food_clf_labels,
        _weight_log_target,
        _weight_log_offset,
    )  # type: ignore
except Exception:
    from core_models import (  # type: ignore
        MODELS_DIR,
        IMG_SIZE,
        MC_SAMPLES,
        DISH_LABEL_MIN_CONF,
        FOOD_CLASSES,
        FOODSEG103_FRUITS,
        INGREDIENT_LABELS,
        _gpu_stats_md,
        _maybe_foodseg103_single_fruit_override,
        device,
        pipeline_mode,
        yolo_model,
        midas_model,
        midas_transform,
        ingredient_yolo_model,
        mlp,
        weight_mlp_model,
        p3_model,
        feat_mean,
        feat_std,
        p4_feat_mean,
        p4_feat_std,
        nutrition_constants,
        target_cols,
        food_clf,
        food_clf_labels,
        _weight_log_target,
        _weight_log_offset,
    )

try:
    from .usda import _lookup_usda, _restaurant_serving_g, _food_name_to_key  # type: ignore
except Exception:
    from usda import _lookup_usda, _restaurant_serving_g, _food_name_to_key  # type: ignore

warnings.filterwarnings('ignore')

# Zero-shot label set for multi-item listing (covers common sides/condiments)
ZERO_SHOT_FOOD_LABELS = [
    'hamburger', 'cheeseburger', 'sandwich', 'hot dog', 'pizza',
    'french fries', 'onion rings',
    'salad', 'soup',
    'ketchup', 'mayonnaise', 'mustard', 'barbecue sauce', 'dipping sauce',
    'fried chicken', 'chicken wings',
    'taco', 'burrito',
    'sushi',
    'cake', 'donut', 'ice cream',
    'apple', 'banana', 'orange',
]

clip_model = None
clip_preprocess = None
clip_tokenizer = None
clip_text_features = None
clip_labels = None


_UNIT_TO_G = {
    'g': 1.0,
    'oz': 28.3495,
    'lb': 453.592,
    'cup': 240.0,
    'tbsp': 15.0,
    'tsp': 5.0,
}


def _portion_to_grams(amount, unit: str):
    try:
        a = float(amount)
    except Exception:
        return None
    if not np.isfinite(a) or a <= 0:
        return None
    u = str(unit or 'g').strip().lower()
    if u not in _UNIT_TO_G:
        return None
    return float(a * _UNIT_TO_G[u])


def _render_table_html(rows):
    table_html = '''
    <div style="background: var(--surface); border: 1px solid var(--border); border-top: none; border-radius: 0 0 var(--radius) var(--radius); overflow: hidden; box-shadow: var(--shadow);">
        <table class="n5k-html-table" style="width: 100%; border-collapse: collapse; text-align: left; background: var(--surface);">
            <thead>
                <tr style="background: var(--surface2); border-bottom: 1px solid var(--border);">
                    <th style="padding: 14px 20px; font-family: 'Outfit', sans-serif; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">Detected Dish</th>
                    <th style="padding: 14px 20px; font-family: 'Outfit', sans-serif; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">Predicted Weight</th>
                    <th style="padding: 14px 20px; font-family: 'Outfit', sans-serif; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">Est. Calories (kcal)</th>
                    <th style="padding: 14px 20px; font-family: 'Outfit', sans-serif; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">Confidence</th>
                </tr>
            </thead>
            <tbody>
    '''

    for row in rows:
        dish, weight, cals, conf = row
        is_alt = "(alt)" in str(dish) or "•" in str(dish)
        pad_left = "36px" if is_alt else "20px"
        text_color = "var(--muted)" if is_alt else "var(--text)"
        font_w = "500" if is_alt else "700"

        table_html += f'''
                <tr class="n5k-row">
                    <td style="padding: 14px 20px; padding-left: {pad_left}; font-size: 0.9rem; font-weight: {font_w}; color: {text_color};">{dish}</td>
                    <td style="padding: 14px 20px; font-size: 0.9rem; color: var(--text);">{weight}</td>
                    <td style="padding: 14px 20px; font-size: 0.95rem; color: var(--purple-dark); font-weight: 800;">{cals}</td>
                    <td style="padding: 14px 20px; font-size: 0.85rem; color: var(--muted); font-weight: 500;">{conf}</td>
                </tr>
        '''
    table_html += '</tbody></table></div>'
    return table_html


def _parse_first_float(x):
    m = re.search(r'[-+]?\d+(?:\.\d+)?', str(x))
    if not m:
        return None
    try:
        v = float(m.group(0))
    except Exception:
        return None
    return v if np.isfinite(v) else None


def update_ingredient_portions(result_json: dict, portion_table):
    if not isinstance(result_json, dict):
        return None, result_json, None

    meta = result_json.get('_meta') if isinstance(result_json.get('_meta'), dict) else None
    if not meta:
        return None, result_json, None

    target_cols = list(meta.get('target_cols') or [])
    base_mean = meta.get('base_mean') if isinstance(meta.get('base_mean'), dict) else {}
    base_ci = meta.get('base_ci') if isinstance(meta.get('base_ci'), dict) else {}
    base_weight_g = meta.get('base_weight_g')
    try:
        base_weight_g = float(base_weight_g) if base_weight_g is not None else None
    except Exception:
        base_weight_g = None

    base_rows = meta.get('ingredient_rows') if isinstance(meta.get('ingredient_rows'), list) else []

    def _clean_name(s: str):
        s = str(s or '').strip()
        if s.startswith('•'):
            s = s[1:].strip()
        return s

    # Gradio Dataframe often arrives as a pandas.DataFrame
    if portion_table is None:
        portion_table = []
    elif not isinstance(portion_table, list):
        try:
            portion_table = portion_table.values.tolist()
        except Exception:
            try:
                portion_table = list(portion_table)
            except Exception:
                portion_table = []

    user_map = {}
    if isinstance(portion_table, list):
        for r in portion_table:
            if not isinstance(r, (list, tuple)) or len(r) < 3:
                continue
            name = _clean_name(r[0])
            if not name:
                continue
            amt = r[1]
            unit = str(r[2] or 'g')
            user_map[name.lower()] = (amt, unit)

    dish_macro_per_g = {}
    if base_weight_g and base_weight_g > 0:
        for col in target_cols:
            try:
                dish_macro_per_g[col] = float(base_mean.get(col, 0.0)) / base_weight_g
            except Exception:
                dish_macro_per_g[col] = 0.0

    new_ing_rows = []
    totals = {col: 0.0 for col in target_cols}
    total_weight_new = 0.0

    for it in base_rows:
        if not isinstance(it, dict):
            continue
        dish = _clean_name(it.get('dish', ''))
        conf = str(it.get('conf', '—'))

        # Default grams from the model output
        base_g = it.get('weight_g')
        if base_g is None:
            base_g = _parse_first_float(it.get('weight', ''))
        try:
            base_g = float(base_g) if base_g is not None else None
        except Exception:
            base_g = None

        # User override grams
        amt_unit = user_map.get(dish.lower())
        if amt_unit is not None:
            g_new = _portion_to_grams(amt_unit[0], amt_unit[1])
        else:
            g_new = base_g
        if g_new is None or g_new <= 0:
            continue

        # Per-ingredient USDA if available
        entry = None
        if dish:
            try:
                entry = _lookup_usda(_food_name_to_key(dish))
            except Exception:
                entry = None

        per_g = {}
        if entry is not None:
            kcal_g, fat_g, pro_g, carb_g, _srv = entry
            per_g['calories'] = float(kcal_g)
            per_g['fat'] = float(fat_g)
            per_g['protein'] = float(pro_g)
            per_g['carb'] = float(carb_g)
            per_g['carbs'] = float(carb_g)

        # Calories fallback: scale from base row
        kcal_new = None
        if 'calories' in per_g:
            kcal_new = float(g_new) * float(per_g['calories'])
        else:
            base_kcal = it.get('cal_kcal')
            if base_kcal is None:
                base_kcal = _parse_first_float(it.get('cals', ''))
            try:
                base_kcal = float(base_kcal) if base_kcal is not None else None
            except Exception:
                base_kcal = None
            if base_kcal is not None and base_g is not None and base_g > 0:
                kcal_new = float(base_kcal) * float(g_new / base_g)
            elif dish_macro_per_g.get('calories', 0.0) > 0:
                kcal_new = float(g_new) * float(dish_macro_per_g['calories'])

        # Macro totals
        for col in target_cols:
            if col in per_g:
                totals[col] += float(g_new) * float(per_g[col])
            else:
                totals[col] += float(g_new) * float(dish_macro_per_g.get(col, 0.0))

        total_weight_new += float(g_new)

        new_ing_rows.append({
            'dish': it.get('dish', dish),
            'weight_g': float(g_new),
            'cal_kcal': float(kcal_new) if kcal_new is not None else None,
            'conf': conf,
            'weight': f'{float(g_new):.0f} g',
            'cals': f'{float(kcal_new):.0f}' if kcal_new is not None else (it.get('cals', '—')),
        })

    # If we failed to compute totals, fall back to scaling the whole dish
    if (total_weight_new <= 0) and base_weight_g and base_weight_g > 0:
        total_weight_new = float(base_weight_g)

    if total_weight_new > 0 and base_weight_g and base_weight_g > 0:
        scale = float(total_weight_new / base_weight_g)
    else:
        scale = 1.0

    # If totals look empty (e.g., no USDA + no base_per_g), scale base_mean
    if all(abs(float(totals.get(c, 0.0))) < 1e-9 for c in target_cols):
        for col in target_cols:
            try:
                totals[col] = float(base_mean.get(col, 0.0)) * scale
            except Exception:
                totals[col] = 0.0

    new_json = dict(result_json)
    for col in target_cols:
        mu = float(totals.get(col, 0.0))
        try:
            ci0 = float(base_ci.get(col, 0.0))
        except Exception:
            ci0 = 0.0
        ci = float(ci0) * scale if ci0 else 0.0
        new_json[col] = f'{mu:.1f} ± {ci:.1f}' if ci > 0 else f'{mu:.1f}'

    detected_food = meta.get('detected_food')
    food_conf = meta.get('food_conf')
    clean_mode = str(meta.get('pipeline_details') or '')

    if detected_food:
        try:
            conf_pct = f"{float(food_conf)*100:.0f}%"
        except Exception:
            conf_pct = '—'
        clean_clf = f'<div style="font-size: 1rem; margin-bottom: 12px; color: var(--text);">🍽 Detected food: <strong>{detected_food}</strong> ({conf_pct} confidence)</div>'
    else:
        clean_clf = ''

    info_line = (
        f'<div style="font-size: 0.82rem; color: var(--muted); margin-bottom: 10px;">'
        f'Updated using your ingredient portions (total ~{total_weight_new:.0f} g).</div>'
    )

    cards_html = '<div style="display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">'
    for col in target_cols:
        val = new_json.get(col, '')
        cards_html += f'''
        <div style="background: var(--surface2); padding: 16px; border-radius: var(--radius); border: 1px solid var(--border); flex: 1; min-width: 100px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-family: 'Outfit', sans-serif; font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px;">{col}</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: var(--purple-dark); font-weight: 800;">{val}</div>
        </div>
        '''
    cards_html += '</div>'

    label_html = f"""
    {clean_clf}
    {info_line}
    {cards_html}
    <div style="background: var(--purple-dim); border-radius: var(--radius); padding: 12px 16px; font-size: 0.8rem; color: var(--text); line-height: 1.5; margin-bottom: 12px; border: 1px solid rgba(123,110,246,0.15);">
        <strong style="color: var(--purple-dark);">⚙️ Pipeline Details:</strong><br> {clean_mode}
    </div>
    <div style="font-size: 0.8rem; color: var(--muted); border-left: 3px solid var(--purple); padding-left: 10px;">
        Units: kcal for calories, grams for macros
    </div>
    """

    # Table
    table_rows = []
    for r in new_ing_rows:
        table_rows.append([r.get('dish', ''), r.get('weight', '—'), r.get('cals', '—'), r.get('conf', '—')])
    table_html = _render_table_html(table_rows) if table_rows else meta.get('base_table_html')

    new_meta = dict(meta)
    new_meta['ingredient_rows_scaled'] = new_ing_rows
    new_meta['scaled_total_weight_g'] = float(total_weight_new)
    new_json['_meta'] = new_meta

    return label_html, new_json, table_html

def _try_load_clip():
    """Best-effort load for CLIP zero-shot classifier.
    Keeps the app runnable if open_clip isn't installed."""
    global clip_model, clip_preprocess, clip_tokenizer, clip_text_features, clip_labels
    if open_clip is None or clip_model is not None:
        return
    try:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        clip_model = clip_model.to(device).eval()
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # Combine Food-101 class names with extra labels (dedupe)
        labels = []
        for x in (food_clf_labels or []):
            labels.append(x.replace('_', ' '))
        labels += ZERO_SHOT_FOOD_LABELS
        seen = set()
        clip_labels = []
        for lab in labels:
            k = lab.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            clip_labels.append(k)

        with torch.no_grad():
            text = clip_tokenizer([f'a photo of {l}' for l in clip_labels]).to(device)
            tf = clip_model.encode_text(text)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            clip_text_features = tf
        print(f'✓ CLIP loaded for zero-shot item labels ({len(clip_labels)} labels)')
    except Exception as _e:
        clip_model = None
        clip_preprocess = None
        clip_tokenizer = None
        clip_text_features = None
        clip_labels = None
        print(f'ℹ CLIP not available ({_e}) — multi-item labels limited to Food-101')


# Feature extraction helpers (Phase 4)

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

        # Crop bounding box of the mask region
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

    # Crop bounding box of the mask region
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


def _classify_crop_food101(crop: Image.Image, top_k: int = 1):
    """Classify a cropped region using the Food-101 classifier.
    Returns list of (display_name, confidence_float) best-first.
    """
    if food_clf is None or not food_clf_labels:
        return []

    _clf_transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = _clf_transform(crop.convert('RGB')).unsqueeze(0).to(device)
    food_clf.eval()
    with torch.no_grad():
        logits = food_clf(x).squeeze(0)
    probs = F.softmax(logits, dim=0).cpu().numpy()
    topk_i = probs.argsort()[-top_k:][::-1]
    return [
        (food_clf_labels[int(i)].replace('_', ' ').title(), float(probs[int(i)]))
        for i in topk_i
    ]


def _detect_and_classify_items(pil_img: Image.Image,
                               max_regions: int = 8,
                               min_area: float = 0.01,
                               min_conf: float = 0.25):
    """Use pretrained YOLO instance masks as region proposals, then classify each
    region with Food-101 to get a multi-item list.

    Returns list[dict{name, conf, area}] sorted by area desc.
    """
    if yolo_model is None:
        return []

    img_rgb = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    results = yolo_model(img_bgr, verbose=False)[0]
    if results.masks is None or len(results.masks) == 0:
        return []

    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    # Reuse the same skip list as mask-building so we avoid people/vehicles/etc.
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

    candidates = []
    for i in range(len(classes)):
        name = yolo_model.names.get(int(classes[i]), '').lower()
        if name in _SKIP_CLASSES:
            continue
        m = cv2.resize(results.masks[i].data[0].cpu().numpy(), (W, H))
        m_bin = (m > 0.5)
        area = float(m_bin.mean())
        if area < min_area:
            continue
        candidates.append((i, float(confs[i]), area, m_bin))

    if not candidates:
        return []

    # Prefer confident + large regions first
    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
    candidates = candidates[:max_regions]

    # Best-effort CLIP load (no training) for labels like fries/sauces
    _try_load_clip()

    def _classify_crop(c: Image.Image):
        """Return (label, confidence) for a crop."""
        if clip_model is not None and clip_preprocess is not None and clip_text_features is not None and clip_labels:
            try:
                img_t = clip_preprocess(c.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    imf = clip_model.encode_image(img_t)
                    imf = imf / imf.norm(dim=-1, keepdim=True)
                    logits = (imf @ clip_text_features.T).squeeze(0)
                    probs = logits.softmax(dim=0).detach().cpu().numpy()
                j = int(probs.argmax())
                return clip_labels[j].title(), float(probs[j])
            except Exception:
                pass
        preds = _classify_crop_food101(c, top_k=1)
        if not preds:
            return None, 0.0
        return preds[0][0], float(preds[0][1])

    items = []
    for _, yconf, area, m_bin in candidates:
        ys, xs = np.where(m_bin)
        if len(ys) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad_y = max(4, int((y1 - y0) * 0.08))
        pad_x = max(4, int((x1 - x0) * 0.08))
        y0 = max(0, y0 - pad_y); y1 = min(H, y1 + pad_y)
        x0 = max(0, x0 - pad_x); x1 = min(W, x1 + pad_x)
        crop = pil_img.crop((x0, y0, x1, y1))

        label, p = _classify_crop(crop)
        if not label:
            continue
        if p < min_conf:
            continue
        items.append({'name': label, 'conf': float(p), 'area': float(area)})

    if not items:
        return []

    # Merge duplicate labels by summing area and keeping max confidence
    merged = {}
    for it in items:
        k = it['name']
        if k not in merged:
            merged[k] = dict(it)
        else:
            merged[k]['area'] += float(it.get('area', 0.0))
            merged[k]['conf'] = max(float(merged[k].get('conf', 0.0)), float(it.get('conf', 0.0)))

    out = list(merged.values())
    out.sort(key=lambda x: x.get('area', 0.0), reverse=True)
    return out


def _detect_ingredient_components(pil_img: Image.Image,
                                 max_items: int = 18,
                                 min_conf: float = 0.15,
                                 min_area: float = 0.001,
                                 min_score: float = 0.01):
    """Detect ingredient/component instances using a fine-tuned YOLOv8-seg model.

    Returns list[dict{name, conf, area}] sorted by area desc.
    - `area` is fraction of image pixels covered by the instance mask.
    """
    if ingredient_yolo_model is None:
        return []

    img_rgb = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # Important: Ultralytics applies a confidence filter internally.
    # Pass our `min_conf` so weaker-but-real ingredients (e.g., rice) aren’t dropped.
    try:
        try:
            results = ingredient_yolo_model.predict(img_bgr, verbose=False, conf=float(min_conf), iou=0.5)[0]
        except Exception:
            results = ingredient_yolo_model(img_bgr, verbose=False, conf=float(min_conf), iou=0.5)[0]
    except Exception:
        try:
            results = ingredient_yolo_model(img_bgr, verbose=False)[0]
        except Exception:
            return []

    if results.masks is None or len(results.masks) == 0:
        return []

    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()

    items = []
    for i in range(len(classes)):
        c = int(classes[i])
        p = float(confs[i])
        if p < min_conf:
            continue

        m = results.masks[i].data[0].cpu().numpy()
        m = cv2.resize(m, (W, H))
        m_bin = (m > 0.5)
        area = float(m_bin.mean())
        if area < min_area:
            continue

        # A single low-confidence tiny mask is almost always noise.
        # Using score = area × confidence keeps large weak detections (e.g., rice)
        # while removing many spurious small ingredients.
        if (area * p) < float(min_score):
            continue

        name = ingredient_yolo_model.names.get(c, str(c))

        # Many YOLO datasets are trained with numeric names ("0", "1", ...).
        # If we have a mapping file, use it to show real ingredient names.
        try:
            name_s = str(name).strip()
        except Exception:
            name_s = str(c)
        if (not name_s) or name_s.isdigit() or name_s.lower() == str(c):
            if INGREDIENT_LABELS and c in INGREDIENT_LABELS:
                name_s = str(INGREDIENT_LABELS[c]).strip()

        name = name_s.replace('_', ' ').replace('-', ' ').strip().title()
        if not name:
            name = str(c)
        items.append({'name': name, 'conf': p, 'area': area})

    if not items:
        return []

    # Merge duplicates by label
    merged = {}
    for it in items:
        k = it['name']
        if k not in merged:
            merged[k] = dict(it)
        else:
            merged[k]['area'] += float(it.get('area', 0.0))
            merged[k]['conf'] = max(float(merged[k].get('conf', 0.0)), float(it.get('conf', 0.0)))

    out = list(merged.values())
    out.sort(key=lambda x: x.get('area', 0.0), reverse=True)
    return out[:max_items]


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


# Inference

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    if image is None:
        return 'Please upload an image.', None, '<div style="color:var(--muted); padding:20px;">No image uploaded.</div>', '', _gpu_stats_md(), []

    if pipeline_mode is None:
        return ('No model checkpoint found. Download best_mlp.pt + mlp_feat_stats.npz '
                'from Kaggle output and place them in models/.'), None, '<div style="color:var(--danger); padding:20px;">Error: No model loaded.</div>', '', _gpu_stats_md(), []

    _timing = {}  # type: dict
    _t0 = time.perf_counter()

    # Safely initialize variables so Phase 3 baseline fallback won't crash the HTML cards
    detected_food = None
    food_conf = None
    mode_note = ''
    w_mean = None

    if pipeline_mode in ('full', 'phase6', 'phase4'):
        # Step 1: YOLO mask + MiDaS depth -> 9 features
        _ts = time.perf_counter()
        feat_raw, items, raw_mask = _extract_features(image)
        _timing['YOLO seg + MiDaS depth (feature extraction)'] = (time.perf_counter() - _ts) * 1000
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

        # Step 2: classify food type
        _ts = time.perf_counter()
        food_type_preds = _classify_food_type(image, raw_mask, top_k=3)
        _timing['EfficientNet-B0 food classification (Phase 5)'] = (time.perf_counter() - _ts) * 1000
        detected_food   = food_type_preds[0][0] if food_type_preds else None
        food_conf       = float(food_type_preds[0][1]) if food_type_preds else None

        _simple_food_override = False

        # If the classifier is unsure, hide the dish label rather than show a wrong one.
        if detected_food and (food_conf is None or food_conf < DISH_LABEL_MIN_CONF):
            detected_food = None

        # Multi-item list (YOLO region proposals + Food-101 classification per region)
        _ts = time.perf_counter()
        item_foods = _detect_and_classify_items(image)
        _timing['Multi-item Food-101 (YOLO regions)'] = (time.perf_counter() - _ts) * 1000

        # Ingredient/component segmentation list (optional, if weights present)
        _ts = time.perf_counter()
        ingredient_items = _detect_ingredient_components(image)
        _timing['Ingredient instance-seg (optional)'] = (time.perf_counter() - _ts) * 1000

        # FoodSeg103 override: if a single fruit dominates, prefer it over Food-101.
        _fs_fruit = _maybe_foodseg103_single_fruit_override(ingredient_items)
        if _fs_fruit is not None:
            detected_food, food_conf = _fs_fruit
            food_type_preds = [(detected_food, float(food_conf or 1.0))]
            _simple_food_override = True
            ingredient_items = [{'name': detected_food, 'conf': float(food_conf or 1.0), 'area': 1.0}]

        # YOLO-name fallback for foods absent from Food-101
        # Food-101 has no plain fruit categories (apple, banana, orange, grape…).
        # When EfficientNet confidence is low AND YOLO directly detected a known
        # food class (especially fruits), prefer the YOLO label so USDA lookup
        # can fire with the correct food name.
        _YOLO_ONLY_FOODS = {
            'banana', 'apple', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'sandwich',
        }
        if items:
            # Strong override for simple fruits: Food-101 doesn't have them, and
            # will often map them to a random dessert with high confidence.
            _SIMPLE_FRUITS = {'banana', 'apple', 'orange'}
            _yolo_fruit = max(
                (it for it in items if str(it.get('name', '')).lower() in _SIMPLE_FRUITS),
                key=lambda x: (float(x.get('area', 0.0)), float(x.get('conf', 0.0))),
                default=None,
            )
            if _yolo_fruit and float(_yolo_fruit.get('area', 0.0)) >= 0.12 and float(_yolo_fruit.get('conf', 0.0)) >= 0.35:
                detected_food = str(_yolo_fruit.get('name', ''))
                food_conf     = float(_yolo_fruit.get('conf', 0.0))
                _simple_food_override = True
            elif (food_conf is None or food_conf < 0.15):
                # Pick the highest-confidence YOLO detection that matches a known food
                _yolo_food = max(
                    (it for it in items if it['name'].lower() in _YOLO_ONLY_FOODS),
                    key=lambda x: x['conf'],
                    default=None,
                )
                if _yolo_food:
                    detected_food = _yolo_food['name']   # e.g. 'Banana', 'Apple'
                    food_conf     = _yolo_food['conf']   # YOLO detection confidence

        # If we overrode the dish label with a simple fruit, force a single-ingredient breakdown.
        if _simple_food_override and detected_food:
            ingredient_items = [{'name': detected_food, 'conf': float(food_conf or 1.0), 'area': 1.0}]

        # If we have multi-item labels but no main dish label, pick the largest item.
        if detected_food is None and item_foods:
            detected_food = item_foods[0]['name']
            food_conf = float(item_foods[0].get('conf', 0.0))

        # Step 3: predict weight -> food-specific constants
        p6_mean = p6_std = None
        w_mean  = None
        w_std_mc = 0.0
        _weight_note = ''
        _weight_breakdown_md = ''   # filled in when WeightMLP is available
        if weight_mlp_model is not None:
            weight_mlp_model.eval()
            for m in weight_mlp_model.modules():
                if isinstance(m, nn.Dropout): m.train()
            torch.manual_seed(42)
            _w_samples = []
            _ts = time.perf_counter()
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    _raw_s = weight_mlp_model(x6).item()
                    if _weight_log_target:
                        _raw_s = float(np.exp(_raw_s) - _weight_log_offset)
                    _w_samples.append(max(_raw_s, 10.0))   # clamp to 10g
            _timing[f'WeightMLP MC-Dropout ×{MC_SAMPLES} passes (Phase 6)'] = (time.perf_counter() - _ts) * 1000
            weight_mlp_model.eval()
            w_raw = float(np.mean(_w_samples));  w_std_mc = float(np.std(_w_samples))

            # Un-normalised features (for heuristics + UI explanation)
            _feat_unnorm = None
            if feat_mean is not None and feat_std is not None:
                _feat_unnorm = (feat_raw * feat_std.flatten() + feat_mean.flatten())

            DATASET_MEAN_WEIGHT = 280.0
            MIN_WEIGHT          = 50.0

            # Nutrition DB lookup (detect -> portion -> database)
            # Do this BEFORE the fallback decision so we can use the USDA serving
            # size as a plausibility check on the WeightMLP prediction.
            _db_entry     = None
            _food_scale_note = ''
            if detected_food and food_conf and food_conf >= 0.15:
                _query    = _food_name_to_key(detected_food)
                _db_entry = _lookup_usda(_query)

            # Weight estimation strategy
            # MiDaS produces *relative* depth (0–1), so WeightMLP cannot learn
            # absolute grams reliably — real-world camera distance varies too much.
            # Best strategy: use USDA/restaurant serving as the absolute anchor and
            # treat the MLP output as a *relative scale factor* encoding "bigger or
            # smaller than the average 280g Nutrition5K dish".
            #
            #   w_final = w_usda × clip(w_mlp / DATASET_MEAN_WEIGHT, 0.4, 2.5)
            #
            # This is strictly better than the old binary "MLP vs USDA" switch:
            #   • When MLP predicts 420g (1.5×mean) for a heaped bowl, we scale up.
            #   • When MLP predicts 140g (0.5×mean) for a small plate, we scale down.
            #   • Clip [0.4, 2.5] prevents runaway scales from bad depth estimates.
            # Hard-floor fallback (no USDA) stays intact as the last resort.

            _usda_serving = float(_db_entry[4]) if _db_entry is not None else None
            # If USDA is unavailable (no API key / lookup miss), still anchor to a
            # plausible meal portion when we have a detected food type.
            _anchor_serving = (
                _usda_serving
                if _usda_serving is not None
                else (_restaurant_serving_g(detected_food) if detected_food else None)
            )
            _cv_fallback  = (w_std_mc > abs(w_raw) and w_raw < DATASET_MEAN_WEIGHT)

            if w_raw < MIN_WEIGHT or _cv_fallback:
                # MLP output is completely unreliable — ignore it
                _used_fallback = True
                if _anchor_serving is not None:
                    w_mean = float(_anchor_serving)
                    _scale_factor = 1.0
                else:
                    w_mean = DATASET_MEAN_WEIGHT
                    _scale_factor = 1.0
            elif _anchor_serving is not None:
                # Anchor (USDA if available, else restaurant default) + relative scale.
                # We blend two weak scale cues:
                #   1) WeightMLP raw / dataset mean (learned, but domain-shifts on phone photos)
                #   2) Mask area vs dataset mean mask area (simple geometry cue)
                # Geometric mean is a conservative blend.
                _mlp_scale  = float(w_raw / DATASET_MEAN_WEIGHT)
                _area_raw   = float(_feat_unnorm[0]) if _feat_unnorm is not None else None
                _area_scale = (float(_area_raw / float(feat_mean.flatten()[0]))
                               if (_area_raw is not None and feat_mean is not None)
                               else 1.0)
                _area_scale = float(np.clip(_area_scale, 0.6, 1.8))
                _scale_factor = float(np.clip(np.sqrt(max(_mlp_scale, 1e-6) * _area_scale), 0.4, 2.5))
                w_mean = float(_anchor_serving) * _scale_factor
                _used_fallback = False
            else:
                # No USDA entry at all — use raw MLP (last resort)
                _scale_factor = 1.0
                w_mean = max(w_raw, MIN_WEIGHT)
                _used_fallback = False

            # Build per-target nutrition from DB (all 4 macros) or fall back to
            # global constants scaled by kcal ratio (old behaviour)
            const_keys = list(nutrition_constants.keys())
            if _db_entry is not None:
                # USDA lookup: each macro has its own per-gram value
                # USDA tuple: (kcal/g, fat/g, protein/g, carb/g, serving_g)
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
                _food_scale_note = (f' · USDA: {_db_kcal:.2f} kcal/g, '
                                    f'{_db_fat:.3f} fat/g, '
                                    f'{_db_pro:.3f} pro/g, '
                                    f'{_db_carb:.3f} carb/g')
            else:
                # No DB match — fall back to global Nutrition5K constants
                active_constants = dict(nutrition_constants)

            p6_mean = np.array([w_mean * active_constants[k] for k in const_keys], dtype=np.float32)
            p6_std  = np.array([w_std_mc * active_constants[k] for k in const_keys], dtype=np.float32)
            if _used_fallback:
                _src = 'Serving anchor (MLP unreliable)'
            elif _db_entry is not None:
                _src = f'USDA×{_scale_factor:.2f} (blended scale)'
            elif detected_food:
                _src = f'Restaurant×{_scale_factor:.2f} (blended scale)'
            else:
                _src = 'WeightMLP (no anchor)'
            _weight_note = f'**{w_mean:.0f}±{w_std_mc*2:.0f}g** ({_src}){_food_scale_note}'

            # Build weight breakdown explanation for UI display
            _FEAT_NAMES = [
                ('mask_area', 'Food mask fraction (0–1)    YOLO segmentation area'),
                ('d_mean',    'Mean depth — whole image     MiDaS relative depth'),
                ('d_std',     'Std depth — whole image      MiDaS spread'),
                ('d_med',     'Median depth — whole image   MiDaS robust centre'),
                ('d_max',     'Max depth — whole image      MiDaS furthest point'),
                ('md_mean',   'Mean depth — food mask       depth of food pixels'),
                ('md_std',    'Std depth — food mask        food depth spread'),
                ('md_med',    'Median depth — food mask     food robust centre'),
                ('md_max',    'Max depth — food mask        highest food point'),
            ]
            # Un-normalise feat_raw back to physical scale for readability
            _feat_unnorm = (feat_raw * feat_std.flatten() + feat_mean.flatten())
            _feat_rows = '\n'.join(
                f'| `{nm}` | {_feat_unnorm[i]:.4f} | {desc} |'
                for i, (nm, desc) in enumerate(_FEAT_NAMES)
            )

            # MC Dropout distribution
            _w_arr = np.array(_w_samples)
            _w_lo  = float(np.percentile(_w_arr, 2.5))
            _w_hi  = float(np.percentile(_w_arr, 97.5))

            # Step-3 explanation for UI
            if _used_fallback:
                if _cv_fallback:
                    _fb_why = (f'High MC-Dropout uncertainty '
                               f'(σ = {w_std_mc:.0f} g > raw mean {w_raw:.0f} g) — unreliable prediction')
                else:
                    _fb_why = f'Raw prediction {w_raw:.0f} g ≤ {MIN_WEIGHT:.0f} g hard floor'
                _step3_txt = (f'⚠️ **MLP unreliable**: {_fb_why}  \n'
                              f'Using **USDA/restaurant serving = {w_mean:.0f} g** instead')
            elif _db_entry is not None:
                _step3_txt = (
                    f'✔️ **USDA-anchored scale**: USDA serving {_usda_serving:.0f} g '
                    f'× blended scale {_scale_factor:.2f}  \n'
                    f'- MLP scale: {w_raw:.0f} g ÷ {DATASET_MEAN_WEIGHT:.0f} g = {_mlp_scale:.2f}  \n'
                    f'- Mask-area scale (vs dataset mean): {_area_scale:.2f}  \n'
                    f'- Final: sqrt(MLP × area), clipped to [0.4, 2.5]  \n'
                    f'→ **{w_mean:.0f} g**'
                )
            elif detected_food and _anchor_serving is not None:
                _step3_txt = (
                    f'✔️ **Restaurant-anchored scale**: restaurant serving {_anchor_serving:.0f} g '
                    f'× blended scale {_scale_factor:.2f}  \n'
                    f'- MLP scale: {w_raw:.0f} g ÷ {DATASET_MEAN_WEIGHT:.0f} g = {_mlp_scale:.2f}  \n'
                    f'- Mask-area scale (vs dataset mean): {_area_scale:.2f}  \n'
                    f'- Final: sqrt(MLP × area), clipped to [0.4, 2.5]  \n'
                    f'→ **{w_mean:.0f} g**'
                )
            else:
                _step3_txt = (f'✔️ **WeightMLP raw** (no anchor) — {w_mean:.0f} g')

            # Per-macro calorie equation rows
            _density_src = (f'USDA "{_food_name_to_key(detected_food)}"'
                            if _db_entry is not None else 'Nutrition5K global constants')
            _cal_rows = '\n'.join(
                f'| **{k.replace("_per_g","")}** | {w_mean:.0f} g | × {v:.4f} /g | **{w_mean*v:.1f}** |'
                for k, v in active_constants.items()
            )

            _weight_breakdown_md = (
                f'### ⚖ How the weight was predicted\n\n'
                f'**Step 1 — Geometric features** (YOLO segmentation mask + MiDaS monocular depth map):\n\n'
                f'| Feature | Value | Description |\n'
                f'|---|---|---|\n'
                f'{_feat_rows}\n\n'
                f'**Step 2 — WeightMLP MC-Dropout** (30 stochastic forward passes, dropout kept active):  \n'
                f'- Raw mean: **{w_raw:.0f} g** \u00b7 σ = {w_std_mc:.0f} g  \n'
                f'- 95% credible interval: [{_w_lo:.0f} g — {_w_hi:.0f} g]  \n'
                f'- Architecture: Linear(9→128) → BN → ReLU → Dropout ×3 → Linear(32→1)  \n'
                f'- ⚠️ MiDaS depth is **relative** (0–1), so raw grams are unreliable.  \n'
                f'  MLP output is used as a **relative scale cue** on a serving anchor (USDA if available, else restaurant default), blended with mask-area:  \n'
                f'  `scale = sqrt((w_MLP/280) × (mask_area/mask_area_mean))` then clip [0.4, 2.5]  \n'
                f'  `w_final = w_anchor × scale`\n\n'
                f'**Step 3 — Serving-size decision**:  \n{_step3_txt}  \n'
                f'**Final weight used: {w_mean:.0f} g**\n\n'
                f'**Step 4 — Nutrition equation** (density source: {_density_src}):  \n'
                f'`nutrition = weight × density`\n\n'
                f'| Macro | Weight | Density | Result |\n'
                f'|---|---|---|---|\n'
                f'{_cal_rows}\n'
            )

        # Step 4: direct regression via MLP
        p4_mean = p4_std = None
        if mlp is not None:
            mlp.eval()
            for m in mlp.modules():
                if isinstance(m, nn.Dropout): m.train()
            torch.manual_seed(42)
            _p4_mc = []
            _ts = time.perf_counter()
            with torch.no_grad():
                for _ in range(MC_SAMPLES):
                    _p4_mc.append(mlp(x4).squeeze(0).cpu().numpy())
            _timing[f'NutritionMLP MC-Dropout ×{MC_SAMPLES} passes (Phase 4)'] = (time.perf_counter() - _ts) * 1000
            mlp.eval()
            _p4_mc = np.stack(_p4_mc)
            p4_mean = _p4_mc.mean(0)
            p4_std  = _p4_mc.std(0)

        # Step 5: Ensemble Phase 4 + Phase 6
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

        # Step 6: ResNet baseline comparison
        p3_note = ''
        if p3_model is not None:
            _ts = time.perf_counter()
            _img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                _p3_pred = p3_model(_img_t).squeeze(0).cpu().numpy()
            _timing['ResNet-50 baseline comparison (Phase 3)'] = (time.perf_counter() - _ts) * 1000
            cal_idx3 = next((i for i, c in enumerate(target_cols) if 'cal' in c.lower()), 0)
            p3_note  = f'\n_Phase 3 baseline (ResNet-50): {_p3_pred[cal_idx3]:.0f} kcal_'

        # Build outputs
        cal_idx  = next((i for i, c in enumerate(target_cols) if 'cal' in c.lower()), 0)
        total_cal = float(mean_pred[cal_idx])

        # Format weight string for the table: "250 g" or "250±66 g"
        _w_str = f'{w_mean:.0f}±{w_std_mc*2:.0f} g' if w_mean else '—'

        # Prefer ingredient/component segmentation for breakdown.
        # Only treat rows as "ingredients" when ingredient_items are present.
        def _clean_ingredient_items(_its):
            if not _its:
                return []
            out = []
            for it in _its:
                if not it:
                    continue
                name = str(it.get('name', '')).strip()
                if not name:
                    continue
                if name.strip().lower() in {'other ingredients', 'other ingredient', 'other'}:
                    continue
                out.append(it)
            return out

        def _truncate_by_coverage(_its, min_cov: float = 0.9, max_keep: int = 8):
            if not _its:
                return []
            its_sorted = sorted(_its, key=lambda x: float(x.get('area', 0.0)), reverse=True)
            total = float(sum(float(it.get('area', 0.0)) for it in its_sorted) or 0.0)
            if total <= 0:
                return its_sorted[:max_keep]
            kept = []
            acc = 0.0
            for it in its_sorted:
                kept.append(it)
                acc += float(it.get('area', 0.0))
                if len(kept) >= max_keep or (acc / total) >= min_cov:
                    break
            return kept

        def _ingredient_seg_is_credible(_its) -> bool:
            if not _its:
                return False
            try:
                its_sorted = sorted(_its, key=lambda x: float(x.get('area', 0.0)), reverse=True)
                total = float(sum(float(it.get('area', 0.0)) for it in its_sorted) or 0.0)
                if total <= 0:
                    return False
                top = float(its_sorted[0].get('area', 0.0))
                top_frac = top / total
                # If there are many components but none dominates, it’s usually noise.
                if len(its_sorted) >= 10 and top_frac < 0.28:
                    return False
                # If nothing is meaningfully sized, don’t trust it.
                if top < 0.02:
                    return False
                return True
            except Exception:
                return False

        ingredient_items_clean = _clean_ingredient_items(ingredient_items)
        ingredient_items_clean = _truncate_by_coverage(ingredient_items_clean, min_cov=0.9, max_keep=8)

        # Safety gate: ingredient segmentation can hallucinate out-of-domain components.
        # Only show a breakdown if:
        #  - we have at least 2 components, OR
        #  - we have exactly 1 component and it matches the dish label (e.g., single-fruit override)
        has_ingredient_seg = False
        if ingredient_items_clean:
            if not _ingredient_seg_is_credible(ingredient_items_clean):
                has_ingredient_seg = False
            elif len(ingredient_items_clean) >= 2:
                has_ingredient_seg = True
            elif detected_food and str(ingredient_items_clean[0].get('name', '')).strip().lower() == str(detected_food).strip().lower():
                has_ingredient_seg = True

        items_for_breakdown = ingredient_items_clean if has_ingredient_seg else (item_foods if item_foods else items)

        # Food-101 is often out-of-domain on real mixed plates.
        # General sanity guard (not dish-specific): if the dish label implies a
        # specific protein/seafood but ingredient segmentation suggests a
        # different protein set (or none), hide the dish label.
        if detected_food and has_ingredient_seg and not _simple_food_override:
            try:
                dish_l = str(detected_food).strip().lower()
                ing_names_l = [
                    str(it.get('name', '')).strip().lower()
                    for it in (ingredient_items_clean or [])
                    if it and str(it.get('name', '')).strip()
                ]

                # Broad protein vocabulary (category-level, not tied to one dish).
                _SEAFOOD = {
                    'salmon', 'tuna', 'fish', 'shrimp', 'prawn', 'crab', 'lobster',
                    'calamari', 'squid', 'octopus', 'mussel', 'mussels', 'oyster',
                    'oysters', 'clam', 'clams', 'scallop', 'scallops',
                }
                _MEAT = {
                    'chicken', 'turkey', 'duck', 'beef', 'steak', 'pork', 'ham',
                    'bacon', 'lamb', 'mutton', 'goat',
                }
                _PROTEIN = _SEAFOOD | _MEAT

                def _hits(text: str, vocab: set[str]) -> set[str]:
                    out = set()
                    for tok in vocab:
                        if tok in text:
                            out.add(tok)
                    return out

                dish_prot = _hits(dish_l, _PROTEIN)
                ing_prot = set()
                for nm in ing_names_l:
                    ing_prot |= _hits(nm, _PROTEIN)

                # If dish implies a specific protein but ingredients imply different ones,
                # the dish label is likely wrong.
                if dish_prot:
                    # If we have no protein evidence at all from ingredients, be conservative:
                    # only suppress when the dish label is very specific seafood/meat.
                    if not ing_prot:
                        detected_food = None
                        food_conf = None
                        food_type_preds = []
                    else:
                        # Suppress when disjoint (e.g., dish says salmon but ingredients show pork/chicken).
                        if dish_prot.isdisjoint(ing_prot):
                            detected_food = None
                            food_conf = None
                            food_type_preds = []
            except Exception:
                pass

        if detected_food:
            # Dish-level label (Food-101 classifier is single-label).
            ingredient_rows = [[detected_food, _w_str, f"{total_cal:.0f}", f"{food_conf*100:.0f}%"]]
            for alt_name, alt_conf in (food_type_preds[1:3] if food_type_preds else []):
                ingredient_rows.append([f'  (alt) {alt_name}', '—', '—', f"{alt_conf*100:.0f}%"])

            # If we have ingredient/component segmentation, compute a nutrition
            # breakdown using USDA per-ingredient densities.
            # To guarantee the row calories sum to the displayed total calories,
            # we compute raw per-ingredient calories and then rescale.
            if items_for_breakdown:
                total_area = float(sum(it.get('area', 0.0) for it in items_for_breakdown) or 1.0)

                # Compute raw per-ingredient calories
                raw_rows = []
                raw_sum = 0.0
                for it in sorted(items_for_breakdown, key=lambda x: x.get('area', 0.0), reverse=True):
                    name = str(it.get('name', 'Item')).strip()
                    if name.strip().lower() == detected_food.strip().lower():
                        continue
                    frac = float(it.get('area', 0.0)) / total_area
                    conf_str = f"{it.get('conf', 0.0)*100:.0f}%"

                    # Allocate grams by area if weight is available.
                    grams = float(w_mean * frac) if w_mean else None

                    kcal_per_g = None
                    if name:
                        _entry = _lookup_usda(_food_name_to_key(name))
                        if _entry is not None:
                            kcal_per_g = float(_entry[0])

                    if grams is not None and kcal_per_g is not None:
                        cal_raw = float(grams * kcal_per_g)
                    else:
                        # No weight or no USDA density — fall back to area-only split
                        # so we can still provide a consistent breakdown.
                        cal_raw = float(total_cal * frac)

                    raw_rows.append((name, grams, conf_str, cal_raw))
                    raw_sum += float(cal_raw)

                # Rescale so sum(rows) == total_cal
                scale = float(total_cal / raw_sum) if raw_sum > 1e-6 else 1.0

                for (name, grams, conf_str, cal_raw) in raw_rows:
                    item_w_str = f"{grams:.0f} g" if grams is not None else '—'
                    ingredient_rows.append([
                        f"• {name}",
                        item_w_str,
                        f"{(cal_raw * scale):.0f}",
                        conf_str,
                    ])

        elif items_for_breakdown:
            # No dish classification — show each YOLO food item.
            total_area = float(sum(it.get('area', 0.0) for it in items_for_breakdown) or 1.0)
            ingredient_rows = []
            if has_ingredient_seg:
                # Ingredient/component segmentation → USDA per-ingredient densities.
                # Compute raw per-ingredient calories and then rescale so sum(rows)=total_cal.
                raw_rows = []
                raw_sum = 0.0
                for it in sorted(items_for_breakdown, key=lambda x: x.get('area', 0.0), reverse=True):
                    name = str(it.get('name', 'Item')).strip()
                    frac = float(it.get('area', 0.0)) / total_area
                    conf_str = f"{it.get('conf', 0.0)*100:.0f}%"

                    grams = float(w_mean * frac) if w_mean else None
                    kcal_per_g = None
                    if name:
                        _entry = _lookup_usda(_food_name_to_key(name))
                        if _entry is not None:
                            kcal_per_g = float(_entry[0])

                    if grams is not None and kcal_per_g is not None:
                        cal_raw = float(grams * kcal_per_g)
                    else:
                        cal_raw = float(total_cal * frac)

                    raw_rows.append((name, grams, conf_str, cal_raw))
                    raw_sum += float(cal_raw)

                scale = float(total_cal / raw_sum) if raw_sum > 1e-6 else 1.0
                for (name, grams, conf_str, cal_raw) in raw_rows:
                    item_w_str = f"{grams:.0f} g" if grams is not None else '—'
                    ingredient_rows.append([
                        f"• {name}" if name else 'Item',
                        item_w_str,
                        f"{(cal_raw * scale):.0f}",
                        conf_str,
                    ])
            else:
                for it in sorted(items_for_breakdown, key=lambda x: x.get('area', 0.0), reverse=True):
                    frac = float(it.get('area', 0.0)) / total_area
                    conf_str = f"{it.get('conf', 0.0)*100:.0f}%"
                    item_w_str = f"{(w_mean * frac):.0f} g" if w_mean else _w_str
                    ingredient_rows.append([it.get('name', 'Item'), item_w_str, f"{total_cal * frac:.0f}", conf_str])

        else:
            ingredient_rows = [['Unknown dish', _w_str, f"{total_cal:.0f}", '—']]

        result_json_display = {col: f'{mu:.1f} ± {sig*2:.1f}' for col, mu, sig in zip(target_cols, mean_pred, std_pred)}
        mode_note = (f'_Pipeline: Phase 6 — YOLOv8-seg + MiDaS → WeightMLP · '
                     f'Phase 4 — YOLO+MiDaS → MLP · Phase 5 — EfficientNet-B0_'
                     f'{ensemble_note}{p3_note}')

    else:  # phase3 only fallback
        _ts = time.perf_counter()
        img_t = val_transform(image.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            mean_pred = p3_model(img_t).squeeze(0).cpu().numpy()
        _timing['ResNet-50 direct regression (Phase 3)'] = (time.perf_counter() - _ts) * 1000
        result_json_display = {col: f'{v:.1f}' for col, v in zip(target_cols, mean_pred)}
        mode_note  = '_Pipeline: ResNet-50 direct regression (Phase 3 baseline)_'
        ingredient_rows = [['N/A (ResNet baseline — no segmentation)', '—', '—', '—']]
        _weight_breakdown_md = '_Weight prediction not available in Phase 3 baseline (ResNet-50 direct regression, no YOLO/MiDaS)._'

    _timing['Total inference (wall-clock)'] = (time.perf_counter() - _t0) * 1000
    gpu_md = _gpu_stats_md(_timing)

    cards_html = '<div style="display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;">'
    for col in target_cols:
        val = result_json_display[col]
        cards_html += f'''
        <div style="background: var(--surface2); padding: 16px; border-radius: var(--radius); border: 1px solid var(--border); flex: 1; min-width: 100px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
            <div style="font-family: 'Outfit', sans-serif; font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px;">{col}</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: var(--purple-dark); font-weight: 800;">{val}</div>
        </div>
        '''
    cards_html += '</div>'

    clean_mode = mode_note.replace('_', '')
    if detected_food:
        clean_clf = f'<div style="font-size: 1rem; margin-bottom: 16px; color: var(--text);">🍽 Detected food: <strong>{detected_food}</strong> ({food_conf*100:.0f}% confidence)</div>'
    else:
        clean_clf = ''

    label_txt = f"""
    {clean_clf}
    {cards_html}
    <div style="background: var(--purple-dim); border-radius: var(--radius); padding: 12px 16px; font-size: 0.8rem; color: var(--text); line-height: 1.5; margin-bottom: 12px; border: 1px solid rgba(123,110,246,0.15);">
        <strong style="color: var(--purple-dark);">⚙️ Pipeline Details:</strong><br> {clean_mode}
    </div>
    <div style="font-size: 0.8rem; color: var(--muted); border-left: 3px solid var(--purple); padding-left: 10px;">
        Units: kcal for calories, grams for macros
    </div>
    """

    table_html = _render_table_html(ingredient_rows)

    ingredient_rows_struct = []
    for dish, weight, cals, conf in ingredient_rows:
        ingredient_rows_struct.append({
            'dish': dish,
            'weight': weight,
            'cals': cals,
            'conf': conf,
            'weight_g': _parse_first_float(weight),
            'cal_kcal': _parse_first_float(cals),
        })

    base_weight_g = w_mean
    if base_weight_g is None:
        base_weight_g = _parse_first_float(ingredient_rows[0][1]) if ingredient_rows else None

    base_mean = {col: float(mu) for col, mu in zip(target_cols, mean_pred)}
    try:
        base_ci = {col: float(sig * 2) for col, sig in zip(target_cols, std_pred)}
    except Exception:
        base_ci = {col: 0.0 for col in target_cols}

    result_json_out = dict(result_json_display)
    result_json_out['_meta'] = {
        'target_cols': list(target_cols),
        'detected_food': detected_food,
        'food_conf': food_conf,
        'pipeline_details': clean_mode,
        'base_weight_g': base_weight_g,
        'base_mean': base_mean,
        'base_ci': base_ci,
        'ingredient_rows': ingredient_rows_struct,
        'base_label_html': label_txt,
        'base_table_html': table_html,
    }

    portion_defaults = []
    for r in ingredient_rows_struct:
        try:
            dish = str(r.get('dish', '')).strip()
        except Exception:
            dish = ''
        g = r.get('weight_g')
        try:
            g = float(g) if g is not None else None
        except Exception:
            g = None
        if not dish:
            continue
        # Allow rows without a numeric weight, but keep them editable
        portion_defaults.append([dish, int(round(g)) if g is not None else 0, 'g'])

    return label_txt, result_json_out, table_html, _weight_breakdown_md, gpu_md, portion_defaults


