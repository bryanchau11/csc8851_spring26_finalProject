"""
Nutrition5K — Gradio Demo App
Upload a food dish image → get predicted Calories, Fat, Protein, Carbohydrates.

Usage:
    pip install gradio torch torchvision pillow
    python app/app.py

Requires:
    models/best_model.pt   (download from Kaggle output after training)
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH   = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pt')
BACKBONE    = 'resnet50'   # must match what was used during training
IMG_SIZE    = 224
TARGET_COLS = ['calories', 'fat', 'protein', 'carbs']  # overridden from checkpoint

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else
                      'cpu')

# ── Model definition (must match 03_model.ipynb exactly) ─────────────────────
class NutritionEstimator(nn.Module):
    def __init__(self, num_targets: int, backbone: str = 'resnet50'):
        super().__init__()
        if backbone == 'resnet50':
            base = models.resnet50(weights=None)
            in_feats = base.fc.in_features          # 2048
            base.fc  = nn.Identity()
        elif backbone == 'efficientnet_b2':
            base = models.efficientnet_b2(weights=None)
            in_feats = base.classifier[1].in_features  # 1408
            base.classifier = nn.Identity()
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_targets),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_model():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(
            f'Checkpoint not found at {CKPT_PATH}\n'
            'Download best_model.pt from Kaggle output and place it in models/'
        )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cols = ckpt.get('target_cols', TARGET_COLS)
    mdl  = NutritionEstimator(num_targets=len(cols), backbone=BACKBONE).to(device)
    mdl.load_state_dict(ckpt['model_state_dict'])
    mdl.eval()
    print(f'✓ Loaded checkpoint (epoch {ckpt.get("epoch", "?")})')
    return mdl, cols

model, target_cols = load_model()

# ── Inference transform (same as val_transform in training) ───────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Prediction function ───────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return {col: 'No image provided' for col in target_cols}
    img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor).squeeze(0).cpu().numpy()
    return {col: f'{val:.1f}' for col, val in zip(target_cols, preds)}

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title='Nutrition5K Estimator') as demo:
    gr.Markdown(
        '# 🥗 Nutrition5K — Dish Nutrition Estimator\n'
        'Upload a photo of a food dish to predict its **calories, fat, protein, and carbohydrates**.'
    )
    with gr.Row():
        img_input  = gr.Image(type='pil', label='Upload Dish Image')
        output_box = gr.JSON(label='Predicted Nutrition (raw units from training data)')

    predict_btn = gr.Button('Predict', variant='primary')
    predict_btn.click(fn=predict, inputs=img_input, outputs=output_box)

    gr.Markdown(
        '> **Note**: Predictions are in the same units as the Nutrition5K dataset '
        '(calories in kcal, macros in grams).'
    )

if __name__ == '__main__':
    demo.launch(share=False)
