"""
Skin Lesion Prediction Web App  (PAD-UFES-20, MobileNetV2)

Loads the model you trained in Colab and serves a web page where you
upload a skin image and get the predicted lesion type + confidence.

SETUP (once):
  1. Download 'best_mobilenet_v2.pth' from your Google Drive folder
     'pad-ufes-20-results' and put it next to this file (project root).
  2. pip install flask torch torchvision pillow
RUN:
  python predict_app.py
  then open  http://127.0.0.1:5001  in your browser.
"""

import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

# ---- config (must match training) ----
PROJECT_DIR = Path(__file__).parent
MODEL_NAME  = 'mobilenet_v2'
CKPT_PATH   = PROJECT_DIR / f'best_{MODEL_NAME}.pth'
IMG_SIZE    = 224
CLASSES     = ['BCC', 'SCC', 'ACK', 'SEK', 'MEL', 'NEV']
CLASS_INFO  = {
    'BCC': ('Basal Cell Carcinoma', 'cancer'),
    'SCC': ('Squamous Cell Carcinoma', 'cancer'),
    'ACK': ('Actinic Keratosis', 'pre-cancer'),
    'SEK': ('Seborrheic Keratosis', 'benign'),
    'MEL': ('Melanoma', 'cancer'),
    'NEV': ('Nevus / mole', 'benign'),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


def load_model():
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {CKPT_PATH}\n"
            f"Download 'best_{MODEL_NAME}.pth' from your Google Drive "
            f"'pad-ufes-20-results' folder and place it next to predict_app.py."
        )
    m = build_model()
    m.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    m.to(device).eval()
    return m


model = load_model()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'no file uploaded'}), 400
    f = request.files['file']
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
    except Exception as e:
        return jsonify({'ok': False, 'error': f'bad image: {e}'}), 400

    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()

    results = []
    for i, code in enumerate(CLASSES):
        full, kind = CLASS_INFO[code]
        results.append({
            'code': code, 'name': full, 'kind': kind,
            'prob': round(probs[i] * 100, 2),
        })
    results.sort(key=lambda r: r['prob'], reverse=True)
    return jsonify({'ok': True, 'top': results[0], 'all': results})


if __name__ == '__main__':
    print(f"[info] model : {CKPT_PATH}")
    print(f"[info] device: {device}")
    print("[info] open   : http://127.0.0.1:5001")
    app.run(debug=False, port=5001)
