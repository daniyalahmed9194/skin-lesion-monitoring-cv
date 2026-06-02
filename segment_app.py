"""
Skin Lesion Segmentation Web App  (PAD-UFES-20, YOLOv8n-seg)

Loads the YOLOv8 instance-segmentation model trained in Colab and serves a
web page where you upload a skin image and get back a per-instance lesion
mask with class + confidence, a colored overlay, and the total lesion-
coverage percentage.

SETUP (once):
  1. Download 'best_yolov8_seg.pt' from your Google Drive folder
     'pad-ufes-20-results' and put it next to this file (project root).
  2. pip install flask ultralytics pillow opencv-python
RUN:
  python segment_app.py
  then open  http://127.0.0.1:5003  in your browser.
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

# ---- config (must match training) ----
PROJECT_DIR = Path(__file__).parent
CKPT_PATH   = PROJECT_DIR / 'best_yolov8_seg.pt'
IMG_SIZE    = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.45

CLASSES = ['BCC', 'SCC', 'ACK', 'SEK', 'MEL', 'NEV']
CLASS_INFO = {
    'BCC': {'name': 'Basal Cell Carcinoma',     'kind': 'cancer'},
    'SCC': {'name': 'Squamous Cell Carcinoma',  'kind': 'cancer'},
    'MEL': {'name': 'Melanoma',                 'kind': 'cancer'},
    'ACK': {'name': 'Actinic Keratosis',        'kind': 'pre-cancer'},
    'SEK': {'name': 'Seborrheic Keratosis',     'kind': 'benign'},
    'NEV': {'name': 'Nevus',                    'kind': 'benign'},
}
KIND_BGR = {
    'cancer':     (96, 84, 255),    # red-ish
    'pre-cancer': (64, 184, 255),   # amber
    'benign':     (160, 200, 80),   # green
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {CKPT_PATH}\n"
            f"Download 'best_yolov8_seg.pt' from your Google Drive "
            f"'pad-ufes-20-results' folder and place it next to segment_app.py."
        )
    return YOLO(str(CKPT_PATH))


model = load_model()
app = Flask(__name__)


def predict(bgr):
    h, w = bgr.shape[:2]
    res = model.predict(bgr, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
                        device=device, verbose=False)[0]
    items = []
    union_mask = np.zeros((h, w), dtype=np.uint8)
    if res.masks is None or len(res.masks.data) == 0:
        return items, union_mask

    masks   = res.masks.data.cpu().numpy()                  # (N, Hm, Wm) in [0,1]
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs   = res.boxes.conf.cpu().numpy()
    boxes   = res.boxes.xyxy.cpu().numpy().astype(int)

    for mk, ci, cf, bx in zip(masks, cls_ids, confs, boxes):
        mk_bin = (mk > 0.5).astype(np.uint8)
        mk_full = cv2.resize(mk_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        union_mask |= mk_full
        code = CLASSES[ci] if 0 <= ci < len(CLASSES) else str(ci)
        info = CLASS_INFO.get(code, {'name': code, 'kind': 'benign'})
        x1, y1, x2, y2 = bx.tolist()
        items.append({
            'code': code,
            'name': info['name'],
            'kind': info['kind'],
            'prob': float(cf),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'mask': mk_full,
        })
    return items, union_mask


def overlay_image(bgr, detections, alpha=0.45):
    out = bgr.copy()
    layer = np.zeros_like(bgr)
    for d in detections:
        color = np.array(KIND_BGR[d['kind']], dtype=np.uint8)
        layer[d['mask'] > 0] = color
    blended = cv2.addWeighted(out, 1.0, layer, alpha, 0)
    # contour + label per instance
    for d in detections:
        contours, _ = cv2.findContours(d['mask'], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        color_t = tuple(int(c) for c in KIND_BGR[d['kind']])
        cv2.drawContours(blended, contours, -1, color_t, 2)
        x1, y1, x2, y2 = d['bbox']
        label = f"{d['code']} {d['prob']*100:.0f}%"
        scale = max(0.5, min(bgr.shape[:2]) / 900)
        thick = max(1, int(2 * scale))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pad = max(3, int(4 * scale))
        ty = max(y1, th + pad + 2)
        cv2.rectangle(blended, (x1, ty - th - pad),
                      (x1 + tw + 2 * pad, ty + pad), color_t, -1)
        cv2.putText(blended, label, (x1 + pad, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                    cv2.LINE_AA)
    return blended


def encode_jpeg_b64(bgr, quality=90):
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def encode_png_b64(gray):
    ok, buf = cv2.imencode('.png', gray)
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()


@app.route('/')
def index():
    return render_template('segment.html')


@app.route('/api/segment', methods=['POST'])
def api_segment():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'no file uploaded'}), 400
    f = request.files['file']
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
    except Exception as e:
        return jsonify({'ok': False, 'error': f'bad image: {e}'}), 400
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    detections, union = predict(bgr)
    coverage = float((union > 0).sum()) / (h * w) * 100.0
    overlay  = overlay_image(bgr, detections)

    detections_json = [
        {'code': d['code'], 'name': d['name'], 'kind': d['kind'],
         'prob': round(d['prob'], 4), 'bbox': d['bbox']}
        for d in detections
    ]

    return jsonify({
        'ok': True,
        'count': len(detections),
        'coverage': round(coverage, 2),
        'has_lesion': bool(coverage > 0.5),
        'detections': detections_json,
        'overlay': encode_jpeg_b64(overlay),
        'mask': encode_png_b64(union * 255),
    })


if __name__ == '__main__':
    print(f"[info] model : {CKPT_PATH}")
    print(f"[info] device: {device}")
    print("[info] open   : http://127.0.0.1:5003")
    app.run(debug=False, port=5003)
