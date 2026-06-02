"""
Skin Lesion Detection Web App  (PAD-UFES-20, YOLOv8)

Loads the YOLOv8 model trained in Colab and serves a web page where you
upload a skin image and get bounding boxes around detected lesions, with
class + confidence for each.

SETUP (once):
  1. Download 'best_yolov8.pt' from your Google Drive folder
     'pad-ufes-20-results' and put it next to this file (project root).
  2. pip install flask ultralytics pillow
RUN:
  python detect_app.py
  then open  http://127.0.0.1:5002  in your browser.
"""

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

# ---- config (must match training) ----
PROJECT_DIR = Path(__file__).parent
CKPT_PATH   = PROJECT_DIR / 'best_yolov8.pt'
CONF_THRES  = 0.25
IOU_THRES   = 0.45
CLASSES     = ['BCC', 'SCC', 'ACK', 'SEK', 'MEL', 'NEV']
CLASS_INFO  = {
    'BCC': ('Basal Cell Carcinoma', 'cancer'),
    'SCC': ('Squamous Cell Carcinoma', 'cancer'),
    'ACK': ('Actinic Keratosis', 'pre-cancer'),
    'SEK': ('Seborrheic Keratosis', 'benign'),
    'MEL': ('Melanoma', 'cancer'),
    'NEV': ('Nevus / mole', 'benign'),
}
KIND_BGR = {
    'cancer':     (96,  84, 255),    # red-ish
    'pre-cancer': (71, 181, 255),    # orange
    'benign':     (167, 212, 45),    # green
}


def load_model():
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {CKPT_PATH}\n"
            f"Download 'best_yolov8.pt' from your Google Drive "
            f"'pad-ufes-20-results' folder and place it next to detect_app.py."
        )
    return YOLO(str(CKPT_PATH))


model = load_model()
app = Flask(__name__)


def draw_detections(bgr, dets):
    out = bgr.copy()
    h, w = out.shape[:2]
    thick = max(2, int(round(min(h, w) / 250)))
    font_scale = max(0.5, min(h, w) / 900)
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        col = KIND_BGR[d['kind']]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        label = f"{d['code']} {d['prob']:.1f}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thick)
        ly1 = max(0, y1 - th - bl - 6)
        cv2.rectangle(out, (x1, ly1), (x1 + tw + 10, y1), col, -1)
        cv2.putText(out, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return out


def encode_jpeg_b64(bgr):
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


@app.route('/')
def index():
    return render_template('detect.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'no file uploaded'}), 400
    f = request.files['file']
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
    except Exception as e:
        return jsonify({'ok': False, 'error': f'bad image: {e}'}), 400
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    res = model.predict(bgr, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            cid = int(b.cls.item())
            code = CLASSES[cid] if 0 <= cid < len(CLASSES) else f'class{cid}'
            name, kind = CLASS_INFO.get(code, (code, 'benign'))
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            dets.append({
                'code': code, 'name': name, 'kind': kind,
                'prob': round(float(b.conf.item()) * 100, 2),
                'bbox': [x1, y1, x2, y2],
            })
    dets.sort(key=lambda d: d['prob'], reverse=True)

    annotated = draw_detections(bgr, dets) if dets else bgr
    return jsonify({
        'ok': True,
        'count': len(dets),
        'detections': dets,
        'image': encode_jpeg_b64(annotated),
    })


if __name__ == '__main__':
    print(f"[info] model : {CKPT_PATH}")
    print("[info] open   : http://127.0.0.1:5002")
    app.run(debug=False, port=5002)
