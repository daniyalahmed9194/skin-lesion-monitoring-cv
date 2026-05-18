"""
Web Annotator for PAD-UFES-20 Skin Lesion Dataset.
Run: python web_annotator.py
Then open http://127.0.0.1:5000 in your browser.

Requires: pip install flask
"""

import json
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory

IMAGES_DIR = Path(r'B:\Semester6\Computer Vision\Project\images')
OUT_FILE = Path(__file__).parent / 'annotations.json'
CLASSES = ['BCC', 'SCC', 'ACK', 'SEK', 'MEL', 'NEV']

app = Flask(__name__)


def list_images():
    if not IMAGES_DIR.is_dir():
        return []
    return sorted([p.name for p in IMAGES_DIR.iterdir()
                   if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])


def load_ann():
    if OUT_FILE.exists():
        with open(OUT_FILE) as f:
            return json.load(f)
    return {}


def save_ann(ann):
    with open(OUT_FILE, 'w') as f:
        json.dump(ann, f, indent=2)


@app.route('/')
def index():
    return render_template('annotator.html', classes=CLASSES)


@app.route('/api/images')
def api_images():
    return jsonify({'images': list_images(), 'folder': str(IMAGES_DIR)})


@app.route('/api/annotations')
def api_annotations():
    return jsonify(load_ann())


@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.get_json(force=True)
    save_ann(data)
    total = sum(len(v) for v in data.values())
    return jsonify({'ok': True, 'images': len(data), 'boxes': total})


@app.route('/image/<path:name>')
def serve_image(name):
    return send_from_directory(IMAGES_DIR, name)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'files' not in request.files:
        return jsonify({'ok': False, 'error': 'no files'}), 400
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in request.files.getlist('files'):
        name = os.path.basename(f.filename or '')
        if not name:
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg'):
            skipped.append(name)
            continue
        dest = IMAGES_DIR / name
        stem, suf = os.path.splitext(name)
        i = 1
        while dest.exists():
            dest = IMAGES_DIR / f"{stem}_{i}{suf}"
            i += 1
        f.save(dest)
        saved.append(dest.name)
    return jsonify({'ok': True, 'saved': saved, 'skipped': skipped})


if __name__ == '__main__':
    if not IMAGES_DIR.is_dir():
        print(f"[warn] images folder not found: {IMAGES_DIR}")
    print(f"[info] images: {IMAGES_DIR}")
    print(f"[info] output: {OUT_FILE}")
    print("[info] open http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
