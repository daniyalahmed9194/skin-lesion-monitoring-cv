"""
Mini Annotator for PAD-UFES-20 Skin Lesion Dataset.
Usage: python annotator.py [optional-path-to-images-folder]
       (defaults to DEFAULT_IMAGES_DIR below if no arg given)
Output: annotations.json in current directory.

Controls:
  Mouse drag      : draw a bounding box
  Keys 1-6        : assign class to the last-drawn box
                    1=BCC  2=SCC  3=ACK  4=SEK  5=MEL  6=NEV
  n / p           : next / previous image
  d               : delete last box on current image
  s               : save annotations to disk
  q               : quit (auto-saves)
"""

import cv2
import json
import os
import sys
from pathlib import Path

CLASSES = {ord('1'): 'BCC', ord('2'): 'SCC', ord('3'): 'ACK',
           ord('4'): 'SEK', ord('5'): 'MEL', ord('6'): 'NEV'}
COLORS = {'BCC': (0, 0, 255), 'SCC': (0, 128, 255), 'ACK': (0, 255, 255),
          'SEK': (0, 255, 0), 'MEL': (255, 0, 255), 'NEV': (255, 128, 0),
          None: (200, 200, 200)}
OUT_FILE = 'annotations.json'
MAX_W, MAX_H = 1000, 700
DEFAULT_IMAGES_DIR = r'B:\Semester6\Computer Vision\Project\images'

state = {'drawing': False, 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'scale': 1.0}


def load_annotations():
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            return json.load(f)
    return {}


def save_annotations(ann):
    with open(OUT_FILE, 'w') as f:
        json.dump(ann, f, indent=2)
    print(f"[saved] {OUT_FILE} ({sum(len(v) for v in ann.values())} boxes across {len(ann)} images)")


def fit_image(img):
    h, w = img.shape[:2]
    s = min(MAX_W / w, MAX_H / h, 1.0)
    state['scale'] = s
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def to_orig(px, py):
    s = state['scale']
    return int(px / s), int(py / s)


def draw_overlay(disp, boxes, idx, total, fname):
    s = state['scale']
    for b in boxes:
        x, y, w, h = b['x'], b['y'], b['w'], b['h']
        x, y, w, h = int(x * s), int(y * s), int(w * s), int(h * s)
        color = COLORS.get(b.get('class'))
        cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
        label = b.get('class') or '?'
        cv2.putText(disp, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if state['drawing']:
        cv2.rectangle(disp, (state['x0'], state['y0']),
                      (state['x1'], state['y1']), (255, 255, 255), 1)
    bar = f"[{idx + 1}/{total}] {fname}  boxes={len(boxes)}  " \
          f"keys: 1=BCC 2=SCC 3=ACK 4=SEK 5=MEL 6=NEV  n/p d s q"
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(disp, bar, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)


def on_mouse(event, x, y, flags, param):
    boxes = param['boxes']
    if event == cv2.EVENT_LBUTTONDOWN:
        state['drawing'] = True
        state['x0'], state['y0'], state['x1'], state['y1'] = x, y, x, y
    elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
        state['x1'], state['y1'] = x, y
    elif event == cv2.EVENT_LBUTTONUP and state['drawing']:
        state['drawing'] = False
        x0, y0 = to_orig(min(state['x0'], x), min(state['y0'], y))
        x1, y1 = to_orig(max(state['x0'], x), max(state['y0'], y))
        if x1 - x0 > 4 and y1 - y0 > 4:
            boxes.append({'x': x0, 'y': y0, 'w': x1 - x0,
                          'h': y1 - y0, 'class': None})


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(DEFAULT_IMAGES_DIR)
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    files = sorted([p.name for p in folder.iterdir()
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not files:
        print(f"No images in {folder}")
        sys.exit(1)

    ann = load_annotations()
    idx = 0
    cv2.namedWindow('annotator')

    while True:
        fname = files[idx]
        boxes = ann.setdefault(fname, [])
        img = cv2.imread(str(folder / fname))
        if img is None:
            print(f"skip unreadable: {fname}")
            idx = (idx + 1) % len(files)
            continue
        disp = fit_image(img.copy())
        cv2.setMouseCallback('annotator', on_mouse, {'boxes': boxes})

        while True:
            view = disp.copy()
            draw_overlay(view, boxes, idx, len(files), fname)
            cv2.imshow('annotator', view)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in CLASSES:
                if boxes:
                    boxes[-1]['class'] = CLASSES[key]
            elif key == ord('n'):
                idx = (idx + 1) % len(files); break
            elif key == ord('p'):
                idx = (idx - 1) % len(files); break
            elif key == ord('d'):
                if boxes: boxes.pop()
            elif key == ord('s'):
                save_annotations(ann)
            elif key == ord('q'):
                save_annotations(ann)
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    main()
