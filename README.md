# Patient Skin Lesion Monitoring

Vision-based skin lesion **classification, detection & segmentation** using the
PAD-UFES-20 dataset.
Computer Vision — Semester 6 — Project 1 (Patient Monitoring System).

---

## Overview

A patient-monitoring system that screens skin lesions from clinical images and
flags potentially malignant ones. Three CV tasks are built on a single dataset:

| Task | Model | Output | App |
|---|---|---|---|
| Classification | MobileNetV2 (transfer) | 6-class lesion diagnosis (BCC, SCC, ACK, SEK, MEL, NEV) | `predict_app.py` |
| Detection | YOLOv8-nano | Bounding boxes + class per lesion | `detect_app.py` |
| Segmentation | YOLOv8n-seg | Per-instance lesion masks + class | `segment_app.py` |

Full rationale and dataset justification: see [`dataset_proposal.md`](dataset_proposal.md).

---

## Dataset — PAD-UFES-20

- **Source:** *Data in Brief*, Vol. 32 (2020), Article 106221 — DOI `10.1016/j.dib.2020.106221`
- **Paper:** https://www.sciencedirect.com/science/article/pii/S235234092031115X
- **Download:** https://data.mendeley.com/datasets/zr7vgbcyr2/1
- **Size:** 2,298 smartphone clinical images, 1,373 patients, 6 classes

The dataset is **not committed to git** (see `.gitignore`). Download it,
then extract all images into an `images/` folder in the project root.

PAD-UFES-20 ships with **classification labels only**. For detection and
segmentation we generate weak supervision automatically from each clinical
image (LAB color → Otsu → largest centred blob) and train on those
pseudo-labels — see the detection / segmentation Colab notebooks for details.

---

## Setup

Requires Python 3.10+.

```bash
# Annotator + dataset utilities
pip install flask opencv-python pillow

# Classification web app
pip install torch torchvision

# Detection web app
pip install ultralytics

# Segmentation web app  (YOLOv8-seg uses the same ultralytics package as detection)
pip install ultralytics
```

---

## Training (Google Colab, T4 GPU)

All three models are trained in Colab notebooks. Put `images.zip` and
`metadata.csv` in your Drive folder `skin-lesion-monitoring-cv_dataset`,
then run each notebook top-to-bottom. Each one writes its best weights to
`MyDrive/pad-ufes-20-results/`.

| Notebook | Model file produced | Metric reported |
|---|---|---|
| `classification_colab.ipynb` | `best_mobilenet_v2.pth` | macro-F1, confusion matrix |
| `detection_colab.ipynb` | `best_yolov8.pt` | mAP@50, mAP@50-95 |
| `segmentation_colab.ipynb` | `best_yolov8_seg.pt` | mask mAP@50, IoU, Dice |

After training, **download the `.pth` / `.pt` files from Drive into the
project root** so the local web apps can load them.

---

## Local Web Apps

Each web app is a small Flask server with a polished DermaScan UI:
drag-and-drop an image, click the action button, see the model's output.

```bash
# Annotation tool                 (Week 1)
python web_annotator.py           # http://127.0.0.1:5000

# Classification                  (Week 2)
python predict_app.py             # http://127.0.0.1:5001

# Detection                       (Week 3)
python detect_app.py              # http://127.0.0.1:5002

# Segmentation                    (Week 4)
python segment_app.py             # http://127.0.0.1:5003
```

The classification / detection / segmentation pages cross-link in the
header so you can move between the three demos for the video walkthrough.

### CLI annotator (alternative)

```bash
python annotator.py
```

Controls: mouse-drag = box · keys `1`–`6` = class · `n`/`p` = next/prev ·
`d` = delete last · `s` = save · `q` = quit (auto-saves).

Both annotation tools write to the same `annotations.json`.

---

## Project Structure

```
Project/
├── web_annotator.py            # Flask annotation tool       (Week 1)
├── annotator.py                # OpenCV CLI annotator        (Week 1)
├── classification_colab.ipynb  # MobileNetV2 training        (Week 2)
├── predict_app.py              # Classification web app      (Week 2)
├── detection_colab.ipynb       # YOLOv8 training             (Week 3)
├── detect_app.py               # Detection web app           (Week 3)
├── segmentation_colab.ipynb    # U-Net training              (Week 4)
├── segment_app.py              # Segmentation web app        (Week 4)
├── templates/
│   ├── annotator.html          # Annotation UI
│   ├── predict.html            # Classification UI
│   ├── detect.html             # Detection UI
│   └── segment.html            # Segmentation UI
├── best_mobilenet_v2.pth       # Classification weights      (download from Drive)
├── best_yolov8.pt              # Detection weights           (download from Drive)
├── best_yolov8_seg.pt          # Segmentation weights        (download from Drive)
├── dataset_proposal.md         # Week 1 written deliverable
├── annotations.json            # Sample annotations           (deliverable)
├── images/                     # Dataset (not in git — download separately)
└── README.md
```

---

## Project Timeline

| Week | Due | Task | Status |
|---|---|---|---|
| 1 | 12-05-2026 | Dataset selection, custom annotator, 20 sample annotations | Done |
| 2 | 19-05-2026 | Annotation + classification (MobileNetV2) | Done |
| 3 | 26-05-2026 | Object detection (YOLOv8, mAP results) | Done |
| 4 | 02-06-2026 | Segmentation (U-Net), evaluation, paper, video demo | Code done, paper + video pending |
| 5 | 09-06-2026 | Final evaluation | Pending |
