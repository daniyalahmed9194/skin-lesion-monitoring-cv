# Patient Skin Lesion Monitoring

Vision-based skin lesion **classification, detection & segmentation** using the
PAD-UFES-20 dataset.
Computer Vision — Semester 6 — Project 1 (Patient Monitoring System).

---

## Overview

A patient-monitoring system that screens skin lesions from clinical images and
flags potentially malignant ones. Three CV tasks are built on a single dataset:

| Task | Output |
|---|---|
| Classification | 6-class lesion diagnosis (BCC, SCC, ACK, SEK, MEL, NEV) |
| Detection | Bounding box around the lesion |
| Segmentation | Pixel mask of the lesion region |

Full rationale and dataset justification: see [`dataset_proposal.md`](dataset_proposal.md).

---

## Dataset — PAD-UFES-20

- **Source:** *Data in Brief*, Vol. 32 (2020), Article 106221 — DOI `10.1016/j.dib.2020.106221`
- **Paper:** https://www.sciencedirect.com/science/article/pii/S235234092031115X
- **Download:** https://data.mendeley.com/datasets/zr7vgbcyr2/1
- **Size:** 2,298 smartphone clinical images, 1,373 patients, 6 classes

The dataset is **not committed to git** (see `.gitignore`). Download it,
then extract all images into an `images/` folder in the project root:

```
Project/
└── images/
    ├── PAT_8_15_820.png
    ├── ...
```

---

## Setup

Requires Python 3.x.

```bash
pip install flask opencv-python
```

---

## Usage

### Web annotator (recommended)

```bash
python web_annotator.py
```

Open <http://127.0.0.1:5000> in your browser.

- Drag on the image to draw a bounding box
- Click a class (or press keys `1`–`6`) to set the active class
- **Next / Prev** to navigate, **Save** to write `annotations.json`

### CLI annotator (OpenCV)

```bash
python annotator.py
```

Controls: mouse-drag = box · keys `1`–`6` = class · `n`/`p` = next/prev ·
`d` = delete last · `s` = save · `q` = quit (auto-saves).

Both tools write to the same `annotations.json`.

---

## Project Structure

```
Project/
├── web_annotator.py      # Flask web annotation tool
├── annotator.py          # OpenCV CLI annotation tool
├── templates/
│   └── annotator.html    # Web UI
├── dataset_proposal.md   # Week 1 written deliverable
├── annotations.json      # Generated annotations (deliverable)
├── images/               # Dataset (not in git — download separately)
└── README.md
```

---

## Project Timeline

| Week | Due | Task | Status |
|---|---|---|---|
| 1 | 12-05-2026 | Dataset selection, custom annotator, 20 sample annotations | In progress |
| 2 | 19-05-2026 | Annotation + classification | Pending |
| 3 | 26-05-2026 | Object detection | Pending |
| 4 | 02-06-2026 | Segmentation, evaluation, video demo | Pending |
| 5 | 09-06-2026 | Final evaluation | Pending |
