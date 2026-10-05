# Retinal Blood Vessel Segmentation

Classical image-processing pipeline for segmenting blood vessels in retinal fundus images. No deep learning is used. Four vessel-detection algorithms run on the same preprocessed image. Each one outputs a vessel **confidence map**, and a **hybrid fusion** step combines the four maps into the final segmentation.

> Course project for Digital Image Processing, ECE, IIIT Hyderabad (December 2025)
> **Sai Sampath Ravikanti** (2023102033) · **Harshita Kumari** (2023102073)

---

## Table of Contents
- [Motivation](#motivation)
- [Pipeline](#pipeline)
- [Preprocessing](#preprocessing)
- [The Four Algorithms](#the-four-algorithms)
- [Hybrid Fusion](#hybrid-fusion)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations & Future Work](#limitations--future-work)

---

## Motivation

Retinal blood vessels are important biomarkers for **diabetic retinopathy, glaucoma and hypertension**. Doctors use vessel width, branching patterns and leakage regions to diagnose and grade these diseases. Segmenting vessels by hand is slow and needs expert ophthalmologists.

Automated segmentation:
- makes early disease detection and large-scale screening possible,
- helps less experienced clinicians make more accurate assessments,
- is the first step for many downstream tasks, such as optic disc detection, lesion detection and severity grading.

**Goal:** evaluate several established vessel-segmentation algorithms, compare them on the DRIVE dataset, and combine their strengths into a hybrid method.

---

## Pipeline

```
                         ┌──► Matched Filtering ──────────┐
                         │                                │
Fundus ──► Preprocessing ┼──► Multi-scale Line Detection ─┼──► Hybrid Fusion ──► Post-processing ──► Binary
 image     (shared)      │                                │    (confidence       (threshold, closing,  vessel map
                         ├──► Scale-space (Frangi) ───────┤     averaging)        small-object removal)
                         │                                │
                         └──► Morphological Processing ───┘
```

Every algorithm takes the same preprocessed input, which has bright vessels on a flat background. Every algorithm returns a `float32` confidence map in **[0, 1]**, so the four maps can be fused directly.

---

## Preprocessing

`preprocess.py` (see also `preprocess.ipynb`, which plots each step)

| Step | What it does | Why |
|---|---|---|
| **1. Green channel** | Keeps only the G channel of the RGB image | The green channel has the highest contrast between vessels and background |
| **2. CLAHE** | Contrast Limited Adaptive Histogram Equalization (clip limit 2.0, 8×8 tiles) | Corrects uneven illumination and boosts local contrast |
| **3. Morphological opening** | 3×3 structuring element | Removes the central light reflex along vessels and smooths bright ridges |
| **4. Background homogenization** | Subtracts a 69×69 mean-filtered background, then robust-scales with `(x − median) / IQR` | Evens out illumination across the field of view |
| **5. Inversion** | Multiplies the image by −1 | Turns dark vessels into **bright ridges**, which is what all four detectors expect |

---

## The Four Algorithms

Retinal vessels vary in width, curvature, orientation and contrast, and they appear next to bright distractors such as the optic disc and exudates. No single detector handles all of these cases well. Each of the four methods below is strong in a different situation.

### 1. Matched Filtering (`matched.py`)
Vessel cross-sections have a roughly Gaussian intensity profile, low curvature and widths of about 2–10 px.
- Builds a bank of zero-mean, unit-norm **Gaussian kernels** at **7 scales** (σ = 0.5 → 3.0, lengths 5 → 17) and **12 orientations** (15° apart).
- At each scale, keeps the maximum response over all orientations, applies gamma 0.5 to boost weak vessels, and weights the result by σ.
- Applies a **sigmoid enhancement** centred at 0.4, then a morphological close and open to clean up noise.
- **Strengths:** medium and large vessels, simple and fast. **Weaknesses:** false positives near the optic disc and misses faint, thin vessels.

### 2. Multi-scale Line Detection (`line_detection.py`)
Treats vessels as **line segments** at several lengths.
- Builds rotated, normalised line kernels with lengths **3, 5, 7, 9, 11, 15 px** at **12 orientations**.
- At each scale, keeps the strongest response over all orientations. The final map is the average across scales, normalised to [0, 1].
- **Strengths:** highest sensitivity, catches both thin and thick vessels. **Weaknesses:** more false positives and sensitive to background noise.

### 3. Scale-space Analysis (`scale_space.py`)
- Computes **Hessian eigenvalues** at 16 Gaussian scales (σ = 0.5 → 8.0) and from them the **Frangi vesselness** (α = 0.5, β = 15). Only bright ridges (λ₂ < 0) are kept.
- Takes the σ-weighted maximum across scales.
- Fuses the result with a **multi-scale top-hat** response (disk radii 1–8): `0.7 · Frangi + 0.3 · Top-hat`.
- **Strengths:** good precision, strong on medium-width vessels. **Weaknesses:** computationally expensive and weaker on the thinnest capillaries.

### 4. Morphological Processing (`morphological.py`)
- Applies a **closing** with a radius-2 disk to fill gaps inside vessels.
- Applies a **modified top-hat** with disk radii **1–8 px**: `img − min(open(closed, r), img)` at each radius.
- Averages adjacent scales in pairs and combines them with a **weighted maximum**, giving more weight to finer scales.
- **Strengths:** very high precision and robust around the optic disc and bright lesions. **Weaknesses:** low sensitivity, misses thin vessels, and depends heavily on the structuring-element size.

---

## Hybrid Fusion

Implemented in `hybrid.ipynb`.

1. Run the four algorithms on each preprocessed batch.
2. **Average the confidence maps** with equal weights:
   `hybrid = 0.25·matched + 0.25·scale_space + 0.25·line + 0.25·morph`
3. Apply a **field-of-view (FOV) mask**. It is the largest contour of the Otsu-thresholded green channel, which removes the black border around the retina.
4. **Post-processing:**
   - threshold the fused map,
   - apply a 2×2 morphological closing,
   - run connected-component analysis and remove components smaller than **100 px**.
5. Evaluate only inside the FOV against the ground truth, and plot an **error map** (green = TP, red = FP, blue = FN).

The averaging step combines the high sensitivity of line detection with the high precision of the morphological and scale-space methods. Their errors partly cancel out.

---

## Results

### Values reported in the presentation

| Method | Accuracy | Sensitivity | Precision | F1-Score |
|---|---:|---:|---:|---:|
| Matched Filtering | 86.88 % | 88.55 % | 44.44 % | 57.55 % |
| Morphological Processing | 92.19 % | 40.86 % | **95.48 %** | 56.98 % |
| Multi-scale Line Detection | 93.21 % | **87.69 %** | 66.59 % | 75.68 % |
| Scale-space Analysis | **94.67 %** | 71.49 % | 83.18 % | **76.88 %** |
| **Hybrid (ours)** | 94.30 % | 76.32 % | 83.18 % | 76.70 % |

The hybrid is the most **balanced** method. Its accuracy and F1 are close to the best single method, and it does not have the large drop in either sensitivity or precision that each individual detector shows.

<details>
<summary>Mean of the top-20 images in each notebook run (80 training images)</summary>

| Method | Accuracy | Sensitivity | Precision | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| Matched Filtering | 86.60 | 88.55 | 44.44 | 86.44 | 57.55 |
| Line Detection | 86.44 | 88.08 | 43.94 | 86.26 | 57.25 |
| Scale-space | 92.34 | 68.11 | 57.84 | 94.82 | 62.07 |
| Morphological | 94.33 | 54.93 | 82.91 | 98.32 | 63.29 |
| Hybrid (threshold 0.35) | 93.77 | 51.39 | 83.58 | 98.62 | 62.76 |

These numbers come from the saved notebook outputs. They depend on the threshold used in each run, which is why they differ from the slide figures.
</details>

---

## Project Structure

```
Retinal-Blood-Vessel-Segmentation/
├── preprocess.py          # Shared preprocessing pipeline
├── matched.py             # Gaussian matched filtering
├── line_detection.py      # Multi-scale line detector
├── scale_space.py         # Frangi vesselness + multi-scale top-hat
├── morphological.py       # Multi-scale morphological top-hat
├── morp.py                # Identical copy of morphological.py
├── preprocess.ipynb       # Step-by-step preprocessing visualisation
├── matched_filter.ipynb   # Matched filter experiments + metrics
├── line_detection.ipynb   # Line detection experiments + metrics
├── scale.ipynb            # Scale-space experiments + metrics
├── morph_algo.ipynb       # Morphological experiments + metrics
├── hybrid.ipynb           # Hybrid fusion, evaluation, error maps
└── dipppt.pdf             # Project presentation
```

---

## Getting Started

### Requirements
- Python 3.8+
- `opencv-python`, `numpy`, `scipy`, `matplotlib`, `jupyter`

```bash
pip install opencv-python numpy scipy matplotlib jupyter
```

### Dataset
The notebooks expect this layout:
```
dataset/
└── train/
    ├── Original/        # fundus images (.png / .jpg / .tif)
    └── Ground truth/    # matching binary vessel masks
```
Images and masks are paired by sorted filename. The project was evaluated on the **[DRIVE](https://drive.grand-challenge.org/)** retinal dataset. The dataset is not included in this repository.

### Run
```bash
jupyter notebook hybrid.ipynb
```
Run all cells. The notebook processes the images in batches, prints a per-image metrics table and plots the best segmentation with its error map. You can tune `SENSITIVITY_THRESHOLD` and `BATCH_SIZE` in the main cell.

To use the modules from your own code:
```python
import cv2
from preprocess import preprocess
import matched, line_detection, scale_space, morphological

imgs = [cv2.imread("fundus.png")]
pre  = preprocess(imgs)
maps = [m(pre)[0] for m in (matched.matched_filter, line_detection.line_detection,
                            scale_space.scale_space, morphological.morphological)]
hybrid = sum(maps) / 4.0          # confidence map in [0, 1]
```

---

## Evaluation Metrics

All metrics are computed per pixel **inside the FOV mask**:

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Sensitivity (Recall) | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| Precision | TP / (TP + FP) |
| F1-Score | 2 · Precision · Sensitivity / (Precision + Sensitivity) |

Most pixels in a fundus image are background, so a method can score high accuracy while still missing many vessels. **F1 and sensitivity** are the more informative metrics for this task.

---

## Limitations & Future Work
- **Fusion uses fixed equal weights.** The presentation proposes using the morphological output in bright regions (optic disc and exudates). That switch is not yet in the notebook. Learned or region-adaptive weights could improve results.
- **One global threshold** is used. Per-image thresholds (Otsu or hysteresis) could recover more thin vessels.
- **Speed:** the Frangi filter at 16 scales and the 84-kernel matched filter bank are slow on high-resolution images.
- **Comparison with deep learning:** benchmarking against a U-Net baseline on DRIVE, STARE and CHASE_DB1 would show how far the classical pipeline lags behind.
