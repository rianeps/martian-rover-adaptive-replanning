# Adaptive Replanning Under Realistic Sensor Degradation for Autonomous Martian Rover Navigation

**Msc Data Science and AI Dissertation**
**Newcastle University — 2026**

---

## Overview

This project develops an autonomous navigation system for a simulated Martian rover that detects and responds to sensor degradation in real time. Most existing navigation systems assume clean, reliable sensor input throughout operation — an assumption that does not reflect the reality of long-duration Mars missions, where camera sensors degrade progressively due to dust accumulation, physical wear, and signal instability.

The system combines a ResNet18 CNN for terrain classification with an A* path planner, augmented by a sensor quality monitor that continuously evaluates incoming camera feed quality. When image quality drops below a defined threshold, the system triggers adaptive replanning, generating a safer route that accounts for the increased uncertainty in terrain classification.

---

## Project Structure

```
martian-rover-adaptive-replanning/
│
├── notebooks/
│   ├── 01_baseline_navigation.ipynb       # Terrain simulation + A* path planner
│   ├── 02_sensor_degradation.ipynb        # Sensor degradation module + quality metrics
│   └── 03_cnn_terrain_classifier.ipynb    # ResNet18 CNN terrain classifier
│
├── figures/                               # Output figures from experiments
├── requirements.txt                       # Python dependencies
└── README.md
```

---

## Notebooks

### Notebook 01 — Baseline Navigation System
Builds the foundation of the project. Generates a simulated 30x30 Martian terrain grid with Safe, Rocky, and Hazardous zones and implements an A* path planner with terrain cost weighting. Evaluates the baseline system across 20 terrain seeds.

**Key result:** 90% navigation success rate, average 52 steps per path.

### Notebook 02 — Sensor Degradation Module
Implements three realistic sensor degradation types — Gaussian noise, partial occlusion, and intermittent dropout — at configurable severity levels. Measures image quality using SSIM and PSNR metrics and establishes an SSIM threshold of 0.7 as the replanning trigger criterion.

**Key result:** Dropout crosses the quality threshold at mild severity; occlusion degrades gradually and only fails at severe levels.

### Notebook 03 — CNN Terrain Classifier
Fine-tunes a pretrained ResNet18 on real NASA Mars surface imagery using a two-phase transfer learning strategy. Evaluates the model with a confusion matrix and per-class classification report. Tests CNN accuracy and confidence under all three degradation types.

**Key result:** 91.71% validation accuracy. CNN confidence scores drop proportionally with degradation severity, validating their use as a real-time sensor quality proxy.

---

## Datasets

### Primary — Mars Surface Image Dataset (Kaggle)
- **URL:** https://www.kaggle.com/datasets/aumthaker/mars-terrain-images
- **Size:** 6,153 greyscale images at 227x227 pixels
- **Classes:** 8 (bright dune, crater, dark dune, impact ejecta, other, slope streak, spider, swiss cheese)
- **Format:** JPG, organised into one folder per class

### Secondary — AI4Mars (Zenodo)
- **URL:** https://zenodo.org/records/15995036
- **Size:** ~326,000 segmentation labels across 35,000 images
- **Rovers:** Curiosity, Opportunity, Spirit, Perseverance
- **Format:** PNG with segmentation masks

---

## Setup and Installation

### Requirements
All notebooks are designed to run in **Google Colab**. No local installation is required beyond the dependencies listed below.

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
torch
torchvision
numpy
matplotlib
opencv-python
scikit-image
seaborn
kaggle
Pillow
pandas
```

### Kaggle API Setup
To download the primary dataset, a Kaggle API token is required:

1. Go to [kaggle.com](https://kaggle.com) → Profile → Settings → API → **Create New Token**
2. In Colab, run:

```python
from google.colab import files
files.upload()  # Upload your kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Google Drive Setup
Mount Google Drive to persist datasets and model checkpoints across Colab sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set your project data path at the top of each notebook:
```python
DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/Mars Rover Project/mars_terrain'
```

---

## Running the Notebooks

Run notebooks in order — each builds on the previous:

1. Open [Google Colab](https://colab.research.google.com)
2. Go to **File → Upload notebook** and upload the relevant `.ipynb` file
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Run all cells top to bottom with **Shift + Enter**

> ⚠️ Notebook 03 requires a GPU. Enable it before running to avoid slow training.

---

## Results Summary

| Component | Result |
|---|---|
| Baseline navigation success rate | 90% across 20 terrain seeds |
| Average path length | 52 steps |
| CNN validation accuracy | 91.71% |
| SSIM replanning threshold | 0.7 |
| Most destructive degradation type | Dropout (SSIM: 0.37 at mild severity) |
| Most gradual degradation type | Occlusion (SSIM: 0.81 at moderate severity) |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core programming language |
| PyTorch | CNN training and inference |
| ResNet18 | Pretrained backbone for terrain classification |
| OpenCV | Image processing and degradation injection |
| scikit-image | SSIM and PSNR quality metrics |
| Matplotlib | Visualisation |
| Google Colab | Cloud GPU training environment |
| Google Drive | Persistent data storage |
| Kaggle API | Dataset download |

---

## Roadmap

- [x] Notebook 01 — Baseline Navigation System
- [x] Notebook 02 — Sensor Degradation Module
- [x] Notebook 03 — CNN Terrain Classifier
- [ ] Notebook 04 — Sensor Quality Monitor
- [ ] Notebook 05 — Adaptive Replanning System
- [ ] Notebook 06 — Evaluation
- [ ] Unity Simulation Environment

---

## Repository

**GitHub:** https://github.com/rianeps/martian-rover-adaptive-replanning

---

## Licence

This project is released under the [MIT Licence](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this code for research and educational purposes with appropriate attribution.

---

## Author

**rianeps** — BSc Computer Science, Newcastle University, 2026
Supervisor: [Supervisor Name]
