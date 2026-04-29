<div align="center">

<h1>CamoFocus</h1>

<h3>Enhancing Camouflage Object Detection with Split-Feature Focal Modulation and Context Refinement</h3>

<p>
  <a href="https://doi.org/10.1109/WACV57701.2024.00146">
    <img src="https://img.shields.io/badge/WACV_2024-Paper-blue?style=flat-square" />
  </a>
  <a href="https://ieeexplore.ieee.org/document/10483928">
    <img src="https://img.shields.io/badge/IEEE-Xplore-red?style=flat-square" />
  </a>
  <img src="https://img.shields.io/badge/PyTorch-1.9+-orange?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.7+-green?style=flat-square&logo=python" />
</p>

<p>
  <strong>Abbas Khan<sup>1</sup>, Mustaqeem Khan<sup>1</sup>, Wail Gueaieb<sup>1,2</sup>, Abdulmotaleb El Saddik<sup>1,2</sup>, Giulia De Masi<sup>3</sup>, Fakhri Karray<sup>1,4</sup></strong>
</p>

<p>
  <sup>1</sup>MBZUAI, UAE &nbsp;|&nbsp;
  <sup>2</sup>University of Ottawa, Canada &nbsp;|&nbsp;
  <sup>3</sup>Technology Innovation Institute, UAE
</p>

</div>

---

## Overview

Camouflage Object Detection (COD) involves isolating a target object that blends seamlessly into its background — a task that remains a formidable challenge for learning algorithms due to low contrast, similar textures, and obscure object boundaries.

**CamoFocus** tackles this by drawing inspiration from Focal Modulation Networks. Rather than treating the scene as a uniform feature map, CamoFocus explicitly decouples foreground and background components and modulates them independently, forcing the model to learn the distinctive characteristics of each.

<div align="center">
  <img src="Examples/framework.jpg" width="85%" alt="CamoFocus Architecture"/>
  <p><em>Overall architecture of CamoFocus. The network uses a PVT-v2-B4 backbone, with the FSM and CRM modules driving feature refinement.</em></p>
</div>

---

## Key Contributions

**1. Feature Split and Modulation (FSM)**
The FSM module separates the input feature map into foreground and background streams using a supervisory mask from the Edge Attention Module (EAM). Each stream is passed through independent focal modulation layers that capture context at multiple receptive field scales, allowing the model to model the distinct statistics of camouflaged objects versus their surroundings. The two streams are recombined with learnable blend weights.

**2. Context Refinement Module (CRM)**
The CRM serves as the top-down decoder. It aggregates the multi-scale features produced by FSM across all four pyramid levels using grouped dilated convolutions at progressively larger dilation rates (4, 6, 8). This enriches spatial context comprehensively and yields highly accurate, boundary-aware prediction maps.

---

## Architecture

```
Input Image (416 × 416)
       │
  PVT-v2-B4 Backbone
  ┌────┴────────────────────┐
  x1 (64ch)  x2 (128ch)  x3 (320ch)  x4 (512ch)
       │
  Channel Reduction (Conv1x1)
  ┌────┴────────────────────┐
  x1r (64)  x2r (128)  x3r (256)  x4r (256)
                │
           EAM (x3r, x2r)
                │
            mask (foreground prob.)
                │
    ┌───────────┴───────────┐
    FSM — focal modulation guided by mask
    ┌────┴────────────────────┐
    x1r′  x2r′  x3r′  x4r′
                │
    CRM — top-down decoder
    x34  ←  CRM(x3r′, x4r′)
    x234 ←  CRM(x2r′, x34)
    x1234 ← CRM(x1r′, x234)
                │
    ┌───────────┴────────────┐
    o3 (×16)  o2 (×8)  o1 (×4)  mask (×8)
```

---

## Qualitative Results

<div align="center">
  <img src="Examples/qualitative.png" width="90%" alt="Qualitative Results"/>
  <p><em>Qualitative comparison of CamoFocus against recent SOTA methods on CAMO, COD10K, CHAMELEON, and NC4K. CamoFocus produces sharper, more complete predictions especially near object boundaries.</em></p>
</div>

---

## Quantitative Results

CamoFocus is evaluated on four standard COD benchmarks: **CHAMELEON**, **CAMO**, **COD10K**, and **NC4K**, using four metrics: structure measure (S<sub>m</sub>), weighted F-measure (F<sup>w</sup><sub>β</sub>), mean absolute error (MAE ↓), and adaptive E-measure (E<sub>m</sub>).

### CHAMELEON

| Method | S<sub>m</sub>↑ | F<sup>w</sup><sub>β</sub>↑ | MAE↓ | E<sub>m</sub>↑ |
|--------|:-:|:-:|:-:|:-:|
| SINet-v2 | 0.882 | 0.828 | 0.030 | 0.942 |
| SegMaR | 0.893 | 0.858 | 0.025 | 0.952 |
| ZoomNet | 0.902 | 0.863 | 0.023 | 0.960 |
| **CamoFocus** | **0.910** | **0.873** | **0.021** | **0.964** |

### CAMO

| Method | S<sub>m</sub>↑ | F<sup>w</sup><sub>β</sub>↑ | MAE↓ | E<sub>m</sub>↑ |
|--------|:-:|:-:|:-:|:-:|
| SINet-v2 | 0.820 | 0.743 | 0.070 | 0.882 |
| SegMaR | 0.842 | 0.781 | 0.071 | 0.897 |
| ZoomNet | 0.838 | 0.778 | 0.066 | 0.891 |
| **CamoFocus** | **0.851** | **0.796** | **0.063** | **0.904** |

### COD10K

| Method | S<sub>m</sub>↑ | F<sup>w</sup><sub>β</sub>↑ | MAE↓ | E<sub>m</sub>↑ |
|--------|:-:|:-:|:-:|:-:|
| SINet-v2 | 0.815 | 0.680 | 0.037 | 0.887 |
| SegMaR | 0.833 | 0.724 | 0.035 | 0.903 |
| ZoomNet | 0.838 | 0.729 | 0.029 | 0.911 |
| **CamoFocus** | **0.845** | **0.742** | **0.028** | **0.918** |

### NC4K

| Method | S<sub>m</sub>↑ | F<sup>w</sup><sub>β</sub>↑ | MAE↓ | E<sub>m</sub>↑ |
|--------|:-:|:-:|:-:|:-:|
| SINet-v2 | 0.847 | 0.770 | 0.048 | 0.903 |
| SegMaR | 0.865 | 0.800 | 0.046 | 0.916 |
| ZoomNet | 0.853 | 0.784 | 0.043 | 0.906 |
| **CamoFocus** | **0.869** | **0.807** | **0.041** | **0.921** |

> **Note:** Please refer to Table 1 in the paper for the full comparison against all 18+ SOTA methods.

---

## Repository Structure

```
CamoFocus/
├── net/
│   ├── network.py          # Main model: Network (FSM + CRM + EAM)
│   ├── pvtv2_encoder.py    # PVT-v2-B4 backbone
│   ├── ResNet.py           # ResNet backbone (alternative)
│   └── Res2Net.py          # Res2Net backbone (alternative)
├── utils/
│   ├── tdataloader.py      # Training data loader
│   └── utils.py            # Helpers: clip_gradient, AvgMeter, poly_lr
├── train.py                # Training script
├── test.py                 # Inference / evaluation script
├── Examples/               # Sample qualitative results
├── models/                 # Pretrained backbone weights (download separately)
├── checkpoints/            # Saved model checkpoints
├── log/                    # Training logs
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/CamoFocus.git
cd CamoFocus

# 2. Create and activate a virtual environment
conda create -n camofocus python=3.8 -y
conda activate camofocus

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt** should include:
```
torch>=1.9.0
torchvision>=0.10.0
timm
numpy
Pillow
```

---

## Datasets

Download the benchmark datasets and place them under `./data/`:

| Dataset | Images | Split | Link |
|---------|--------|-------|------|
| CHAMELEON | 76 | test only | [link](http://www.polsl.pl/rau6/chameleon-database-animal-camouflage-analysis/) |
| CAMO | 1,250 | 1,000 train / 250 test | [link](https://drive.google.com/drive/folders/1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6) |
| COD10K | 5,066 | 3,040 train / 2,026 test | [link](https://dengpingfan.github.io/pages/COD.html) |
| NC4K | 4,121 | test only | [link](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment) |

Training uses **4,040 images** combined from CAMO (1,000) and COD10K (3,040).

Expected directory structure:
```
data/
├── TrainDataset/
│   ├── Imgs/
│   ├── GT/
│   └── Edge/
└── TestDataset/
    ├── CHAMELEON/
    ├── CAMO/
    ├── COD10K/
    └── NC4K/
```

---

## Pretrained Weights

Download the PVT-v2-B4 ImageNet pretrained backbone and place it at `./models/pvt_v2_b4.pth`:

| File | Description | Link |
|------|-------------|------|
| `pvt_v2_b4.pth` | PVT-v2-B4 ImageNet weights | [Google Drive](#) |
| `CamoFocus.pth` | CamoFocus trained checkpoint | [Google Drive](#) |

---

## Training

```bash
python train.py \
  --train_path ./data/TrainDataset \
  --epoch 90 \
  --batchsize 24 \
  --trainsize 416 \
  --lr 1e-4 \
  --train_save CamoFocus
```

Checkpoints are saved every 30 epochs to `checkpoints/CamoFocus/`. Training logs are written to `log/BGNet.txt`.

---

## Inference

```bash
python test.py \
  --checkpoint ./checkpoints/CamoFocus/CamoFocus.pth \
  --test_path ./data/TestDataset \
  --save_path ./results/
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{khan2024camofocus,
  title     = {CamoFocus: Enhancing Camouflage Object Detection with Split-Feature
               Focal Modulation and Context Refinement},
  author    = {Khan, Abbas and Khan, Mustaqeem and Gueaieb, Wail and
               El Saddik, Abdulmotaleb and De Masi, Giulia and Karray, Fakhri},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications
               of Computer Vision (WACV)},
  pages     = {1434--1443},
  year      = {2024}
}
```

---

## Acknowledgements

This work was carried out at **MBZUAI (Mohamed Bin Zayed University of Artificial Intelligence)**, UAE, in collaboration with the University of Ottawa and the Technology Innovation Institute.

We thank the authors of [PVT-v2](https://github.com/whai362/PVT), [Focal Modulation Networks](https://github.com/microsoft/FocalNet), and the COD benchmark datasets ([COD10K](https://dengpingfan.github.io/pages/COD.html), [CAMO](https://drive.google.com/drive/folders/1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6), [NC4K](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment)) for making their code and data publicly available.

---

<div align="center">
  <sub>WACV 2024 · IEEE/CVF Winter Conference on Applications of Computer Vision · pp. 1434–1443</sub>
</div>
