# Fundus_US Uveal Melanoma Diagnosis
## Overview
This repository contains code for training and evaluating a multi-modal deep learning classifier using three aligned imaging inputs:

1. Optos fundus images
2. Longitudinal ultrasound images (LUS)
3. Transverse ultrasound images (TUS)

This methodology is described in: Dadzie AK, Iddir SP, Abtahi M, et al (2025). Attention-Based Multimodal Deep Learning for Uveal Melanoma Classification Using Ultra-Widefield Fundus Images and Ocular Ultrasound.

## Problem Statement
UM is the most common primary intraocular malignancy, but distinguishing it from benign choroidal nevi remains a critical clinical challenge due to overlapping features and variability in clinician interpretation. Current imaging methods such as fundus photography and ultrasound each provide important but incomplete information when used in isolation.

To address this gap, we developed a deep learning framework that integrates ultra-widefield fundus photographs and B-scan ultrasound (both longitudinal and transverse views) for the automated classification of UM versus choroidal nevi. Using an EfficientNetV2-S backbone with a convolutional block attention module (CBAM) for feature-level fusion, the multimodal model effectively combines surface-level information from fundus images with depth-resolved tumor characteristics from ultrasound.


## Deep Learning Strategies
Each imaging modality was first trained independently to evaluate its contribution to classification performance. To integrate complementary information across modalities, we experimented with two fusion strategies: prediction averaging and attention-based feature fusion. In prediction averaging, classification probabilities from the three single-modality models were averaged to produce the final output. In contrast, the attention-based fusion strategy employed convolutional block attention modules (CBAMs) to refine feature maps before concatenation, enabling the model to selectively emphasize the most relevant multimodal information. CBAM consists of two components: the Channel Attention Module (CAM), which highlights diagnostically important feature channels, and the Spatial Attention Module (SAM), which emphasizes critical spatial regions. ogether, CAM and SAM adaptively enhanced meaningful features from each modality, allowing the fused representation to capture both surface-level and depth-resolved tumor characteristics, ultimately achieving superior classification performance compared to single-modality models and prediction averaging. <br/><br/>


<img width="3795" height="2244" alt="Fig 2" src="https://github.com/user-attachments/assets/fa3b252b-f6bd-4481-b3b0-470457a29b8e" />
Deep learning framework. A–C: single-modality classifiers (Optos, LUS, TUS). D: prediction averaging strategy. E: attention-based feature fusion strategy. F: CBAM architecture with CAM and SAM components.<br/><br/>

## Repository structure

```text
├── configs/
│   └── config.yaml                  # Main experiment configuration
├── data/
│   └── README.md                    # Expected data organization
├── scripts/
│   ├── train_cross_validation.py    # Main training/evaluation script
│   └── predict.py                   # Inference script
├── src/
│   └── mm_fusion/
│       ├── data.py                  # CSV loading and image generators
│       ├── metrics.py               # Evaluation and plotting utilities
│       ├── model.py                 # CBAM and fusion model definitions
│       └── utils.py                 # Config, seed, and GPU helpers
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

To make the local `src/` package importable, run scripts from the repository root using:

```bash
export PYTHONPATH=$PWD/src
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="$PWD\src"
```

## Data organization

By default, the code expects:

```text
data/
  FileNames/
    image_classes_OPTOS.csv
    image_classes_LUS.csv
    image_classes_TUS.csv
  images/
    OPTOS_RG/
    LUS/
    TUS/
```

Each CSV should contain at least two columns:

```text
Image Name,Class
example_001.png,Nevus
example_002.png,UM
```

The rows in the Optos, LUS, and TUS CSV files must be aligned because the model combines corresponding images from the three modalities.

## Configuration

Edit `configs/config.yaml` before running the code. The most important fields are:

```yaml
data:
  optos_dir: "data/images/OPTOS_RG"
  lus_dir: "data/images/LUS"
  tus_dir: "data/images/TUS"
  optos_csv: "data/FileNames/image_classes_OPTOS.csv"
  lus_csv: "data/FileNames/image_classes_LUS.csv"
  tus_csv: "data/FileNames/image_classes_TUS.csv"

model:
  base_architecture_path: "models/MM_Model_EV2S.h5"
  optos_weights: "models/Early_EV2S_best_weights{fold}.h5"
  lus_weights: "models/LUS_EV2S_best_weights{fold}.h5"
  tus_weights: "models/TUS_EV2S_best_weights{fold}.h5"
```

The original code used fold-specific pretrained single-modality weights. Those files are not included in this repository, so users must either provide them or train single-modality models to get the weights.

## Training

From the repository root:

```bash
export PYTHONPATH=$PWD/src
python scripts/train_cross_validation.py --config configs/config.yaml
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts/train_cross_validation.py --config configs/config.yaml
```

The training script performs stratified cross-validation and saves:

```text
outputs/cross_validation_metrics.xlsx
outputs/prediction_results_fold_*.xlsx
outputs/confusion_matrix_fold_*.png
outputs/roc_curve_fold_*.png
```

## Inference

After training or after placing a trained model in the expected location:

```bash
export PYTHONPATH=$PWD/src
python scripts/predict.py --config configs/config.yaml --model outputs/CBAM_model.h5
```

## Notes for users

This repository is intended to share the research code in a usable form. It does not include clinical images, protected patient data, or trained weights. These can be obtained upon reasonable request and appropriate documentations.

If you use this code, please cite the related publication.

```text
Dadzie, A. K., Iddir, S. P., Abtahi, M., Ebrahimi, B., Rahimi, M., Ganesh, S., ... & Yao, X. (2025). Attention-Based multimodal deep learning for uveal melanoma classification using ultra-widefield fundus images and ocular ultrasound. Ophthalmology Science, 100985.
```

## Important implementation notes

- The model assumes a binary classification task by default: `Nevus` vs `UM`.
- Image size is set to `307 x 390 x 3`.
- The fusion model loads three pretrained single-modality models and freezes their layers.
- A CBAM attention block is applied to each modality-specific feature map before feature concatenation.
- Validation generators are not augmented beyond rescaling.

## Support
For support, email adadzi2@uic.edu or xcy@uic.edu, or open an issue on the repository's issue tracker.

## Acknowledgments
• National Eye Institute.\
• Research to Prevent Blindness.\
• Richard and Loan Hill Endowment.\
• All contributors who participated in data collection, development and validation of the deep learning models.

## Citation
Dadzie, A. K., Iddir, S. P., Abtahi, M., Ebrahimi, B., Rahimi, M., Ganesh, S., ... & Yao, X. (2025). Attention-Based Multimodal Deep Learning for Uveal Melanoma Classification Using Ultra-Widefield Fundus Images and Ocular Ultrasound. Ophthalmology Science, 100985.

## Contact
Lead Author: Albert Kofi Dadzie - adadzi2@uic.edu\
Project Supervisors: Dr. Xincheng Yao - xcy@uic.edu & Dr. Michael J. Heiferman - mheif@uic.edu\
Institutions: Department of Biomedical Engineering and Department of Ophthalmology & Visual Sciences, University of Illinois Chicago.
