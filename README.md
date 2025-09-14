# Fundus_US Uveal Melanoma Diagnosis
## Overview
This repository provides the models and pretrained weights for the automated classification of uveal melanoma (UM) and choroidal nevi using ultra-widefield fundus photography and B-scan ocular ultrasound (longitudinal and transverse views), as described in: Dadzie AK, Iddir SP, Abtahi M, et al (2025). Attention-Based Multimodal Deep Learning for Uveal Melanoma Classification Using Ultra-Widefield Fundus Images and Ocular Ultrasound.

UM is the most common primary intraocular malignancy, but distinguishing it from benign choroidal nevi remains a critical clinical challenge due to overlapping features and variability in clinician interpretation. Current imaging methods such as fundus photography and ultrasound each provide important but incomplete information when used in isolation.

To address this gap, we developed a deep learning framework that integrates ultra-widefield fundus photographs and B-scan ultrasound (both longitudinal and transverse views) for the automated classification of UM versus choroidal nevi. Using an EfficientNetV2-S backbone with a convolutional block attention module (CBAM) for feature-level fusion, the multimodal model effectively combines surface-level information from fundus images with depth-resolved tumor characteristics from ultrasound.


## Deep Learning Strategies
Each imaging modality was first trained independently to evaluate its contribution to classification performance. To integrate complementary information across modalities, we experimented with two fusion strategies: prediction averaging and attention-based feature fusion. In prediction averaging, classification probabilities from the three single-modality models were averaged to produce the final output. In contrast, the attention-based fusion strategy employed convolutional block attention modules (CBAMs) to refine feature maps before concatenation, enabling the model to selectively emphasize the most relevant multimodal information. CBAM consists of two components: the Channel Attention Module (CAM), which highlights diagnostically important feature channels, and the Spatial Attention Module (SAM), which emphasizes critical spatial regions. ogether, CAM and SAM adaptively enhanced meaningful features from each modality, allowing the fused representation to capture both surface-level and depth-resolved tumor characteristics, ultimately achieving superior classification performance compared to single-modality models and prediction averaging. <br/><br/>


<img width="3795" height="2244" alt="Fig 2" src="https://github.com/user-attachments/assets/fa3b252b-f6bd-4481-b3b0-470457a29b8e" />
Deep learning framework. A–C: single-modality classifiers (Optos, LUS, TUS). D: prediction averaging strategy. E: attention-based feature fusion strategy. F: convolutional block attention module (CBAM) architecture with channel attention model (CAM) and spatial attention module (SAM) components.<br/><br/>



## Models included
1. Ultra-widefield fundus images (Optos)
2. Longitudinal ultrasound (LUS)
3. Transverse ultrasound (TUS)
4. Attention-based fusion.

## Support
For support, email adadzi2@uic.edu or xcy@uic.edu, or open an issue on the repository's issue tracker.

## Acknowledgments
• National Eye Institute.\
• Research to Prevent Blindness.\
• Richard and Loan Hill Endowment.\
• All contributors who participated in data collection, development and validation of the deep learning models.

## Citation
Dadzie AK, Iddir SP, Abtahi M, et al. (2025). Attention-Based Multimodal Deep Learning for Uveal Melanoma Classification Using Ultra-Widefield Fundus Images and Ocular Ultrasound

## Contact
Lead Author: Albert Kofi Dadzie - adadzi2@uic.edu\
Project Supervisor: Dr. Xincheng Yao - xcy@uic.edu\ & Dr. Michael J. Heiferman - mheif@uic.edu\
Institutions: Department of Biomedical Engineering and Department of Ophthalmology & Visual Sciences, University of Illinois Chicago.
