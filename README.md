# Fundus_US-Uveal-Melanoma-Diagnosis
## Overview
This repository provides the models and pretrained weights for the automated classification of uveal melanoma (UM) and choroidal nevi using ultra-widefield fundus photography and B-scan ocular ultrasound (longitudinal and transverse views), as described in: Dadzie AK, Iddir SP, Abtahi M, et al (2025). Attention-Based Multimodal Deep Learning for Uveal Melanoma Classification Using Ultra-Widefield Fundus Images and Ocular Ultrasound.

Uveal melanoma (UM) is the most common primary intraocular malignancy, but distinguishing it from benign choroidal nevi remains a critical clinical challenge due to overlapping features and variability in clinician interpretation. Current imaging methods such as fundus photography and ultrasound each provide important but incomplete information when used in isolation.

To address this gap, we developed a deep learning framework that integrates ultra-widefield fundus photographs and B-scan ultrasound (both longitudinal and transverse views) for the automated classification of UM versus choroidal nevi. Using an EfficientNetV2-S backbone with convolutional block attention modules (CBAMs) for feature-level fusion, the multimodal model effectively combines surface-level information from fundus images with depth-resolved tumor characteristics from ultrasound.


## Models Included
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
