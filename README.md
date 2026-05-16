# CMKT-LAM-Monitoring

Cross-Modality Knowledge Transfer Framework for Laser Additive Manufacturing Process Monitoring

---

## Overview

This repository implements a cross-modality knowledge transfer (CMKT) framework for laser additive manufacturing (LAM) process monitoring and defect prediction.

The proposed framework transfers semantic knowledge between:
- melt pool images
- audio spectrograms

through:
- shared convolutional feature extraction
- contrastive semantic alignment
- shared classification learning

The framework learns modality-invariant semantic representations that capture underlying process states across heterogeneous sensing modalities.

During training:
- both melt pool images and audio spectrograms are utilized to facilitate cross-modality semantic learning.

During deployment:
- only melt pool images are required for defect prediction.

This enables:
- enhanced visual representation learning
- improved deployability
- reduced sensing requirements during inference

---

## Framework Architecture

The framework consists of:

1. SharedConvNet  
   A shared convolutional feature extractor used for both:
   - melt pool images
   - audio spectrograms

2. ClassifierNet  
   A shared classifier that predicts process conditions based on learned latent representations.

3. Cross-Modality Semantic Alignment  
   Contrastive semantic alignment is employed to:
   - align semantically similar multimodal representations
   - separate semantically dissimilar representations

The optimization objective combines:
- classification loss \(L_C\)
- contrastive semantic alignment loss \(L_{CSA}\)
- combined semantic alignment objective \(L_{CCSA}\)

---

## Repository Structure

```text
cmkt-lam-monitoring/
│
├── data/
│
├── notebooks/
│   └── cmkt_experiments.ipynb
│
├── outputs/
│
├── src/
│   ├── preprocessing/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── requirements.txt
├── environment.yml
└── README.md