# Zero-Shot Rare Disaster Recognition Using ResNet50 and CLIP

A comparative deep-learning and multimodal learning project for recognizing rare disaster categories using visual feature extraction and vision-language representations.

**Research paper:** *Comparative Analysis of Deep Learning and Multimodal Architectures for Zero-Shot Rare Disaster Recognition*

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/vaishnavipagad/mldsclip4)
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Xplore-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11507386)

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Project Links](#project-links)
- [Dataset and Experimental Split](#dataset-and-experimental-split)
- [Methodology](#methodology)
- [Model Architectures Compared](#model-architectures-compared)
- [Results](#results)
- [Implementation Details](#implementation-details)
- [Running the Notebook](#running-the-notebook)
- [Repository Structure](#repository-structure)
- [Limitations and Reporting Notes](#limitations-and-reporting-notes)
- [Research Publication](#research-publication)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

## Overview

Conventional supervised image classifiers rely on labeled training examples for the classes they are expected to recognize. This creates a challenge when a disaster category is rare, previously unseen, or poorly represented in the training dataset.

This project investigates zero-shot rare disaster recognition by comparing conventional convolutional neural networks (CNNs), a vision-language model, and a hybrid architecture combining ResNet50 visual features with CLIP RN50 embeddings.

The proposed hybrid approach combines learned visual representations with semantic information from text embeddings. The objective is to improve recognition of disaster categories that are not represented in the training classes.

The implementation and experiments are available in the Kaggle notebook, and the research is documented in the linked IEEE Xplore publication.

## Objectives

- Compare conventional deep-learning architectures and vision-language approaches for disaster recognition.
- Evaluate recognition performance on seen and unseen disaster categories.
- Extract complementary representations using ResNet50 and CLIP RN50.
- Combine visual and semantic features into a unified representation.
- Investigate zero-shot classification using text prompts and embedding similarity.
- Compare model performance using classification accuracy and F1-score metrics.

## Project Links

| Resource | Link |
|---|---|
| Executable notebook | [Kaggle — MLDSCLIP4](https://www.kaggle.com/code/vaishnavipagad/mldsclip4) |
| Published research | [IEEE Xplore — Paper](https://ieeexplore.ieee.org/document/11507386) |
| Source repository | [GitHub — Zero-Shot Disaster Recognition](https://github.com/vaishzz-18/zero-shot-disaster-recognition) |

## Dataset and Experimental Split

The supplied notebook reports the following experimental configuration.

### Disaster categories

**Seen classes**

- Cyclone
- Earthquake
- Flood
- Wildfire

These classes are represented in the training and validation sets.

**Unseen classes**

- Drought
- Sandstorm
- Thunderstorm

These classes are evaluated in the test set and are not included in the notebook's training or validation labels.

### Sample distribution reported by the notebook

| Split | Samples | Purpose |
|---|---:|---|
| Training | 1,600 | Learning from the four seen disaster classes |
| Validation | 400 | Evaluating the model on seen classes |
| Test | 266 | Evaluating recognition of unseen disaster classes |
| **Total** | **2,266** | All reported experimental samples |

The notebook's test set contains:

| Unseen category | Test samples |
|---|---:|
| Drought | 60 |
| Sandstorm | 86 |
| Thunderstorm | 120 |
| **Total** | **266** |

The notebook uses the Kaggle dataset mounted in its execution environment. Refer to the notebook's dataset configuration and the dataset provider's terms for the exact source and access requirements.

## Methodology

### 1. Image preprocessing

The paper describes resizing images to 224 × 224 pixels, normalization using ImageNet statistics, and training-time augmentation using random horizontal flips, rotations of ±15°, and color jittering.

These are the preprocessing and augmentation procedures described in the paper; exact execution details should be checked against the notebook cells.

### 2. Visual feature extraction using ResNet50

ResNet50 extracts high-level visual representations from disaster images. Its residual connections help train deep convolutional networks and learn discriminative image features.

The notebook reports a 2,048-dimensional ResNet50 feature vector.

### 3. Semantic representation using CLIP RN50

CLIP provides a shared representation space for images and natural-language descriptions.

The notebook creates multiple descriptive prompts for each disaster category, including general photographs, satellite images, disaster damage, aerial views, and ground-level views. The resulting text embeddings are normalized and averaged to produce class representations.

The notebook reports a 512-dimensional CLIP feature vector.

### 4. Multimodal feature fusion

The hybrid model concatenates the ResNet50 and CLIP RN50 feature vectors:

- ResNet50 features: 2,048 dimensions
- CLIP RN50 features: 512 dimensions
- **Combined feature representation: 2,560 dimensions**

This fused representation combines detailed visual information with semantic information from text embeddings.

### 5. Classification and zero-shot inference

The notebook implements a hybrid classifier that combines a traditional classifier with CLIP-based zero-shot predictions.

For unseen categories, CLIP text embeddings represent the disaster classes. Image and text representations are compared using cosine similarity to support zero-shot prediction.

The paper describes a training strategy involving a frozen-backbone phase followed by partial fine-tuning, and a prompt-based zero-shot inference stage. Refer to the paper and notebook for the full implementation details.

## Model Architectures Compared

The research compares four approaches:

| Model | Seen accuracy | Unseen accuracy | Macro F1 reported in paper |
|---|---:|---:|---:|
| Pure CLIP RN50 | 83.63% | 75.00% | 82.00% |
| ResNet50 | 89.00% | 76.00% | 88.20% |
| EfficientNet-B0+B1 (combined, fine-tuned) | 97.28% | 95.48% | 97.00% |
| **Hybrid ResNet50 + CLIP RN50** | **98.00%** | **99.62%** | **99.66%** |

*Source: Comparative results table in the supplied research paper. These values are reported by the publication and should not be assumed to be independently reproduced by the currently uploaded notebook.*

## Results

### Results printed by the supplied notebook

The final project summary in the supplied code PDF reports:

| Metric | Reported result |
|---|---:|
| Validation accuracy | 98.00% |
| Test accuracy | 99.62% |
| Rare-disaster accuracy | 99.62% |
| Fused feature dimension | 2,560 |

The notebook's test classification report provides the following per-class results:

| Unseen class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Drought | 1.00 | 1.00 | 1.00 | 60 |
| Sandstorm | 1.00 | 0.99 | 0.99 | 86 |
| Thunderstorm | 1.00 | 1.00 | 1.00 | 120 |

The displayed test report rounds its overall accuracy to 1.00. The more precise value printed separately by the notebook is **0.9962 (99.62%)**.

### Interpretation

The supplied results indicate strong classification performance on the reported test samples. The paper's comparative table also reports the hybrid ResNet50 + CLIP RN50 architecture as the best-performing approach among the four models evaluated.

These results should be interpreted in the context of the particular dataset, split, model implementation, and evaluation procedure. They do not establish that the same accuracy will generalize to every disaster dataset or real-world deployment environment.

## Implementation Details

The notebook uses Python and machine-learning libraries to load disaster images, extract features, perform classification, and evaluate predictions.

| Component | Role |
|---|---|
| Python | Main implementation language |
| NumPy and pandas | Numerical operations and data handling |
| PyTorch | Deep-learning tensor operations and model execution |
| ResNet50 | Visual feature extraction |
| CLIP RN50 | Image and text representations for zero-shot recognition |
| scikit-learn | Classification metrics and evaluation |
| Matplotlib and Seaborn | Visualization of evaluation results |

The exact installed package versions and execution environment should be taken from the Kaggle notebook. A pinned `requirements.txt` should only be added after those dependencies have been verified.

## Running the Notebook

The primary execution environment for this project is Kaggle.

1. Open the [MLDSCLIP4 notebook](https://www.kaggle.com/code/vaishnavipagad/mldsclip4).
2. Sign in to Kaggle if required.
3. Review the notebook's dataset configuration and confirm that the required dataset is attached.
4. Check the accelerator and library requirements used by the notebook.
5. Run the cells in order.
6. Review the feature-extraction logs, validation metrics, test classification report, and final project summary.

The notebook uses Kaggle-specific input paths. Running it locally requires obtaining the dataset, adapting those paths, and configuring compatible Python dependencies and hardware.

## Repository Structure

The current repository contains the original notebook:

```text
zero-shot-disaster-recognition/
└── zero-shot-disaster-recognition.ipynb
```

Additional documentation, reusable source modules, verified result figures, and dependency specifications can be added as the project is organized for reproducibility.

## Limitations and Reporting Notes

The supplied paper and code PDF contain discrepancies that should be resolved before treating every reported metric as directly comparable.

1. **Unseen-class accuracy:** The paper's abstract states 99.92%, whereas its comparative results table and the notebook's final summary report 99.62%. This README uses 99.62% for the notebook result and the paper's comparative table.
2. **Dataset distribution:** The paper's dataset table describes a different class distribution from the notebook's final summary. The notebook reports 1,600 training, 400 validation, and 266 test samples, with drought, sandstorm, and thunderstorm in the test set.
3. **Macro F1:** The paper's comparative table reports 99.66% for the hybrid model. The notebook's displayed test classification report shows a macro-average F1-score of 0.75, including a class with zero test support. These values use reporting contexts that need further verification and should not be presented as the same metric without clarification.
4. **Reproducibility:** The reported results depend on the dataset, experimental split, implementation, model configuration, and runtime environment. Independent reproduction is not claimed here.

These discrepancies are documented to keep the repository transparent. The source paper and notebook should be consulted for the full experimental context.

## Research Publication

**Title:** Comparative Analysis of Deep Learning and Multimodal Architectures for Zero-Shot Rare Disaster Recognition

**Publication:** IEEE Xplore

**Paper:** [https://ieeexplore.ieee.org/document/11507386](https://ieeexplore.ieee.org/document/11507386)

Please consult the published paper for the official author list, publication metadata, and citation details.

## Acknowledgements

This project uses pretrained model architectures and a disaster image dataset. Please refer to the Kaggle notebook, dataset documentation, and relevant model licenses for attribution, access conditions, and usage restrictions.

## Author

**Vaishnavi Pagad**

- GitHub: [@vaishzz-18](https://github.com/vaishzz-18)
- Research publication: [IEEE Xplore](https://ieeexplore.ieee.org/document/11507386)

