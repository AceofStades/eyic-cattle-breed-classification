# Indian Bovine Breed Classification Using Transfer Learning

> A deep learning pipeline for the automated classification of 41 Indian bovine breeds from photographic imagery, employing EfficientNet-B0 transfer learning with extensive data augmentation and ONNX-based deployment support.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Dataset Description](#dataset-description)
   - [Source Data](#source-data)
   - [Class Distribution](#class-distribution)
   - [Data Augmentation Strategy](#data-augmentation-strategy)
3. [Methodology](#methodology)
   - [Architecture Selection](#architecture-selection)
   - [Transfer Learning Protocol](#transfer-learning-protocol)
   - [Regularisation Techniques](#regularisation-techniques)
   - [Training Configuration](#training-configuration)
4. [Pipeline Architecture](#pipeline-architecture)
   - [Training Pipeline](#training-pipeline)
   - [Inference Pipeline](#inference-pipeline)
   - [Model Inspection Utility](#model-inspection-utility)
5. [Implementation Details](#implementation-details)
   - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
   - [Phase 1: Frozen Backbone Training](#phase-1-frozen-backbone-training)
   - [Phase 2: Full Fine-Tuning](#phase-2-full-fine-tuning)
   - [Checkpoint Management](#checkpoint-management)
   - [ONNX Export and Validation](#onnx-export-and-validation)
6. [Object Detection Integration](#object-detection-integration)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Repository Structure](#repository-structure)
9. [Environment Setup](#environment-setup)
10. [Usage](#usage)
    - [Training](#training)
    - [Resuming Training](#resuming-training)
    - [Inference](#inference)
    - [Model Inspection](#model-inspection)
11. [Technical Specifications](#technical-specifications)
12. [License](#license)
13. [Acknowledgements](#acknowledgements)

---

## Abstract

This repository presents an end-to-end deep learning system for the fine-grained visual classification of 41 bovine breeds native to or commonly found in India. The system is developed as part of the EYIC (Engineering Youth for India Competition) initiative. It leverages the EfficientNet-B0 convolutional neural network architecture, pre-trained on ImageNet, and fine-tuned on a curated dataset of 5,948 labelled bovine images. To combat class imbalance and limited per-class sample sizes, the pipeline incorporates a comprehensive offline data augmentation strategy that expands the training set by a factor of 11 (original plus 10 augmented copies per image), yielding approximately 52,338 training samples. The trained model is exported to the ONNX interchange format for cross-platform deployment and runtime optimisation.




---

## Dataset Description

### Source Data

The dataset comprises **5,948 images** distributed across **41 bovine breed classes**, sourced and organised under the `dataset/Indian_bovine_breeds/` directory. Each breed occupies its own subdirectory, and a CSV metadata file (`bovine_breeds_metadata.csv`) provides image-level annotations including the image identifier, breed label, and relative file path.

The dataset includes both indigenous Indian breeds (e.g., Gir, Sahiwal, Kangayam, Ongole) and internationally established breeds found in Indian dairy operations (e.g., Holstein Friesian, Jersey, Brown Swiss, Ayrshire).

### Class Distribution

The dataset exhibits significant class imbalance, a common challenge in fine-grained recognition tasks. The per-class sample counts are as follows:

| Breed | Samples | | Breed | Samples |
|---|---:|---|---|---:|
| Sahiwal | 439 | | Kangayam | 91 |
| Gir | 372 | | Nili Ravi | 89 |
| Holstein Friesian | 328 | | Nagori | 89 |
| Ayrshire | 234 | | Bhadawari | 86 |
| Brown Swiss | 225 | | Nimari | 84 |
| Tharparkar | 217 | | Dangi | 82 |
| Jersey | 203 | | Umblachery | 76 |
| Ongole | 191 | | Surti | 64 |
| Nagpuri | 187 | | Kenkatha | 55 |
| Hallikar | 186 | | Kherigarh | 36 |
| Kankrej | 179 | | Alambadi | 99 |
| Murrah | 173 | | Amritmahal | 94 |
| Red Dane | 167 | | Bargur | 94 |
| Red Sindhi | 166 | | Kasargod | 95 |
| Rathi | 149 | | Mehsana | 95 |
| Vechur | 140 | | Deoni | 99 |
| Krishna Valley | 136 | | Banni | 109 |
| Hariana | 129 | | Malnad Gidda | 107 |
| Pulikulam | 125 | | Jaffrabadi | 102 |
| Toda | 124 | | Khillari | 113 |
| Guernsey | 119 | | | |

The imbalance ratio between the most represented class (Sahiwal, 439 images) and the least represented class (Kherigarh, 36 images) is approximately 12:1. This imbalance is addressed through stratified splitting and augmentation strategies described below.

### Data Augmentation Strategy

To mitigate overfitting and class imbalance, an offline augmentation pipeline generates **10 augmented copies per training image**, stored persistently in the `dataset/augmented/` directory. The augmentation pipeline employs the [Albumentations](https://albumentations.ai/) library with the following stochastic transformations:

| Transformation | Parameters | Probability |
|---|---|---:|
| RandomResizedCrop | scale (0.6, 1.0), ratio (0.75, 1.33), output 224x224 | 1.0 |
| HorizontalFlip | -- | 0.5 |
| VerticalFlip | -- | 0.1 |
| Rotation | limit +/-30 degrees | 0.5 |
| ShiftScaleRotate | shift 0.1, scale 0.2, rotate +/-25 degrees | 0.5 |
| ColorJitter | brightness 0.3, contrast 0.3, saturation 0.3, hue 0.1 | 0.6 |
| RandomBrightnessContrast | brightness 0.2, contrast 0.2 | 0.5 |
| GaussNoise | -- | 0.3 |
| GaussianBlur | kernel (3, 5) | 0.2 |
| CoarseDropout | 1--8 holes, 8--20 px each | 0.3 |
| Affine | scale (0.8, 1.2), translate +/-10%, rotate +/-15 degrees | 0.4 |

This composition is applied independently to each copy, ensuring substantial variation across augmented samples. The augmented dataset totals approximately **52,338 images** and is persisted to disk with its own metadata CSV to avoid redundant regeneration across training runs.

---

## Methodology

### Architecture Selection

The classifier employs **EfficientNet-B0** as the backbone feature extractor, instantiated via the [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models) library. EfficientNet-B0 was selected for the following reasons:

1. **Parameter efficiency**: With approximately 5.3 million parameters, EfficientNet-B0 achieves a favourable accuracy-to-computation ratio, making it suitable for deployment on resource-constrained hardware.
2. **Compound scaling**: The EfficientNet family employs compound coefficient scaling across depth, width, and resolution, yielding architectures that are empirically more efficient than their manually designed counterparts (e.g., ResNet, VGG).
3. **Pre-trained representations**: ImageNet-pre-trained weights provide a strong initialisation for the convolutional feature hierarchy, enabling effective transfer to the bovine breed domain with limited labelled data.

The classification head consists of a single fully connected layer mapping the 1,280-dimensional feature embedding to 41 output logits, with a dropout rate of 0.4 applied prior to the final projection.

### Transfer Learning Protocol

Training follows a **two-phase transfer learning** strategy:

**Phase 1 -- Frozen Backbone (5 epochs):**
All parameters in the convolutional backbone are frozen, and only the classification head is trained. This allows the randomly initialised head to converge to a reasonable operating point without disrupting the pre-trained feature representations. A relatively high learning rate of 3e-3 is used with cosine annealing.

**Phase 2 -- Full Fine-Tuning (up to 30 epochs):**
All parameters are unfrozen and trained end-to-end. A discriminative learning rate scheme is applied:
- Backbone parameters: 1e-5 (FINETUNE_LR x 0.1)
- Classification head parameters: 1e-4

This differential rate ensures that pre-trained backbone weights are updated conservatively while the head adapts more aggressively. The scheduler employs cosine annealing with warm restarts (T_0 = 10, T_mult = 2) to facilitate escape from local minima.

### Regularisation Techniques

The pipeline incorporates multiple regularisation mechanisms to prevent overfitting on the relatively small per-class sample sizes:

| Technique | Implementation Details |
|---|---|
| **Label Smoothing** | Cross-entropy loss with smoothing factor epsilon = 0.1, distributing 10% of the probability mass uniformly across non-target classes. |
| **Dropout** | Applied at the classification head with rate p = 0.4. |
| **Mixup** | With probability 0.5 during fine-tuning, input images and labels are convexly combined using a Beta(0.3, 0.3) mixing coefficient. |
| **CutMix** | With probability 0.5 during fine-tuning (mutually exclusive with Mixup per batch), rectangular regions are cut and pasted between training samples using a Beta(1.0, 1.0) mixing coefficient. |
| **Gradient Clipping** | L2 norm clipping with max_norm = 1.0 to stabilise training dynamics. |
| **Weight Decay** | AdamW optimiser with weight decay coefficient lambda = 1e-2. |
| **Early Stopping** | Training halts if validation accuracy does not improve for 10 consecutive epochs. |
| **Offline Augmentation** | 10x augmented copies per training image (described above). |

### Training Configuration

The complete set of training hyperparameters is summarised below:

| Parameter | Value |
|---|---:|
| Input resolution | 224 x 224 pixels |
| Batch size | 32 |
| Phase 1 epochs (frozen) | 5 |
| Phase 2 epochs (fine-tune) | 30 (max) |
| Phase 1 learning rate | 3e-3 |
| Phase 2 learning rate (head) | 1e-4 |
| Phase 2 learning rate (backbone) | 1e-5 |
| Optimiser | AdamW |
| Weight decay | 1e-2 |
| Label smoothing | 0.1 |
| Dropout rate | 0.4 |
| Mixup alpha | 0.3 |
| CutMix alpha | 1.0 |
| Mixup/CutMix probability | 0.5 |
| Gradient clip norm | 1.0 |
| Early stopping patience | 10 epochs |
| Validation split | 20% (stratified) |
| Random seed | 42 |
| Number of workers | 4 |
| Augmentation copies | 10 |

---

## Pipeline Architecture

### Training Pipeline

The training pipeline (`train_pipeline.py`) is organised as a sequential six-stage process:

```
[1/6] Load Metadata
  |-- Read bovine_breeds_metadata.csv
  |-- Validate file paths
  v
[2/6] Encode Labels and Split
  |-- LabelEncoder: breed names -> integer indices
  |-- Stratified train/val split (80/20)
  |-- Persist label map to breed_labels.json
  v
[3/6] Generate Augmented Dataset
  |-- 10x offline augmentation per training image
  |-- Persist to dataset/augmented/
  v
[4/6] Train EfficientNet-B0
  |-- Phase 1: Frozen backbone (5 epochs)
  |-- Phase 2: Full fine-tuning (up to 30 epochs)
  |-- Checkpoint best and last models
  v
[5/6] Model Diagnostics
  |-- Load best checkpoint
  |-- Compute validation metrics (accuracy, F1, precision, recall)
  |-- Per-class classification report
  v
[6/6] ONNX Export
  |-- Export best model to breed_classifier.onnx
  |-- Validate ONNX graph integrity
  |-- Verify numerical equivalence with PyTorch model
```

### Inference Pipeline

The inference subsystem combines **object detection** with **breed classification** in a two-stage cascade:

1. **Detection**: YOLOv8-nano (`yolov8n.pt`) detects bovine instances in the input image using COCO class ID 19 (cow) with a configurable confidence threshold (default 0.5). Detected animals are cropped from the image.
2. **Classification**: Each cropped region is resized, normalised, and passed through the trained EfficientNet-B0 classifier. Softmax probabilities are computed to yield a breed prediction and associated confidence score.

If no bovine instances are detected, the full image is passed directly to the classifier as a fallback.

### Model Inspection Utility

The `model_info.py` script provides comprehensive diagnostic reporting for both PyTorch and ONNX model artefacts:

- **PyTorch inspection**: Parameter counts (total, trainable, frozen), memory footprint, layer-by-layer breakdown, per-block parameter distribution, and forward pass validation.
- **ONNX inspection**: Graph validity, opset version, node and initialiser counts, operator breakdown, and ONNX Runtime inference verification.
- **Cross-format comparison**: File size comparison, CPU inference latency benchmarking (averaged over 100 runs), and numerical equivalence testing between PyTorch and ONNX outputs.

---

## Implementation Details

### Data Loading and Preprocessing

Two PyTorch `Dataset` implementations are provided:

- **`BreedDataset`**: Serves the original (non-augmented) images from the source dataset. Used for the validation set.
- **`AugmentedBreedDataset`**: Serves images from the pre-generated augmented dataset directory. Used for the training set.

Both datasets apply transform pipelines at load time:

**Training transform:**
- Resize to 224 x 224
- ImageNet normalisation (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

**Validation transform:**
- Resize to 256 x 256
- Centre crop to 224 x 224
- ImageNet normalisation

Images are loaded via OpenCV and converted from BGR to RGB colour space.

### Phase 1: Frozen Backbone Training

During Phase 1, all parameters except those belonging to the `classifier` module are frozen (i.e., `requires_grad = False`). Only the final fully connected layer is updated, using AdamW with a learning rate of 3e-3 and cosine annealing over 5 epochs. Mixup and CutMix are disabled during this phase to provide clean gradient signals for head initialisation.

### Phase 2: Full Fine-Tuning

All parameters are unfrozen and trained with discriminative learning rates. The AdamW optimiser uses two parameter groups:

```python
[
    {"params": backbone_params, "lr": 1e-5},   # conservative backbone updates
    {"params": head_params,     "lr": 1e-4},    # aggressive head updates
]
```

Cosine annealing with warm restarts (T_0 = 10, T_mult = 2) enables periodic learning rate resets. Mixup and CutMix augmentation is active during this phase, applied stochastically with per-batch probability 0.5.

### Checkpoint Management

The pipeline maintains two checkpoint files:

| File | Purpose |
|---|---|
| `best_breed_classifier.pth` | Best model by validation accuracy. Contains model state dict, optimiser state, scheduler state, epoch, validation accuracy, and class count. |
| `last_checkpoint.pth` | Most recent epoch checkpoint. Enables training resumption from an arbitrary interruption point. |

Both checkpoints store sufficient state (including patience counters and training phase indicators) to enable seamless mid-training resumption via the `--resume` flag.

### ONNX Export and Validation

The best-performing PyTorch model is exported to the ONNX interchange format using `torch.onnx.export` with the following configuration:

- ONNX opset version 13
- Constant folding enabled
- Dynamic batch size axis
- Named inputs (`input`) and outputs (`output`)

Post-export validation includes:
1. ONNX graph structural validation via `onnx.checker.check_model`
2. ONNX Runtime inference test with random input
3. Numerical equivalence verification against PyTorch outputs (max absolute difference threshold: 1e-5)

---

## Object Detection Integration

The inference pipeline integrates **YOLOv8-nano** (from the [Ultralytics](https://github.com/ultralytics/ultralytics) framework) as a preprocessing stage for bovine localisation. This serves two purposes:

1. **Region of interest extraction**: Isolating individual animals from multi-subject or cluttered field photographs reduces background noise and improves classification accuracy.
2. **Multi-animal handling**: When multiple bovines are present in a single image, each is independently detected, cropped, and classified.

The detector targets COCO class 19 (cow) and applies a confidence threshold of 0.5 by default. Detection bounding boxes are used to crop PIL Image regions, which are then independently processed by the classification pipeline.

---

## Evaluation Metrics

The pipeline computes the following metrics on the held-out validation set (20% of the original data, stratified by class):

| Metric | Description |
|---|---|
| **Accuracy** | Proportion of correctly classified samples. |
| **Macro F1-Score** | Unweighted mean of per-class F1 scores. Treats all classes equally regardless of support, making it sensitive to performance on rare breeds. |
| **Weighted F1-Score** | Support-weighted mean of per-class F1 scores. Reflects overall performance proportional to class frequency. |
| **Macro Precision** | Unweighted mean of per-class precision values. |
| **Macro Recall** | Unweighted mean of per-class recall values. |
| **Per-Class Report** | Full classification report including precision, recall, F1, and support for each of the 41 breed classes. |

These metrics are computed using scikit-learn's `classification_report`, `f1_score`, `precision_score`, and `recall_score` functions.

---

## Repository Structure

```
eyic-cattle-breed-classification/
|
|-- train_pipeline.py              # End-to-end training, evaluation, and export pipeline
|-- model_info.py                  # Model inspection, diagnostics, and graph generation
|-- breed_labels.json              # Integer-to-breed-name label mapping (41 classes)
|-- best_breed_classifier.pth      # Best model checkpoint (PyTorch, git-ignored)
|-- last_checkpoint.pth            # Latest epoch checkpoint (PyTorch, git-ignored)
|-- environment.yml                # Conda environment specification
|-- LICENSE                        # MIT License
|-- README.md                      # This document
|
|-- graphs/                        # Generated visualisation outputs
|   |-- dataset_distribution.png
|   |-- parameter_distribution.png
|   |-- layer_type_distribution.png
|   |-- confusion_matrix.png
|   |-- confusion_matrix_normalised.png
|   |-- per_class_f1.png
|   |-- per_class_precision_recall.png
|   |-- metrics_summary.png
|   |-- top_bottom_classes.png
|   |-- model_size_comparison.png
|   +-- latency_comparison.png
|
|-- dataset/
|   |-- Indian_bovine_breeds/      # Source dataset
|   |   |-- bovine_breeds_metadata.csv
|   |   |-- Alambadi/
|   |   |-- Amritmahal/
|   |   |-- ...                    # 41 breed subdirectories
|   |   +-- Vechur/
|   |
|   +-- augmented/                 # Generated augmented training data
|       |-- augmented_metadata.csv
|       |-- Alambadi/
|       |-- Amritmahal/
|       |-- ...                    # 41 breed subdirectories
|       +-- Vechur/
```

---

## Environment Setup

The project requires Python 3.12 and is managed via Conda. A complete environment specification is provided in `environment.yml`.

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/eyic-cattle-breed-classification.git
cd eyic-cattle-breed-classification

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate breed-classifier
```

### Dependencies

The following key dependencies are utilised:

| Library | Purpose |
|---|---|
| PyTorch | Deep learning framework; model training and inference |
| timm | Pre-trained EfficientNet-B0 model and utilities |
| Albumentations | High-performance image augmentation pipeline |
| scikit-learn | Label encoding, train/test splitting, evaluation metrics |
| OpenCV (cv2) | Image I/O and colour space conversion |
| Pillow (PIL) | Image manipulation for inference cropping |
| Ultralytics | YOLOv8-nano object detection |
| ONNX | Model interchange format and graph validation |
| ONNX Runtime | Cross-platform optimised model inference |
| pandas | Metadata management and CSV I/O |
| NumPy | Numerical operations |
| tqdm | Progress bar display |
| Matplotlib / Seaborn | Visualisation and graph generation |

---

## Usage

### Training

Execute the full pipeline (augmentation, training, evaluation, and ONNX export):

```bash
python train_pipeline.py
```

To force regeneration of the augmented dataset (e.g., after modifying augmentation parameters):

```bash
python train_pipeline.py --force-augment
```

To skip training and only run evaluation and export on an existing checkpoint:

```bash
python train_pipeline.py --skip-training
```

To skip the ONNX export stage:

```bash
python train_pipeline.py --skip-export
```

### Resuming Training

If training is interrupted, it can be resumed from the last saved checkpoint:

```bash
python train_pipeline.py --resume
```

The pipeline will automatically detect the training phase (frozen or fine-tune), epoch, optimiser state, scheduler state, and patience counter from the checkpoint, ensuring seamless continuation.

### Inference

To classify a single image using the trained model and YOLOv8-based detection:

```bash
python train_pipeline.py --skip-training --skip-export --demo-image path/to/image.jpg
```

The output reports each detected bovine instance along with the predicted breed, classification confidence, and detection confidence.

### Model Inspection

To display detailed model diagnostics:

```bash
# Full report (PyTorch + ONNX + comparison)
python model_info.py

# PyTorch model only
python model_info.py --pytorch-only

# ONNX model only
python model_info.py --onnx-only

# Cross-format comparison (size, latency, numerical equivalence)
python model_info.py --compare
```

### Visualisation and Graph Generation

The `model_info.py` script can generate a comprehensive suite of publication-quality visualisations. All graphs are saved as PNG files to the `graphs/` directory.

```bash
# Generate all graphs (runs validation evaluation automatically)
python model_info.py --graphs

# Generate graphs only, skip text diagnostics
python model_info.py --graphs-only

# Run evaluation metrics without graphs
python model_info.py --eval
```

The following visualisations are produced:

| Graph | Description |
|---|---|
| `dataset_distribution.png` | Horizontal bar chart of per-breed sample counts in the source dataset, revealing class imbalance. |
| `parameter_distribution.png` | Parameter count distribution across top-level model blocks (conv_stem, blocks, classifier, etc.) with percentage annotations. |
| `layer_type_distribution.png` | Pie chart of layer type composition (Conv2d, BatchNorm2d, SiLU, etc.) within the EfficientNet-B0 architecture. |
| `confusion_matrix.png` | Full 41x41 confusion matrix heatmap with absolute counts, showing classification patterns and common misclassifications. |
| `confusion_matrix_normalised.png` | Row-normalised confusion matrix (per-class recall), useful for identifying classes with systematic misclassification. |
| `per_class_f1.png` | Horizontal bar chart of F1 scores for all 41 breed classes, sorted ascending, with macro-average reference line. |
| `per_class_precision_recall.png` | Grouped horizontal bar chart comparing precision and recall side-by-side for each breed class. |
| `metrics_summary.png` | Bar chart of aggregate evaluation metrics (accuracy, macro F1, weighted F1, macro precision, macro recall). |
| `top_bottom_classes.png` | Side-by-side comparison of the 10 best-performing and 10 worst-performing classes by F1 score, with support counts. |
| `model_size_comparison.png` | Bar chart comparing PyTorch (.pth) vs ONNX (.onnx) file sizes. |
| `latency_comparison.png` | Bar chart comparing CPU inference latency between PyTorch and ONNX Runtime (averaged over 100 forward passes). |

### Generated Graphs

#### Dataset Distribution
![Dataset Distribution](graphs/dataset_distribution.png)

#### Parameter Distribution
![Parameter Distribution](graphs/parameter_distribution.png)

#### Layer Type Distribution
![Layer Type Distribution](graphs/layer_type_distribution.png)

#### Confusion Matrix
![Confusion Matrix](graphs/confusion_matrix.png)

#### Normalised Confusion Matrix
![Normalised Confusion Matrix](graphs/confusion_matrix_normalised.png)

#### Per-Class F1 Score
![Per-Class F1 Score](graphs/per_class_f1.png)

#### Per-Class Precision and Recall
![Per-Class Precision and Recall](graphs/per_class_precision_recall.png)

#### Metrics Summary
![Metrics Summary](graphs/metrics_summary.png)

#### Top and Bottom Performing Classes
![Top and Bottom Classes](graphs/top_bottom_classes.png)

#### Model Size Comparison
![Model Size Comparison](graphs/model_size_comparison.png)

#### Latency Comparison
![Latency Comparison](graphs/latency_comparison.png)

---

## Technical Specifications

| Specification | Detail |
|---|---|
| Architecture | EfficientNet-B0 (timm) |
| Pre-training | ImageNet-1K |
| Input dimensions | 3 x 224 x 224 (RGB, normalised) |
| Output dimensions | 41 (one logit per breed class) |
| Total parameters | ~5.3 million |
| Classification head | Linear(1280, 41) with Dropout(0.4) |
| Loss function | CrossEntropyLoss with label smoothing (epsilon = 0.1) |
| Optimiser | AdamW |
| Training framework | PyTorch |
| Export format | ONNX (opset 18, dynamic batch) |
| Detection model | YOLOv8-nano (Ultralytics, COCO pre-trained) |
| Number of classes | 41 |
| Training samples | ~52,338 (after 10x augmentation) |
| Validation samples | ~1,190 (20% stratified holdout, no augmentation) |

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Varun Jhaveri

---

## Acknowledgements

- **EfficientNet**: Tan, M. and Le, Q. V., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.
- **timm**: Wightman, R., "PyTorch Image Models," GitHub repository, 2019. Available: https://github.com/huggingface/pytorch-image-models
- **Albumentations**: Buslaev, A. et al., "Albumentations: Fast and Flexible Image Augmentations," Information, vol. 11, no. 2, p. 125, 2020.
- **YOLOv8**: Jocher, G. et al., "Ultralytics YOLOv8," 2023. Available: https://github.com/ultralytics/ultralytics
- **Mixup**: Zhang, H. et al., "mixup: Beyond Empirical Risk Minimization," in Proceedings of the International Conference on Learning Representations (ICLR), 2018.
- **CutMix**: Yun, S. et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
