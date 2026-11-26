# Fetal Head Plane Classification with ResNet50

Classifies fetal head planes in ultrasound images using ResNet50 transfer learning, with Grad-CAM for explainability.

## Summary

| Metric | Value |
|---|---|
| Accuracy (test) | 0.83 |
| Weighted F1 | 0.83 |
| Best val accuracy | 0.8525 (epoch 34) |
| Split | 70% train / 15% val / 15% test (seed 42) |
| Total images | 2,845 (1,986 train) |

## Dataset

Three fetal head ultrasound planes:

| Class | Train | Val | Test |
|---|---|---|---|
| Transcerebellar | 478 | 103 | 103 |
| Transthalamic | 1095 | 235 | 235 |
| Transventricular | 413 | 89 | 89 |

The dataset is imbalanced, with transthalamic as the majority class. Training uses a `WeightedRandomSampler` and smoothed class weights (sqrt-normalized) to compensate.

## Per-class metrics (test set, 427 images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Transcerebellar | 0.86 | 0.91 | 0.89 |
| Transthalamic | 0.88 | 0.81 | 0.84 |
| Transventricular | 0.70 | 0.80 | 0.74 |

Transventricular is the weakest class (precision 0.70), which is common because the planes overlap visually.

## Model

- Backbone: ResNet50 pretrained on ImageNet
- Transfer learning: all layers frozen except `layer3` and `layer4`
- Head: `Dropout(0.4) → FC(256) → ReLU → Dropout(0.3) → FC(3)`
- Train augmentation: horizontal/vertical flip, 20° rotation, color jitter, conversion to 3-channel grayscale
- Loss: class-weighted `CrossEntropyLoss`
- Optimizer: Adam, lr=1e-4, StepLR (step 5, gamma 0.5)
- Early stopping: patience 7, max 40 epochs (stopped at epoch 34)
- Input size: 224×224

## Explainability with Grad-CAM

Grad-CAM maps activations from `layer4[-1]` to a heatmap showing which image regions drove each prediction. The implementation is bundled locally with no external dependency, so it runs in internet-off environments such as Kaggle.

Output: `outputs/gradcam_result.png`

## Results

- Training accuracy reaches about 0.96 while validation ends near 0.84. The gap of roughly 0.12 points to mild overfitting despite dropout and augmentation.
- Test accuracy is 0.83 with weighted F1 of 0.83.
- Results are consistent across runs: v7 scored 0.84 on test and v8 scored 0.83, so the score does not depend on a single lucky split.

## Reproducing

The notebook is ready to run on a Kaggle GPU. Grad-CAM is bundled, so no internet is needed to install requirements.

```bash
kaggle kernels push -p .
```

## Files

- `fetal-head-classification.ipynb`: full notebook (training, evaluation, Grad-CAM)
- `kernel-metadata.json`: Kaggle push configuration
- `raw_data/`: raw dataset (one folder per class)
- `outputs/`: plots written during the run
