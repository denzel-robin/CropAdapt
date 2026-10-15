# CropAdapt

**Discrimination-Guided Feature-Weighted Domain Adaptation for Plant Disease Recognition**

CropAdapt investigates domain adaptation for plant-disease image classification, focusing on the performance gap between controlled PlantVillage images and more varied, real-world PlantDoc images.

## Overview

Models trained on controlled leaf photographs may struggle when images contain different backgrounds, viewpoints, lighting, scales, and image quality. CropAdapt explores whether aligning source and target feature distributions—while reducing alignment pressure on dimensions that help distinguish source classes—can improve transfer performance.

The project uses:

- **DINOv2 ViT-S/14** as the pretrained visual encoder.
- **Low-Rank Adaptation (LoRA)** on the joint query-key-value (QKV) projections in all 12 transformer blocks.
- A **linear classifier** for 16 shared crop-disease categories.
- **Discrimination-guided, feature-weighted Maximum Mean Discrepancy (DGW-MMD)** for unlabeled target-domain alignment.
- A source class-separation statistic to regulate the strength of alignment.

## Repository

Source code and implementation:

[https://github.com/denzel-robin/CropAdapt/](https://github.com/denzel-robin/CropAdapt/)

## Datasets and experimental protocol

- **Source domain:** PlantVillage, with labeled images used for supervised training.
- **Target domain:** PlantDoc, with target adaptation images used without labels during adaptation.
- **Task:** Closed-set image classification over 16 categories shared by the two domains.
- **Target validation:** A separate 300-image set whose labels are used for checkpoint selection and diagnostic evaluation.
- **Target test:** Reserved and not evaluated in the report's reported results.

The report records 17,604 source training images, 4,487 source validation images, 1,199 unlabeled target adaptation images, and 300 target validation images. See the project report for the full split protocol and caveats.

## Architecture

1. An input image is processed by the shared DINOv2 ViT-S/14 encoder.
2. LoRA adapters update the joint QKV projections; the pretrained backbone weights remain frozen.
3. A linear classification head predicts one of the 16 crop-disease categories.
4. Source images contribute a supervised cross-entropy loss.
5. Source and unlabeled target features are compared using a feature-weighted, multi-kernel MMD objective.
6. A source class-separation statistic controls the alignment coefficient during adaptation.

Only the LoRA adapters and linear classifier are trainable. The report lists **153,616 trainable parameters out of 22,210,192 total parameters**.

## Reported validation results

The best result in the report's latest matched-budget comparison is the spread-aware **DGW-MMD v2-A** variant.

| Experiment | Target validation accuracy | Macro-F1 |
|---|---:|---:|
| Source only | 45.00% | 0.396580 |
| Conventional MMD, maximum weight 0.1 | 47.67% | 0.410413 |
| DGW-MMD v2-A, spread-aware | 48.67% | 0.419785 |
| DGW-MMD v2-B, bounded weighting | 47.67% | 0.410323 |
| DGW-MMD v2-C, smoothed controller | 47.67% | 0.410323 |

**Important:** These are exploratory results on a 300-image target validation set, using one recorded random seed. The reserved target test set remains unevaluated. The report notes class-level failures and does not establish that the improvement is statistically significant or suitable for practical diagnosis.

## Experimental setup

The report describes experiments run in Google Colab on an NVIDIA Tesla T4 GPU.

| Setting | Value |
|---|---|
| Encoder | DINOv2 ViT-S/14 |
| Feature dimension | 384 |
| LoRA rank | 8 |
| LoRA scale | 16 |
| LoRA dropout | 0.1 |
| Classifier | Linear, 384 → 16 |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 0.01 |
| Adaptation budget | 3 epochs |
| Maximum MMD coefficient | 0.1 |
| Random seed | 42 |
| Checkpoint selection | Best target-validation macro-F1 across adaptation epochs 1–3 |

Refer to the report and repository for implementation details and exact configuration files.

## Limitations

- Results are based on a single recorded seed and a small target validation set.
- Target validation labels were repeatedly used for checkpoint and method selection, which may inflate the reported validation scores.
- The reserved target test set has not been evaluated.
- Some classes have poor or zero recall in the reported class-level audit.
- The method performs closed-set classification; unknown-class rejection, localization, and recognition of unseen crop species are outside the evaluated scope.
- The reported results should not be compared directly with studies using different datasets, class mappings, or supervision protocols.

## Documentation

The project report, **CropAdapt: Discrimination-Guided Feature-Weighted Domain Adaptation for Plant Disease Recognition**, describes the motivation, related work, architecture, mathematical formulation, datasets, experimental protocol, results, ablations, and limitations.

## Citation

If you use this project, please cite the project report or repository according to the citation details provided by the authors.

## License

No license information is specified here. Check the repository for the project's current license and usage terms.
