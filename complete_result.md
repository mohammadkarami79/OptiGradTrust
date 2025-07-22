# COMPLETE VERIFIED RESULTS REPORT

This document lists **all experiments whose results have been fully executed and verified** on the current code-base. These numbers are directly reproducible by running `python main.py` with the corresponding dataset/model configuration and the optimiser **FedBN-Prox ("fedBN-P")** as described in `federated_learning/config/config.py`.

> NOTE Experiments not listed here (e.g.
> CIFAR-10 – ResNet18 – *IID – fedBN-P*) have **not yet been executed to completion** and are therefore excluded until real runs are performed.

---

## 1 MNIST – CNN – IID – FedBN-P

### Configuration Snapshot
| Hyper-parameter | Value |
|-----------------|-------|
| Dataset | MNIST (28×28 grayscale) |
| Model | 3-layer CNN |
| Batch Size | 64 (root + clients) |
| Root Epochs | 20 |
| Client Local Epochs | 8 |
| Global Rounds | 15 |
| Learning Rate | 0.01 (SGD, momentum 0.9) |
| Weight Decay | 1 × 10⁻⁴ |
| Optimiser | FedBN-Prox (μ = 0.1) |
| VAE Epochs | 15 |
| Dual-Attention Epochs | 8 |
| Shapley Samples | 15 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Baseline Accuracy | **99.41 %** |
| Final Accuracy | **99.41 %** |
| Accuracy Change | ±0.00 % |
| Detection Precision |
| • Scaling Attack | 30 % |
| • Partial Scaling Attack | 69.23 % |
| • Sign Flipping Attack | 47.37 % |
| • Noise Attack | 30 % |
| • Label Flipping | 27.59 % |

_Source file_: `results/final_paper_submission_ready/mnist_verified_results.csv` (2025-06-27 10:51–11:17)

---

## 2 Alzheimer – ResNet18 – IID – FedBN-P

### Configuration Snapshot
| Hyper-parameter | Value |
|-----------------|-------|
| Dataset | Alzheimer MRI (224×224 RGB) |
| Model | ResNet18 (ImageNet-pretrained) |
| Batch Size | 16 |
| Root Epochs | 20 |
| Client Local Epochs | 5 |
| Global Rounds | 25 |
| Learning Rate | 0.005 (SGD, momentum 0.9) |
| Weight Decay | 1 × 10⁻⁴ |
| Optimiser | FedBN-Prox (μ = 0.1) |
| VAE Epochs | 50 |
| Dual-Attention Epochs | 50 |
| Shapley Samples | 100 |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Baseline Accuracy | **97.24 %** |
| Final Accuracy | **96.92 %** |
| Accuracy Drop | −0.32 % |
| Detection Precision |
| • Scaling Attack | 42.86 % |
| • Partial Scaling Attack | 50.00 % |
| • Sign Flipping Attack | 57.14 % |
| • Noise Attack | 60.00 % |
| • Label Flipping | 75.00 % |

_Source file_: `results/final_paper_submission_ready/alzheimer_experiment_summary.txt` (25-round run, verified)

---

## 3 CIFAR-10 – ResNet18 – IID – FedBN-P (17 Jul 2025)

### Configuration Snapshot
| Hyper-parameter | Value |
|-----------------|-------|
| Dataset | CIFAR-10 |
| Model | ResNet18 (ImageNet-pretrained) |
| Clients | 10 (3 malicious – 30 %) |
| Global Rounds | 30 |
| Root Epochs | 20 |
| Client Local Epochs | 5 |
| Batch Size | 64 |
| Learning Rate | 0.005 (SGD, momentum 0.9) |
| VAE Epochs | 50 |
| Dual-Attention Epochs | 50 |
| Aggregation | FedBN-Prox (hybrid RL) |
| Data Distribution | IID |

### Performance Metrics
| Attack | Initial Acc | Final Acc | ΔAcc | Precision | Recall | F1 |
|--------|-------------|-----------|------|-----------|--------|----|
| Scaling | 84.28 % | 83.67 % | –0.61 % | 48.1 % | 70.0 % | 57.0 % |
| Partial Scaling | 83.67 % | 82.97 % | –0.70 % | 50.4 % | 68.9 % | 58.2 % |
| Sign Flipping | 82.97 % | 82.60 % | –0.37 % | 36.5 % | 51.1 % | 42.6 % |
| Noise | 82.60 % | 81.71 % | –0.89 % | 100 % | 100 % | 100 % |
| Label Flipping | 81.71 % | 81.27 % | –0.44 % | 50.0 % | 97.8 % | 66.2 % |

_Source files_:  `results/complete_result/comprehensive_attack_summary_20250717_032216.csv`  & `experiment_config_20250717_032216.csv`

---

### Pending Experiments (Not Yet Verified)

The following configurations are still **pending real execution**:

* All NON-IID configurations for MNIST, CIFAR-10 and Alzheimer

Once those runs finish and their outputs are validated, append their sections here and then update `ULTIMATE_COMPREHENSIVE_RESULTS_REPORT.md` accordingly.

---

_Last updated: 2025-07-16_ 