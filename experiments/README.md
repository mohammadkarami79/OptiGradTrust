# OptiGradTrust Experiments Suite

## Comprehensive Experiments for Reviewer Feedback

This directory contains all experiments designed to address reviewer feedback for the OptiGradTrust paper submission.

---

## 📋 Overview

### Critical Experiments (Priority 1) - MUST RUN
1. **Ablation Study** - Drop-one-feature analysis
2. **Computational Overhead** - Runtime & memory profiling
3. **Fair Comparison** - Baselines with FedBN-P
4. **Extended Metrics** - Precision, Recall, F1, AUC
5. **Confidence Intervals** - Multiple seeds (statistical validity)

### Important Experiments (Priority 2) - SHOULD RUN
6. **Combined Attacks** - Multiple simultaneous attacks
7. **Scalability (Clients)** - 10, 20, 50, 100 clients
8. **Scalability (Adversarial)** - 10-50% malicious ratios
9. **Feature Correlation** - Independence analysis
10. **Extreme Heterogeneity** - α=0.05, 0.01

### Additional Analyses (Priority 3) - NICE TO HAVE
11. **Statistical Significance** - t-tests for comparisons
12. **Preprocessing Docs** - Complete pipeline documentation
13. **Comprehensive Visualizations** - All plots for paper

---

## 🚀 Quick Start

### Run ALL Experiments (Recommended for Paper)

```bash
# Windows
python experiments/run_all_experiments.py --all

# Linux/Mac
python3 experiments/run_all_experiments.py --all
```

### Run Specific Priority Level

```bash
# Priority 1 only (critical - ~2-3 hours)
python experiments/run_all_experiments.py --priority 1

# Priority 1 + 2 (critical + important - ~5-6 hours)
python experiments/run_all_experiments.py --priority 2

# All experiments (~8-10 hours)
python experiments/run_all_experiments.py --priority 3
```

### Quick Test (Verify Setup)

```bash
python experiments/run_all_experiments.py --quick
```

---

## 📂 Directory Structure

```
experiments/
├── README.md                          # This file
├── run_all_experiments.py             # Master runner
├── __init__.py
│
├── ablation_study.py                  # Drop-one-feature analysis
├── combined_attacks.py                # Multiple simultaneous attacks
├── computational_overhead.py          # Runtime & memory profiling
├── confidence_intervals.py            # Multiple seeds
├── extended_metrics.py                # P/R/F1/AUC/CM
├── fair_comparison.py                 # Baselines with FedBN-P
├── scalability_tests.py               # Varying clients & adversarial ratios
├── feature_correlation.py             # Correlation heatmap
├── extreme_heterogeneity.py           # α=0.05, 0.01
├── statistical_tests.py               # Significance tests
├── visualization_suite.py             # Comprehensive plots
├── preprocessing_docs.py              # Pipeline documentation
│
├── configs/                           # Configuration files
└── results/                           # All experiment outputs
    ├── session_YYYYMMDD_HHMMSS/      # Timestamped session
    │   ├── ablation/
    │   ├── combined_attacks/
    │   ├── overhead/
    │   ├── extended_metrics/
    │   ├── confidence_intervals/
    │   ├── fair_comparison/
    │   ├── scalability/
    │   ├── correlation/
    │   ├── extreme_heterogeneity/
    │   ├── statistical_tests/
    │   ├── visualizations/
    │   ├── preprocessing_docs/
    │   └── master_results.json        # Summary of all experiments
    └── ...
```

---

## 🔬 Individual Experiment Modules

### 1. Ablation Study

Tests contribution of each feature by removing it:

```bash
python experiments/ablation_study.py --output-dir experiments/results/ablation
```

**Output:**
- `ablation_results_*.json` - Detailed results
- `ablation_summary_*.csv` - Summary table (for paper)

**Addresses:** Reviewer 3's concern about feature necessity and independence.

---

### 2. Combined Attacks

Tests robustness against multiple simultaneous attacks:

```bash
python experiments/combined_attacks.py --output-dir experiments/results/combined_attacks
```

**Attack Combinations:**
- Scaling + Noise
- Sign Flip + Label Flip
- Partial Scaling + Noise
- Scaling + Sign Flip
- All Combined (worst case)

**Addresses:** Reviewer 2's question: "Do attacks occur isolated or combined?"

---

### 3. Computational Overhead

Profiles runtime and memory usage:

```bash
python experiments/computational_overhead.py --output-dir experiments/results/overhead --rounds 10
```

**Metrics:**
- Per-round timing
- Component breakdown (VAE, Shapley, Attention, RL)
- Memory usage (peak & average)
- Scalability estimates

**Addresses:** Reviewers 1, 3, 4 concerns about computational cost.

---

### 4. Extended Metrics

Computes comprehensive evaluation metrics:

```bash
python experiments/extended_metrics.py --output-dir experiments/results/extended_metrics
```

**Metrics:**
- Precision, Recall, F1-Score (per class & averaged)
- AUC-ROC
- Confusion Matrix
- Per-class performance

**Addresses:** Reviewers 1 & 2 request for metrics beyond accuracy.

---

### 5. Confidence Intervals

Runs multiple experiments with different seeds:

```bash
python experiments/confidence_intervals.py --output-dir experiments/results/confidence_intervals --num-runs 5
```

**Output:**
- Mean ± Std for all metrics
- 95% Confidence intervals
- Statistical variability assessment

**Addresses:** Reviewer 3's requirement for statistical validity.

---

### 6. Scalability Tests

Tests varying number of clients and adversarial ratios:

```bash
# Varying clients
python experiments/scalability_tests.py --test clients --output-dir experiments/results/scalability

# Varying adversarial ratios
python experiments/scalability_tests.py --test adversarial --output-dir experiments/results/scalability

# Both
python experiments/scalability_tests.py --test both --output-dir experiments/results/scalability
```

**Addresses:** Reviewers 2 & 3 concerns about scalability and performance limits.

---

### 7. Feature Correlation

Analyzes independence of 6 fingerprint features:

```bash
python experiments/feature_correlation.py --output-dir experiments/results/correlation --samples 200
```

**Output:**
- Pearson correlation matrix
- Correlation heatmap
- Independence analysis

**Addresses:** Reviewer 3's concern about feature redundancy.

---

### 8. Extreme Heterogeneity

Tests under extreme non-IID conditions:

```bash
python experiments/extreme_heterogeneity.py --output-dir experiments/results/extreme_heterogeneity
```

**Conditions:**
- Dirichlet α = 0.05, 0.01
- Label skew = 95%, 99%

**Addresses:** Reviewer 4's request for testing extreme heterogeneity.

---

### 9. Fair Comparison

Compares against baselines when ALL use FedBN-P:

```bash
python experiments/fair_comparison.py --output-dir experiments/results/fair_comparison
```

**Baselines (all with FedBN-P):**
- OptiGradTrust
- FLGuard
- FLTrust
- FLAME

**Addresses:** Reviewer 2's concern about unfair optimizer advantage.

---

### 10. Statistical Significance

Performs t-tests for statistical significance:

```bash
python experiments/statistical_tests.py --output-dir experiments/results/statistical_tests
```

**Addresses:** Need for statistical validation of improvements.

---

### 11. Preprocessing Documentation

Documents complete preprocessing pipeline:

```bash
python experiments/preprocessing_docs.py --output-dir experiments/results/preprocessing_docs
```

**Addresses:** Reviewer 1's concern about reproducibility.

---

## 📊 Expected Runtime

| Experiment | Estimated Time | Priority |
|-----------|----------------|----------|
| Ablation Study | 2-3 hours | P1 |
| Computational Overhead | 30 min | P1 |
| Fair Comparison | 2 hours | P1 |
| Extended Metrics | 1 hour | P1 |
| Confidence Intervals | 3-4 hours | P1 |
| Combined Attacks | 1.5 hours | P2 |
| Scalability (Clients) | 3 hours | P2 |
| Scalability (Adversarial) | 1.5 hours | P2 |
| Feature Correlation | 30 min | P2 |
| Extreme Heterogeneity | 1 hour | P2 |
| Statistical Tests | 5 min | P3 |
| Preprocessing Docs | 2 min | P3 |
| Visualizations | 10 min | P3 |

**Total (All):** ~15-18 hours on single GPU  
**Total (P1 only):** ~8-10 hours  

---

## 🖥️ Server Deployment

### Copy to Server

```bash
# From local machine
scp -r new_paper/ user@server:/path/to/project/
```

### Run on Server (Background)

```bash
# SSH to server
ssh user@server

# Navigate to project
cd /path/to/project/new_paper

# Activate virtual environment (if needed)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run experiments in background
nohup python experiments/run_all_experiments.py --all > experiments_output.log 2>&1 &

# Check progress
tail -f experiments_output.log
```

---

## 📦 Output Files

Each experiment generates:
- **JSON** - Detailed results with full metadata
- **CSV** - Summary tables (ready for paper/Excel)
- **PNG** - Visualization plots (publication quality, 300 DPI)

All files are timestamped and organized in session directories.

---

## ✅ Verification Checklist

Before paper revision:

- [ ] Run Priority 1 experiments (critical)
- [ ] Verify all CSV files generated
- [ ] Check plots are publication quality
- [ ] Review master_results.json for completeness
- [ ] Compare results with original paper claims
- [ ] Update paper tables with new results
- [ ] Add new plots to paper figures
- [ ] Address each reviewer comment with corresponding experiment

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `federated_learning/config/config.py`
- Use fewer clients for initial testing
- Run experiments sequentially instead of in batch

### Import Errors
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/new_paper"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;D:\new_paper  # Windows
```

### Slow Experiments
- Use `--quick` flag for initial testing
- Reduce number of rounds (edit modules)
- Use smaller subsets of tests

---

## 📧 Contact

For questions about experiments setup or results interpretation, please refer to the main project README.

---

**Last Updated:** 2025-10-08

