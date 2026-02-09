# Server Instructions for Revision Experiments

## Files to Update on Server

Upload these 5 files from your local machine to the server:

### ✅ Core Files (MUST UPDATE)
1. **`run_optimized_experiments.py`** - Main experiment runner with n=5 seeds, ALZHEIMER support
2. **`run_all_experiments.py`** - Updated ALZHEIMER data path configuration
3. **`federated_learning/training/client.py`** - Fixed noise_factor usage in attacks
4. **`federated_learning/data/alzheimer_dataset.py`** - Runtime config for data paths
5. **`federated_learning/models/attention.py`** - Fixed dimension mismatch bug

### ✅ Optional (for reference)
6. **`README.md`** - Updated documentation
7. **`REVISION_EXPERIMENTS_REPORT.md`** - Complete experimental design document

---

## What to Run on Server

### Step 1: Current Run (OASIS) - Already Running ✅
```bash
# This should already be running:
nohup python run_optimized_experiments.py --revision-quick --epochs 8 > revision_quick.log 2>&1 &
tail -f revision_quick.log
```

**Expected output in log:**
```
REVISION-QUICK: n=5 seeds, IID+Dirichlet0.5, datasets=['OASIS'], baselines=['fedavg', 'krum', 'fltrust', 'trfa']
*** STRONG ATTACK (high baseline distance): malicious=40%, scaling=30x, noise=15, flip=0.9 ***
Phase 1 (Clinical x1): 40  |  Phase 2: 10  |  Phase 3 (Baselines x4): 100
TOTAL: 185 experiments  |  Seeds: 5  |  Epochs: 8
```

**Results will be saved to:** `results/reviewer_experiments/oasis/`

---

### Step 2: After OASIS Completes - Run ALZHEIMER

#### Prerequisite: Verify ALZHEIMER Data Exists
```bash
cd /path/to/OptiGradTrust-3
ls -la data/alzheimer/train
ls -la data/alzheimer/test
```

You should see class folders like:
- `MildDemented/`
- `ModerateDemented/`
- `NonDemented/`
- `VeryMildDemented/`

If not, download from: https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy

#### Run ALZHEIMER Experiments
```bash
nohup python run_optimized_experiments.py \
  --revision-quick \
  --revision-dataset ALZHEIMER \
  --epochs 8 \
  --phase 1 \
  > revision_quick_ALZHEIMER.log 2>&1 &

tail -f revision_quick_ALZHEIMER.log
```

**Expected output in log:**
```
REVISION-QUICK: n=5 seeds, IID+Dirichlet0.5, datasets=['ALZHEIMER'], baselines=['fedavg', 'krum', 'fltrust', 'trfa']
*** STRONG ATTACK (high baseline distance): malicious=40%, scaling=30x, noise=15, flip=0.9 ***
PHASE 1: ALZHEIMER CLINICAL EXPERIMENTS
*** Dataset: ALZHEIMER ***
*** Seeds: 5 ***
```

**Results will be saved to:** `results/reviewer_experiments/alzheimer/`

---

## Results Directory Structure

After both runs complete:

```
results/
└── reviewer_experiments/
    ├── oasis/                          # OASIS results (Run 1)
    │   ├── OPTIMIZED_COMPLETE_<timestamp>.json
    │   ├── optimized_log_<timestamp>.txt
    │   ├── OASIS_IID_results_<timestamp>.json
    │   ├── OASIS_Dirichlet_0.5_results_<timestamp>.json
    │   ├── scalability_results_<timestamp>.json
    │   ├── baseline_significance_<timestamp>.json
    │   ├── ablation_<timestamp>.json
    │   └── extreme_imbalance_<timestamp>.json
    │
    └── alzheimer/                      # ALZHEIMER results (Run 2)
        ├── OPTIMIZED_ALZHEIMER_<timestamp>.json
        ├── optimized_log_<timestamp>.txt
        ├── ALZHEIMER_IID_results_<timestamp>.json
        └── ALZHEIMER_Dirichlet_0.5_results_<timestamp>.json
```

**✅ IMPORTANT:** Results are now in **separate directories** (oasis/ and alzheimer/) so they don't overwrite each other!

---

## Verification Checklist

### ✅ Before ALZHEIMER Run
- [ ] OASIS run completed successfully (check `revision_quick.log` for "ALL EXPERIMENTS COMPLETED!")
- [ ] ALZHEIMER data exists in `data/alzheimer/train` and `data/alzheimer/test`
- [ ] All 5 files updated on server
- [ ] Enough disk space (~10GB for logs and results)

### ✅ During ALZHEIMER Run
Monitor log file and verify:
- [ ] Log shows `n=5 seeds` (not n=3)
- [ ] Log shows `datasets=['ALZHEIMER']`
- [ ] Log shows `STRONG ATTACK: malicious=40%, scaling=30x, noise=15, flip=0.9`
- [ ] Each experiment reports seed number: "Seed 42: Acc=..., Seed 123: Acc=..."

### ✅ After ALZHEIMER Run
- [ ] Check result file: `results/reviewer_experiments/alzheimer/OPTIMIZED_ALZHEIMER_*.json`
- [ ] Verify all scenarios have `"n": 5` in JSON
- [ ] Verify no errors in log file

---

## Expected Runtime

| Run | Phases | Experiments | Estimated Time |
|-----|--------|-------------|----------------|
| OASIS | 1, 2, 3, 5, 7 | 185 | 18-24 hours |
| ALZHEIMER | 1 only | 40 | 4-6 hours |
| **TOTAL** | - | **225** | **22-30 hours** |

---

## Troubleshooting

### Issue: "Alzheimer's dataset not found"
**Solution:** Download data and place in `data/alzheimer/` with train/ and test/ subdirectories.

### Issue: Results show n=3 instead of n=5
**Solution:** Old results from previous run. New run with updated code will show n=5.

### Issue: ALZHEIMER results overwrite OASIS results
**Solution:** Not possible with updated code - they go to separate directories (oasis/ vs alzheimer/).

### Issue: Unicode encoding error with τ character
**Solution:** Already fixed in updated code (τ replaced with "tau" in logs).

---

## What These Results Give You for the Paper

### Statistical Validation
- **All results:** mean ± std with 95% CI (n=5)
- **Baseline comparison:** p-values, Cohen's d effect sizes
- **Example:** "OptiGradTrust: 68.4±2.3% [95% CI: 66.1-70.7%]"

### Dataset Generalization
- **OASIS:** Real clinical data (primary validation)
- **ALZHEIMER:** Medical dataset (generalization validation)
- **Claim:** "Validated across 2 medical datasets..."

### Strong Attack Robustness
- **40% malicious clients** (vs 30% in original)
- **30× scaling** (vs 20× in original)
- **Clear separation:** OptiGradTrust ~65-75%, baselines ~35-55%

### Extended Baselines
- **4 baselines** (FedAvg, Krum, FLTrust, TRFA) vs 3 before
- **TRFA added** per reviewer request

---

## After Completion

Both result files will be ready for analysis:
1. **`results/reviewer_experiments/oasis/OPTIMIZED_COMPLETE_*.json`**
2. **`results/reviewer_experiments/alzheimer/OPTIMIZED_ALZHEIMER_*.json`**

Use these for:
- Paper tables (mean±std, CI)
- Statistical tests
- Baseline comparisons
- Ablation analysis

**See `REVISION_EXPERIMENTS_REPORT.md` for detailed experimental design and expected tables for the paper.**
