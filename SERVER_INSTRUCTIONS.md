# 🖥️ Server Deployment Instructions for OptiGradTrust Revision Experiments

## Overview

This guide provides **step-by-step instructions** for deploying and running OptiGradTrust revision experiments on your server for both **OASIS** and **ALZHEIMER** datasets.

---

## ✅ Prerequisites

Before starting, ensure you have:

1. ✅ **Git repository access**: `https://github.com/mohammadkarami79/OptiGradTrust.git`
2. ✅ **Python environment**: Python 3.8+ with conda/venv
3. ✅ **GPU**: CUDA-capable GPU (recommended)
4. ✅ **OASIS dataset**: Located at `oasis_cross-sectional_disc1/disc1/` (or update path in config)
5. ✅ **ALZHEIMER dataset**: Located at `data/alzheimer/` (with train/ and test/ subdirectories)

---

## 📋 Step-by-Step Deployment

### **Step 1: Clone/Update Repository on Server**

```bash
# If not yet cloned:
cd /home/gpu/FLBrain/
git clone https://github.com/mohammadkarami79/OptiGradTrust.git OptiGradTrust-3
cd OptiGradTrust-3

# If already cloned, pull latest changes:
cd /home/gpu/FLBrain/OptiGradTrust-3
git pull
```

**Expected output:**
```
Already up to date.
# OR
Updating 8847879..4565504
Fast-forward
 .gitignore                                | 11 +++++++++++
 REVISION_EXPERIMENTS_REPORT.md (deleted)  | ...
 SERVER_INSTRUCTIONS.md (deleted)          | ...
 3 files changed, 11 insertions(+), 664 deletions(-)
```

---

### **Step 2: Verify Critical Files Exist**

```bash
# Check experiment scripts
ls -lh run_optimized_experiments.py run_all_experiments.py

# Expected output:
# -rw-r--r-- 1 user group 45K Feb  5 10:00 run_optimized_experiments.py
# -rw-r--r-- 1 user group 23K Feb  5 10:00 run_all_experiments.py

# Check federated_learning module
ls -lh federated_learning/training/server.py federated_learning/training/client.py

# Check OASIS demographics file
ls -lh oasis_cross-sectional.xlsx
```

**If any files are missing**, re-pull from GitHub:
```bash
git reset --hard HEAD
git pull
```

---

### **Step 3: Activate Python Environment**

```bash
# Activate conda environment
conda activate optigrad_py311

# OR activate virtualenv
source venv/bin/activate

# Verify Python version
python --version
# Expected: Python 3.8+ (preferably 3.11)

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.0+, CUDA available: True
```

---

### **Step 4: Clean Old Logs (Optional but Recommended)**

```bash
# Remove old experiment logs
rm -f revision_quick*.log optimized*.log

# Remove old result files (if you want fresh results)
rm -rf results/reviewer_experiments/oasis/*
rm -rf results/reviewer_experiments/alzheimer/*

# Keep directory structure
mkdir -p results/reviewer_experiments/oasis
mkdir -p results/reviewer_experiments/alzheimer
```

---

## 🚀 Running Experiments

### **Experiment 1: OASIS Dataset (Complete Suite)**

This runs **all 7 phases** for OASIS dataset with **n=5 seeds** and **strong attacks**.

```bash
cd /home/gpu/FLBrain/OptiGradTrust-3
conda activate optigrad_py311

# Run in background with nohup
nohup python run_optimized_experiments.py --revision-quick --epochs 8 > revision_quick.log 2>&1 &

# Get process ID
echo $!
# Save this PID in case you need to stop the process
```

**Expected runtime:** 18-24 hours

**Phases executed:**
- Phase 1: OASIS Clinical (40 experiments)
- Phase 2: Scalability (10 experiments)
- Phase 3: Baselines (80 experiments)
- Phase 4: RL Sensitivity (25 experiments)
- Phase 5: Ablation (20 experiments)
- Phase 6: τ Sensitivity (15 experiments)
- Phase 7: Extreme Imbalance (10 experiments)

**Total:** ~185 experiments

---

### **Experiment 2: ALZHEIMER Dataset (Phase 1 Only)**

**⚠️ IMPORTANT: Run this AFTER OASIS experiments complete!**

```bash
cd /home/gpu/FLBrain/OptiGradTrust-3
conda activate optigrad_py311

# Run ALZHEIMER experiments
nohup python run_optimized_experiments.py --revision-quick --revision-dataset ALZHEIMER --epochs 8 --phase 1 > revision_quick_ALZHEIMER.log 2>&1 &

# Get process ID
echo $!
```

**Expected runtime:** 4-6 hours

**Phases executed:**
- Phase 1 only: ALZHEIMER Clinical (40 experiments)

**Why only Phase 1?** Other phases (scalability, ablation, etc.) are dataset-independent and already covered by OASIS experiments.

---

## 📊 Monitoring Progress

### **Real-time Log Viewing**

```bash
# View OASIS log in real-time
tail -f revision_quick.log

# View ALZHEIMER log in real-time
tail -f revision_quick_ALZHEIMER.log

# View last 100 lines
tail -100 revision_quick.log

# Search for specific phase
grep "Phase 1" revision_quick.log
grep "Phase 3" revision_quick.log
```

### **Check Running Processes**

```bash
# Find Python processes
ps aux | grep python

# Expected output:
# user  12345  ... python run_optimized_experiments.py --revision-quick --epochs 8

# Check GPU usage
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi
```

### **Stop a Running Experiment**

```bash
# Find process ID
ps aux | grep run_optimized_experiments.py

# Kill process (replace <PID> with actual PID)
kill <PID>

# Force kill if needed
kill -9 <PID>
```

---

## 📂 Result Files

### **Directory Structure**

After experiments complete, results will be in:

```
results/
└── reviewer_experiments/
    ├── oasis/
    │   ├── OPTIMIZED_COMPLETE_20260205_143022.json    # All OASIS phases combined
    │   ├── optimized_log_20260205_143022.txt          # Detailed experiment log
    │   ├── phase1_oasis_clinical_20260205_143022.json
    │   ├── phase2_scalability_20260205_143022.json
    │   ├── phase3_baselines_20260205_143022.json
    │   ├── phase4_rl_sensitivity_20260205_143022.json
    │   ├── phase5_ablation_20260205_143022.json
    │   ├── phase6_tau_sensitivity_20260205_143022.json
    │   └── phase7_extreme_imbalance_20260205_143022.json
    └── alzheimer/
        ├── OPTIMIZED_ALZHEIMER_20260206_120334.json   # ALZHEIMER Phase 1 only
        └── optimized_log_20260206_120334.txt          # Detailed experiment log
```

### **Download Results to Local Machine**

```bash
# From your local machine, run:
scp -r user@server:/home/gpu/FLBrain/OptiGradTrust-3/results/reviewer_experiments/ ./local_results/

# Or download specific files:
scp user@server:/home/gpu/FLBrain/OptiGradTrust-3/results/reviewer_experiments/oasis/OPTIMIZED_COMPLETE_*.json ./
scp user@server:/home/gpu/FLBrain/OptiGradTrust-3/results/reviewer_experiments/alzheimer/OPTIMIZED_ALZHEIMER_*.json ./
```

---

## 🔍 Verification Checks

### **Before Running Experiments**

```bash
# Check Python environment
python -c "import torch, federated_learning; print('✅ All imports successful')"

# Verify OASIS data path
python -c "from federated_learning.config import config; print(f'OASIS path: {config.OASIS_DATA_ROOT}')"

# Verify ALZHEIMER data path
python -c "from federated_learning.config import config; print(f'ALZHEIMER path: {config.ALZHEIMER_DATA_DIR}')"

# Quick test (runs 1-2 experiments to verify setup)
python run_optimized_experiments.py --test
```

### **After Experiments Complete**

```bash
# Check if all result files exist
ls -lh results/reviewer_experiments/oasis/
ls -lh results/reviewer_experiments/alzheimer/

# Verify JSON files are valid
python -c "import json; json.load(open('results/reviewer_experiments/oasis/OPTIMIZED_COMPLETE_*.json'))"

# Count total experiments run
grep -c "SUCCESS" revision_quick.log
```

---

## ⚙️ Configuration Details

### **Attack Severity (Already Configured)**

```python
ATTACK_SEVERITY_CONFIGS = {
    'revision': {
        'malicious_ratio': 0.4,       # 40% malicious clients
        'scaling_factor': 30.0,        # 30× gradient scaling
        'noise_factor': 15.0,          # 15× noise magnitude
        'flip_probability': 0.9        # 90% label flipping
    }
}
```

This ensures **strong attacks** for clear baseline differentiation.

### **Seeds Configuration**

```python
REVISION_QUICK_SEEDS = [42, 123, 456, 789, 1024]  # n=5 seeds
```

Meets reviewer requirements for statistical robustness.

### **Dataset Paths**

OASIS:
```python
config.OASIS_DATA_ROOT = "oasis_cross-sectional_disc1/disc1"
config.OASIS_DEMOGRAPHICS = "oasis_cross-sectional.xlsx"
```

ALZHEIMER:
```python
config.ALZHEIMER_DATA_DIR = "data/alzheimer"
config.ALZHEIMER_DATA_ROOT = "data/alzheimer"
```

**If your paths differ**, update in `federated_learning/config/config.py` before running.

---

## ❓ Troubleshooting

### **Problem: "ImportError: No module named federated_learning"**

**Solution:**
```bash
# Ensure you're in project root
cd /home/gpu/FLBrain/OptiGradTrust-3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_optimized_experiments.py --revision-quick --epochs 8
```

### **Problem: "CUDA out of memory"**

**Solution:**
```bash
# Reduce batch size in config.py
# BATCH_SIZE = 32  # Default 64
# Or use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
```

### **Problem: "FileNotFoundError: oasis_cross-sectional.xlsx"**

**Solution:**
```bash
# Verify file exists
ls -lh oasis_cross-sectional.xlsx

# If missing, ensure it's not in .gitignore
git check-ignore oasis_cross-sectional.xlsx

# If ignored, download manually from server or local backup
```

### **Problem: Process stopped unexpectedly**

**Solution:**
```bash
# Check last error in log
tail -50 revision_quick.log

# Resume from checkpoint (if supported)
python run_optimized_experiments.py --revision-quick --epochs 8 --resume
```

---

## 📝 Summary Checklist

Before running experiments, verify:

- ✅ Latest code pulled from GitHub
- ✅ Python environment activated
- ✅ All critical files exist (run_optimized_experiments.py, run_all_experiments.py, etc.)
- ✅ OASIS and ALZHEIMER datasets accessible
- ✅ GPU available and working
- ✅ Old logs cleaned (optional)

**Run commands:**

1. **OASIS**: `nohup python run_optimized_experiments.py --revision-quick --epochs 8 > revision_quick.log 2>&1 &`
2. **Wait** for OASIS to complete (~18-24 hours)
3. **ALZHEIMER**: `nohup python run_optimized_experiments.py --revision-quick --revision-dataset ALZHEIMER --epochs 8 --phase 1 > revision_quick_ALZHEIMER.log 2>&1 &`
4. **Monitor**: `tail -f revision_quick.log` and `tail -f revision_quick_ALZHEIMER.log`
5. **Results**: Check `results/reviewer_experiments/oasis/` and `results/reviewer_experiments/alzheimer/`

---

## 📧 Contact

If you encounter issues:
- Check logs carefully: `tail -100 revision_quick.log`
- Verify GPU: `nvidia-smi`
- Test imports: `python -c "import federated_learning"`

Good luck with your experiments! 🚀
