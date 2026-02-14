# Paper revision results (OASIS & ALZHEIMER)

This folder contains the **reproducible results** for the OptiGradTrust paper revision, used for reviewer verification.

## Layout

| Path | Description |
|------|-------------|
| **`oasis/`** | Results for **OASIS** (real clinical MRI). Full suite: Phase 1 (clinical), Phase 2 (scalability), Phase 3 (baselines), Phase 5 (ablation), Phase 7 (extreme imbalance). |
| **`alzheimer/`** | Results for **ALZHEIMER** (Kaggle medical MRI). Phase 1 only (clinical validation), same protocol as OASIS for comparability. |

## Result files

- **`OPTIMIZED_COMPLETE_<timestamp>.json`** (in `oasis/`): Combined OASIS results (all phases), with per-scenario statistics.
- **`OPTIMIZED_ALZHEIMER_<timestamp>.json`** (in `alzheimer/`): ALZHEIMER Phase 1 results.
- Phase-specific JSONs (e.g. `*_results_*.json`, `ablation_*.json`, `baseline_significance_*.json`, `scalability_results_*.json`, `extreme_imbalance_*.json`) when present.

## JSON format

Each scenario key (e.g. `OASIS_IID_scaling_attack`, `ALZHEIMER_Dirichlet_0.5_noise_attack`) has:

```json
{
  "mean": 0.7234,
  "std": 0.0156,
  "ci_95_lower": 0.7078,
  "ci_95_upper": 0.7390,
  "min": 0.7012,
  "max": 0.7421,
  "n": 5
}
```

- **n**: Number of random seeds (5 for revision).
- **mean ± std**: Accuracy over seeds.
- **ci_95_***: 95% confidence interval.

## How to reproduce

From the repository root:

1. **OASIS (full suite)**  
   ```bash
   python run_optimized_experiments.py --revision-quick --epochs 8
   ```
   Output is written under `results/reviewer_experiments/oasis/`.

2. **ALZHEIMER (Phase 1 only)**  
   ```bash
   python run_optimized_experiments.py --revision-quick --revision-dataset ALZHEIMER --epochs 8 --phase 1
   ```
   Output is written under `results/reviewer_experiments/alzheimer/`.

Configuration: **seeds** = [42, 123, 456, 789, 1024], **attack** = 40% malicious, 30× scaling, 15× noise, 90% label flip. See the main [README](../../README.md) and `run_optimized_experiments.py` for details.
