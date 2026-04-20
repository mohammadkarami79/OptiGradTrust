# OptiGradTrust — R3 Implementation Plan
**Prepared by:** Implementation Team
**Audience:** Writing Team + Ali (operator)
**Date:** April 2026
**Target:** Address all experimental asks from `R3_IMPLEMENTATION_TEAM_REPORT.md` in a single coherent runner.

---

## 1. Mapping reviewer asks → concrete experiments

| Reviewer ask (R3 report) | Experiment ID | What we run | New code needed |
|---|---|---|---|
| Comment 3 — attack budget failure-mode sweep | **R3-A** | Alzheimer, 25 rounds, 3 seeds, scaling ∈ {×1,×10,×50,×100} + sign-flip with malicious-fraction ∈ {10%,20%,40%,60%} | None — reuses existing scaling/sign-flipping attacks + existing server |
| Comment 4 — computational / communication overhead | **R3-B** | 5 methods (FedAvg, FedBN, Krum, FLTrust, OptiGradTrust), Alzheimer, 25 rounds, single seed, 30 % scaling | Krum + FLTrust aggregators, per-round timing hooks, comm-volume accounting |
| Comment 7 — multi-seed headline table | **R3-C** | 2 distributions (IID + α=0.1) × 3 attacks (scaling ×20, sign-flip, Gaussian σ=15) × 4 methods (FedAvg, FedBN, FLTrust, OptiGradTrust) × 3 seeds = **72 runs**. Paired stats (Wilcoxon) + 95 % CI | Need FLTrust baseline |
| Comment 2 — adaptive attacker | **R3-D** | VAE-aware adaptive attack vs. static scaling ×20, Alzheimer, 3 seeds | New attack type `adaptive_vae_attack` |
| Comment 5c — RL vs rule-based ablation | **R3-E** | 3 configs: full OptiGradTrust (RL) / rule-based-trust / no-RL (from R2), dynamic attack, 3 seeds | Rule-based trust mode (equal-weight fingerprint sum) |
| Comment 6 — 2 extra baselines | **R3-F** | RFA (geometric median) + SignGuard at 30 % scaling, Alzheimer, 3 seeds | RFA + SignGuard aggregators |

Total: **6 core experiments, ~60–75 GPU-hours** (matches the writing-team budget).

---

## 2. Architecture changes

### 2.1 New module: `federated_learning/aggregators/byzantine_baselines.py`
Pure-function PyTorch implementations of:
- `krum_aggregate(gradients, num_byzantine)` — Blanchard et al. 2017
- `fltrust_aggregate(gradients, root_gradient)` — Cao et al. 2021
- `rfa_aggregate(gradients, num_iters=5, eps=1e-6)` — Pillutla et al. 2019 (Weiszfeld iteration for geometric median)
- `signguard_aggregate(gradients)` — Xu et al. 2022

Each returns `(aggregated_gradient, detected_indices_in_batch)` so the server can populate detection metrics uniformly.

### 2.2 `federated_learning/training/server.py` — surgical additions
1. **Register baselines in `pure_baselines` dispatch.** The function `configure_for_dataset` in `run_all_experiments.py` already recognises `krum`, `fltrust`, `median`, `trimmed_mean`, `trfa`, `fedbn`, `fedprox`, `fedavg`. I add `rfa` and `signguard` to that list so VAE/Shapley/DualAttention are auto-disabled.
2. **New method `_aggregate_byzantine_baseline(gradients, client_indices, method)`** that dispatches to the four new aggregators and writes detection metrics into `round_metrics[round_idx]['detection_results']`.
3. **New dispatch branches** in the main `train()` aggregation block (around line 994) for `krum / fltrust / rfa / signguard`.
4. **Optional `USE_RULE_BASED_TRUST` flag** (default False). When True, after features are computed but before dual-attention runs, trust = mean of 6 fingerprint features; dual-attention is bypassed.
5. **Optional `MEASURE_OVERHEAD` flag** (default False). When True, server logs `time.perf_counter()` around aggregation and records per-round metrics (`agg_wall_time_s`, `comm_volume_mb`, `peak_mem_mb`) into `round_metrics`.

### 2.3 `federated_learning/attacks/attack_utils.py` + `federated_learning/training/client.py`
Add **`adaptive_vae_attack`**: client receives a reference to the server VAE (snapshot before the round); the attacker solves a tiny PGD-style inner loop to minimise VAE reconstruction error while keeping a malicious direction (scaled fraction of the pre-attack gradient). 20 inner steps, step size 0.05, ε-ball = 5 × ‖g‖.

Fallback: if VAE is not passed, it degrades gracefully to cosine-similarity-aware scaling (minimises |1 − cos(g', ĝ_ref)| while preserving malicious magnitude).

### 2.4 New runner: `run_r3_experiments.py`
One file, same style as `run_r2_experiments.py`. Subcommands:
```
--experiment r3a      # attack budget sweep
--experiment r3b      # overhead comparison
--experiment r3c      # multi-seed headline table
--experiment r3d      # adaptive attack
--experiment r3e      # RL vs rule-based
--experiment r3f      # RFA + SignGuard baselines
--experiment all
--seeds 42 123 456
--dry-run             # 2-round smoke test per config
```
Outputs go to `results/r3_revision/<experiment>/`: CSV + JSON per run, summary table, PNG plots.

---

## 3. Files to change or create on the server

| # | Path | Action | New LOC (approx) |
|---|---|---|---|
| 1 | `federated_learning/aggregators/__init__.py` | **create** | 3 |
| 2 | `federated_learning/aggregators/byzantine_baselines.py` | **create** | ~200 |
| 3 | `federated_learning/attacks/attack_utils.py` | **modify** (add adaptive attack) | +40 |
| 4 | `federated_learning/training/client.py` | **modify** (adaptive-attack hook + VAE snapshot) | +50 |
| 5 | `federated_learning/training/server.py` | **modify** (baseline dispatch + rule-based trust + overhead timing) | +120 |
| 6 | `run_all_experiments.py` | **modify** (add `rfa`, `signguard`, `fedadmm` to `pure_baselines` list) | +1 |
| 7 | `federated_learning/config/config.py` | **modify** (add `USE_RULE_BASED_TRUST=False`, `MEASURE_OVERHEAD=False` defaults so they propagate via `_sync_config_to_modules`) | +10 |
| 8 | `run_r3_experiments.py` | **create** | ~900 |
| 9 | `R3_IMPLEMENTATION_PLAN.md` | **create** (this file, documentation) | — |

Files to leave alone: `run_r2_experiments.py`, `run_revision_experiments.py`, `run_oasis_experiments.py`, everything else.

---

## 4. Run order on the server (use `nohup -u` and `tail -f`)

All commands assume `conda activate optigrad_py311` and `cd ~/FLBrain/OptiGradTrust-3`. Logs go to `logs/r3_*.log`.

```bash
mkdir -p logs

# 0) Quick smoke test (2 rounds) — confirms the code loads + trains ~10 min
nohup python -u run_r3_experiments.py --experiment r3a --dry-run \
    > logs/r3_dryrun.log 2>&1 &
tail -f logs/r3_dryrun.log
# (wait for "DRY-RUN COMPLETE")

# 1) R3-F — RFA + SignGuard (shortest, validates new baselines work) ~8 h
nohup python -u run_r3_experiments.py --experiment r3f --seeds 42 123 456 \
    > logs/r3f.log 2>&1 &
tail -f logs/r3f.log

# 2) R3-E — RL vs rule-based ablation ~6 h
nohup python -u run_r3_experiments.py --experiment r3e --seeds 42 123 456 \
    > logs/r3e.log 2>&1 &
tail -f logs/r3e.log

# 3) R3-B — overhead comparison (single seed) ~10 h
nohup python -u run_r3_experiments.py --experiment r3b --seeds 42 \
    > logs/r3b.log 2>&1 &
tail -f logs/r3b.log

# 4) R3-D — adaptive attack ~9 h
nohup python -u run_r3_experiments.py --experiment r3d --seeds 42 123 456 \
    > logs/r3d.log 2>&1 &
tail -f logs/r3d.log

# 5) R3-A — attack budget sweep ~9 h
nohup python -u run_r3_experiments.py --experiment r3a --seeds 42 123 456 \
    > logs/r3a.log 2>&1 &
tail -f logs/r3a.log

# 6) R3-C — multi-seed headline table (longest) ~22 h
nohup python -u run_r3_experiments.py --experiment r3c --seeds 42 123 456 \
    > logs/r3c.log 2>&1 &
tail -f logs/r3c.log
```

### Why this order?
1. **F first** — smallest new-code surface; if RFA/SignGuard compile and finish, every other experiment will.
2. **E second** — reuses only the rule-based-trust flag (trivial).
3. **B third** — reuses Krum/FLTrust/RFA just proven in F/E.
4. **D fourth** — adaptive attack is the riskiest new component; once B is done we have GPU time freed.
5. **A fifth** — ~8 h, uses only existing attacks.
6. **C last** — 72 runs, longest; runs with maximum confidence that everything else works.

If you only have **one GPU**, run them one at a time using the order above.
If you have **two GPUs**, set `CUDA_VISIBLE_DEVICES=0` for one group (F + E + D) and `CUDA_VISIBLE_DEVICES=1` for another (A + B + C) to halve wall-clock.

---

## 5. Git workflow

You do the git push yourself. I only edit files; I do not push.

```bash
# On your local machine where Cursor edited the files:
cd d:/OptiGradTrust-3
git status
git add federated_learning/aggregators/ \
        federated_learning/attacks/attack_utils.py \
        federated_learning/training/client.py \
        federated_learning/training/server.py \
        run_all_experiments.py \
        run_r3_experiments.py \
        R3_IMPLEMENTATION_PLAN.md

git commit -m "R3: add Krum/FLTrust/RFA/SignGuard baselines, VAE-aware adaptive attack, rule-based trust mode, overhead instrumentation, and run_r3_experiments.py"

git push origin main       # or whatever branch you use
```

Then on the server:
```bash
cd ~/FLBrain/OptiGradTrust-3
git pull
```

---

## 6. Answers to questions from the Writing Team

1. **Does Alzheimer MRI have subject metadata?** The ADNI-derived Kaggle set we use has class labels only (NonDemented / VeryMild / Mild / Moderate); no age, site, or scanner metadata. → **Honest answer: no subgroup analysis possible**. Writing team should decline Comment 1b for Alzheimer as well (consistent with OASIS).
2. **Are FLTrust / Krum still functional?** No — `configure_for_dataset` lists them in `pure_baselines` but there is **no implementation** in `server.py` (they fall through to `fedavg`). R3-F and R3-B fix this with a real implementation (`byzantine_baselines.py`).
3. **Adaptive-attack blockers?** Main one: the VAE snapshot must be passed from server → client each round. I handle this by stashing a `torch.no_grad()` cloned copy of `server.vae` into `client.vae_snapshot` inside `Server.train()` right before `client.train()` is called (one small hook). No memory blow-up — VAE is ~2 MB.
4. **60–75 h in 2–3 weeks?** Yes, comfortably, on 1× RTX GPU. Worst-case single-GPU wall-clock ≈ 65 h = ~3 days of actual GPU time. Two parallel GPUs finishes in ~1.5 days.

---

## 7. Honesty notes for the Writing Team (pre-results)

- **RFA has no per-client detection mechanism.** Report F1 as `N/A` with a one-line explanation — do not fabricate a score. Same for pure Krum (its "detection" is trivial: N−1 clients are "rejected" each round; F1 is dominated by false positives; report and discuss rather than compare).
- **FLTrust detection** is derived from its ReLU-cos trust scores — threshold at 0 gives a binary detection, reported.
- **Adaptive attack expected to reduce accuracy.** This is the published consensus for any adaptive attack against a static defence; we report it transparently. The narrative is: "under *one* adaptive variant, OptiGradTrust degrades by X % but remains superior to Y" (fill in after results).
- **Rule-based vs RL (R3-E):** expect similar mean accuracy but higher std for the rule-based variant. Narrative: RL gives operational consistency, not peak accuracy — same framing as R2.
- **Overhead (R3-B):** OptiGradTrust will be the slowest. Expected ~3–5× FedAvg per round, dominated by Shapley. This matches our R2 disclosure.

---

## 8. Deliverables after completion

One Markdown report (`R3_REVISION_REPORT.md`) with:
- Per-experiment raw CSVs + JSONs
- Summary tables ready for the paper
- PNG figures (300 DPI)
- Response-letter paragraphs for each reviewer comment
- Honest limitations section

This report will be authored after the server run finishes and results are sent back.
