# OptiGradTrust — Third Revision (R3) Implementation Team Report (LEAN VERSION)
## Prepared by the Writing Team — For the Implementation Team

**Paper:** Byzantine robust federated learning for heterogeneous brain MRI using multisignal gradient fingerprinting and adaptive trust aggregation
**Journal:** Scientific Reports
**Round:** 3rd revision (Reviewer 2 recommends acceptance; Reviewer 1 major revision)
**Target timeline:** 2-3 weeks for core experiments; writing team can then integrate results.

---

## Context and Strategy

**Reviewer 2** recommends acceptance with no further changes.
**Reviewer 1** has raised 8 major and 7 minor comments. The editor noted four priority areas: clinical validation, threat model specification, computational overhead, and statistical reporting.

**Our strategic approach (continuing from R1 and R2):**
- **Less experiments + strong writing + honest explanations.** This worked in R1 and R2 — we will continue.
- Run only **focused, well-executed experiments** that give the highest reviewer-impact-per-GPU-hour ratio.
- Many reviewer comments will be addressed through **text improvements only** (no experiments needed). The writing team will handle all text-only fixes.
- If time permits after core experiments, we can add complementary runs — but core experiments alone are sufficient for acceptance.

---

## Division of Work: Experiments vs Text

### Text-only fixes (handled by Writing Team, no experiments needed):
- **Comment 1c** — Tone down clinical-deployment language throughout
- **Comment 2 (first half)** — Formalize threat model (knowledge, collusion, adaptivity assumptions)
- **Comment 5a** — MDP specification (state, action, reward, horizon) for RL controller
- **Comment 8** — Privacy discussion expansion (compatibility with secure aggregation, DP)
- **All 7 Minor Comments** — title typos, FedBN-P definition, notation table, figure error-bar captions, reproducibility commitment, LaTeX abstract macros, related work positioning

### Experiments required (implementation team):
- **R3-A** — Attack budget failure-mode sweep (Comment 3)
- **R3-B** — Overhead comparison (Comment 4)
- **R3-C** — Multi-seed re-run of headline table (Comment 7)
- **R3-D** — One adaptive attacker experiment (Comment 2, second half)
- **R3-E** — RL vs rule-based ablation (Comment 5c)
- **R3-F** — Two additional baselines (Comment 6)

---

## Core Experiments (Target: ~60-75 GPU-hours total)

### R3-A: Attack Budget Failure-Mode Sweep (Comment 3)
**Purpose:** Characterize when OptiGradTrust fails under increasing attack intensity.

**Scope:**
- Dataset: Alzheimer's MRI only
- Attack types: **2 types** — scaling (magnitude) + sign-flipping (magnitude-preserving)
- Intensity levels: 4-5 levels per attack type
  - Scaling: {×1, ×10, ×50, ×100}
  - Sign-flipping: malicious fraction {10%, 20%, 40%, 60%}
- Seeds: 3 per configuration
- Rounds: 25

**Deliverables:**
- CSV with per-configuration accuracy, detection F1 (mean ± std over 3 seeds)
- One plot per attack type: accuracy vs intensity with error bars
- Identified "failure threshold" (intensity at which accuracy drops meaningfully)

**Estimated time:** ~8-10 GPU-hours

---

### R3-B: Computational and Communication Overhead Comparison (Comment 4)
**Purpose:** Fair cost comparison with key baselines.

**Baselines (5 total — focused set):**
1. FedAvg
2. FedBN
3. Krum
4. FLTrust
5. OptiGradTrust (full)

**Note:** We do NOT need Median, Trimmed Mean, and Multi-Krum for this table. Their costs are well-documented in the literature and similar to Krum; we will cite this honestly in the paper.

**Metrics:**
- Wall-clock time per round (client + server separately)
- Server-side FLOPs per round (use `torch.profiler` or `thop`)
- Peak memory usage (server)
- Communication volume per round (MB)

**Setup:**
- Alzheimer's MRI, 10 clients, 25 rounds, IID, 30% scaling attack
- Single seed (timing is deterministic enough)

**Deliverables:**
- CSV with per-baseline per-metric values
- Summary table ready for the paper
- Bar chart: time-per-round across baselines (log scale)

**Estimated time:** ~10 GPU-hours

---

### R3-C: Multi-Seed Re-Run of Headline Accuracy Table (Comment 7)
**Purpose:** Convert the main result from single-seed to mean ± std with statistical tests.

**Scope (FOCUSED):**
- **1 dataset:** Alzheimer's MRI
- **2 distributions:** IID + 1 Non-IID (Dirichlet α=0.1)
- **3 attack types:** Scaling ×20, Sign-flipping, Gaussian noise (σ=15.0)
- **Baselines:** FedAvg, FedBN, FLTrust, OptiGradTrust (4 methods)
- **Seeds:** 3 per configuration (seed=42, 123, 456)

**This gives:** 2 distributions × 3 attack types × 4 methods × 3 seeds = **72 runs**

**Statistical analysis:**
- Paired t-test OR Wilcoxon signed-rank test: OptiGradTrust vs each baseline on matching seed triples
- Report p-values and effect sizes (Cohen's d)
- Explicitly state: "No multiple-comparison correction applied — tests are exploratory"
- 95% confidence intervals for F1 scores

**NOT re-running (documented honestly in paper):**
- R2 ablations (noise, sign-flipping) remain single-seed — disclose explicitly
- OASIS results remain as-is (already small-sample, disclosed)

**Deliverables:**
- Updated headline table with mean ± std
- Separate CSV of p-values and effect sizes for each pairwise comparison
- 95% CI for headline F1 numbers

**Estimated time:** ~20-25 GPU-hours

---

### R3-D: One Adaptive Attacker Experiment (Comment 2)
**Purpose:** Stress-test the defense against a targeted adaptive attacker.

**Single attack to implement — the most relevant to our claims:**
- **VAE-aware adaptive attack:** Malicious client estimates the VAE reconstruction error contribution and crafts gradients with low VAE error (e.g., by slightly perturbing a legitimate gradient rather than doing pure scaling). This directly tests our strongest claim — that VAE fingerprinting defends against noise-based manipulation.

**If simpler to implement:** A black-box adaptive attacker that runs a local optimization to minimize one of: cosine-similarity-to-reference OR gradient-norm deviation, while still injecting malicious direction.

**Setup:**
- Alzheimer's MRI, 10 clients, 30% malicious, 25 rounds, 3 seeds
- Compare OptiGradTrust accuracy + detection against adaptive attack vs static scaling ×20

**Honest framing:** This is a "stress-test with one adaptive variant," not a comprehensive adaptive evaluation. If the defense degrades, we report transparently — this is expected for adaptive attacks in the literature.

**Estimated time:** ~8-10 GPU-hours

---

### R3-E: RL vs Rule-Based Aggregator Ablation (Comment 5c)
**Purpose:** Justify the RL component against a simpler baseline.

**Configurations (3 total):**
1. **Full OptiGradTrust** (with RL)
2. **Rule-based variant:** Replace RL with fixed equal-weight linear combination of fingerprints (w_i = 1/6 for all 6 signals)
3. **No-RL baseline** (already have from R2 — reuse)

**Setup:**
- Dynamic attack schedule (same as R2 Exp 2), 3 seeds, 25 rounds
- Report accuracy + detection F1 + F1 standard deviation for each

**Expected outcome (honest prediction):** Rule-based and RL may give similar mean accuracy. Our argument becomes: "RL provides operational stability/consistency, not accuracy gains — and this is the value-add over simpler approaches." This matches our R2 framing.

**Do NOT tune the rule-based variant aggressively.** Use out-of-the-box equal weights. We want a fair comparison.

**Estimated time:** ~5-6 GPU-hours

---

### R3-F: Two Additional Recent Baselines (Comment 6)
**Purpose:** Extend baseline comparison beyond classical methods.

**Exactly 2 baselines:**
1. **RFA (Robust Federated Aggregation / Geometric Median)** — simple to implement, well-documented
2. **SignGuard** — has publicly available code; sign-based defense

**Skip these** (honestly justified in paper):
- DnC — we will cite and justify exclusion as in R2
- 2024-2025 medical FL methods — mention in related work, note lack of standardized baselines as a limitation

**Setup:**
- Alzheimer's MRI, 10 clients, 30% scaling attack, 25 rounds, 3 seeds
- Include in the headline table from R3-C

**Deliverables:**
- Accuracy + detection F1 for RFA and SignGuard
- Honest reporting — even if they match OptiGradTrust on some metrics

**Estimated time:** ~8-10 GPU-hours

---

## Summary Table: GPU Budget

| Experiment | Estimated Hours | Priority |
|---|---|---|
| R3-A: Attack budget sweep | 8-10 | Core |
| R3-B: Overhead comparison | 10 | Core |
| R3-C: Multi-seed headline table | 20-25 | Core |
| R3-D: Adaptive attack (one variant) | 8-10 | Core |
| R3-E: RL vs rule-based ablation | 5-6 | Core |
| R3-F: Two extra baselines | 8-10 | Core |
| **Total (Core)** | **~60-75 hours** | |

If you have spare GPU-time after completing all six core experiments, optional additions:
- Second adaptive attack variant (Shapley-aware): +8h
- Multi-seed re-run of R2 ablations: +15h
- Third extra baseline: +5h

But **core experiments alone are sufficient** — do not over-extend if timeline is tight.

---

## What We CANNOT Run (Writing Team Will Handle Textually)

### Clinical validation on ADNI/UK Biobank/BraTS (Comment 1a)
**Status:** DECLINED. Data access and IRB approval take months to years.
**Writing team mitigation:** Significantly tone down clinical claims. Explicitly frame OASIS as proof-of-concept only. Add strong "Future work" statement committing to multi-centre validation.

### Per-site/subgroup analysis on OASIS (Comment 1b)
**Status:** DECLINED for OASIS (too small, ~87 samples, near-zero statistical power).
**Question for implementation team:** Does the Alzheimer's MRI dataset have ANY metadata (age, scanner, source) we could use for even a coarse subgroup analysis? If yes, a quick aggregate-level report would help. If not, we decline honestly.

---

## Deliverables Format (Please follow, matches R2)

For each experiment:
1. **Raw CSV files** with per-seed per-configuration values
2. **JSON files** with full structured data (hyperparameters, runtime, hardware)
3. **PNG figures** at 300 DPI where applicable
4. **Summary Markdown report** (like R2_REVISION_REPORT.md) with:
   - What was run and why
   - Key findings in plain language
   - Any surprises, limitations, or failed configurations
   - Honest notes for the writing team

Folder structure: `r3_revision/` with one subfolder per experiment.

---

## Honesty Notes (Same as R2)

1. **Do not hide failed experiments.** If a baseline fails to converge or an adaptive attack breaks our defense, report it — we will frame honestly.
2. **Do not tune our method favourably vs baselines.** Use out-of-the-box hyperparameters.
3. **Report wall-clock time honestly** including Shapley overhead.
4. **If something is infeasible,** say so clearly. The writing team handles textual accommodation.

---

## Reproducibility Commitment (Comment 6 / Minor Comments)

Please also prepare (as time permits — not blocking):
- Clean codebase with documented hyperparameters
- Trained VAE checkpoints for Alzheimer's MRI
- Exact data splits (pickle or numpy) for reported experiments
- environment.yml or Dockerfile for reproducibility

We will commit to releasing this upon acceptance.

---

## Questions for the Implementation Team

Please answer before starting so we can adjust the plan if needed:

1. Does the Alzheimer's MRI dataset have ANY metadata (age, scanner, source) we can use for a simple subgroup analysis?
2. Are FLTrust and Krum baselines still functional in the current codebase?
3. Any technical blockers you foresee in R3-D (VAE-aware adaptive attack)?
4. Can you complete ~60-75 GPU-hours within 2-3 weeks? If not, tell us which experiment to cut first.

---

## Final Note

Thank you for your excellent work in R1 and R2. This round is more focused than the original plan: **6 targeted experiments (~60-75 GPU-hours)** instead of sprawling comprehensive evaluation. The strategy is the same as R1 and R2 — honest, focused results + strong writing + transparent limitations.

With this lean plan plus thorough text improvements by the writing team, we will address every reviewer comment. If time permits after core experiments, we can add complementary runs. But core experiments alone are sufficient.

Please confirm receipt and estimated completion timeline. The writing team is on standby to start text-only improvements immediately (while you run experiments in parallel).

— Writing Team, OptiGradTrust
