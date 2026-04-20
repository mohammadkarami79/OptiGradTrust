#!/usr/bin/env python3
"""
=============================================================================
OptiGradTrust -- Second Revision (R2) Experiment Runner
=============================================================================

Implements ALL experiments required by Reviewer 1 in the second revision:

  Experiment 1A: Multi-attack ablation under GAUSSIAN NOISE INJECTION
                 (validates VAE component)
  Experiment 1B: Multi-attack ablation under SIGN-FLIPPING
                 (validates cosine-similarity / sign-consistency)
  Experiment 2:  Dynamic attack schedule (RL justification) -- CRITICAL
                 Phase transition: rounds 1-12 benign, 13-25 scaling x20
  Experiment 3:  Trust score / Shapley visualization (qualitative figure)
  Experiment 4:  Optimizer comparison under adversarial conditions

USAGE:
    python run_r2_experiments.py --experiment exp1a          # Noise ablation
    python run_r2_experiments.py --experiment exp1b          # Sign-flip ablation
    python run_r2_experiments.py --experiment exp2           # Dynamic attack (CRITICAL)
    python run_r2_experiments.py --experiment exp3           # Trust visualization
    python run_r2_experiments.py --experiment exp4           # Optimizer adversarial
    python run_r2_experiments.py --experiment all            # Run all
    python run_r2_experiments.py --experiment exp2 --dry-run # Sanity check

    # Multi-seed options (use for critical experiments):
    python run_r2_experiments.py --experiment exp2 --seeds 42 123 456
    python run_r2_experiments.py --experiment exp1a --seeds 42 123 456 789 1024

Author: OptiGradTrust Team
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import time
import copy
import csv
import argparse
import traceback
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from run_all_experiments import (
    set_all_seeds, configure_for_dataset,
    compute_statistics, Logger, _sync_config_to_modules
)

# ---------------------------------------------------------------------------
# Output directory & logger
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'r2_revision')
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    RESULTS_DIR,
    f'r2_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
)
logger = Logger(LOG_FILE)


# ===========================================================================
# SHARED ABLATION CONFIGURATIONS
# ===========================================================================

# The five standard ablation configurations (same structure as Table 9)
ABLATION_CONFIGS = [
    {'name': 'full',          'vae': True,  'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_vae',        'vae': False, 'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_shapley',    'vae': True,  'shapley': False, 'dual_attention': True,  'rl': True},
    {'name': 'no_rl',         'vae': True,  'shapley': True,  'dual_attention': True,  'rl': False},
    {'name': 'fedavg_no_def', 'vae': False, 'shapley': False, 'dual_attention': False, 'rl': False,
     '_fedavg': True},
]

# Extra sign-flipping specific configs (optional if time allows)
SIGNFLIP_EXTRA_CONFIGS = [
    {'name': 'no_cosine_sim',   'vae': True, 'shapley': True, 'dual_attention': True, 'rl': True,
     '_zero_features': [1, 2]},   # zero root-cosine (1) and peer-consensus (2)
    {'name': 'no_sign_consist', 'vae': True, 'shapley': True, 'dual_attention': True, 'rl': True,
     '_zero_features': [4]},       # zero sign-consistency (4)
]


# ===========================================================================
# FEATURE-ZEROING PATCH  (same technique as run_revision_experiments.py)
# ===========================================================================

@contextmanager
def _patch_feature_zeroing(zero_indices):
    """Monkey-patch Server._compute_all_gradient_features to zero specific feature columns."""
    from federated_learning.training.server import Server
    original_fn = Server._compute_all_gradient_features

    def _patched(self, client_gradients):
        features = original_fn(self, client_gradients)
        if isinstance(features, torch.Tensor):
            for idx in zero_indices:
                if idx < features.size(1):
                    features[:, idx] = 0.5   # neutral / uninformative
        return features

    Server._compute_all_gradient_features = _patched
    try:
        yield
    finally:
        Server._compute_all_gradient_features = original_fn


# ===========================================================================
# LOW-LEVEL EXPERIMENT RUNNER  (shared by Exp 1A, 1B)
# ===========================================================================

def _apply_alzheimer_ablation_config(cfg_mod, ablation_cfg, num_rounds):
    """
    Set Alzheimer ablation config parameters exactly as described in the R2 report:
    - 10 clients, 40% malicious, Dirichlet alpha=0.1, 25 rounds
    - Batch=16, LR=1e-4, weight_decay=5e-5, FedBN-P (mu=0.01)
    """
    cfg_mod.GLOBAL_EPOCHS  = num_rounds
    cfg_mod.FRACTION_MALICIOUS = 0.4
    cfg_mod.NUM_MALICIOUS  = 4
    cfg_mod.BATCH_SIZE     = 16
    cfg_mod.LR             = 1e-4
    cfg_mod.LEARNING_RATE  = 1e-4
    cfg_mod.WEIGHT_DECAY   = 5e-5
    cfg_mod.FEDPROX_MU     = 0.01

    is_fedavg_no_def = ablation_cfg.get('_fedavg', False)

    if is_fedavg_no_def:
        cfg_mod.ENABLE_VAE            = False
        cfg_mod.ENABLE_SHAPLEY        = False
        cfg_mod.ENABLE_DUAL_ATTENTION = False
        cfg_mod.RL_AGGREGATION_METHOD = 'dual_attention'
        cfg_mod.RL_WARMUP_ROUNDS      = 9999
        cfg_mod.AGGREGATION_METHOD    = 'fedavg'
        cfg_mod.GRADIENT_COMBINATION_METHOD = 'fedavg'
    else:
        cfg_mod.ENABLE_VAE            = ablation_cfg.get('vae', True)
        cfg_mod.ENABLE_SHAPLEY        = ablation_cfg.get('shapley', True)
        cfg_mod.ENABLE_DUAL_ATTENTION = ablation_cfg.get('dual_attention', True)
        if not ablation_cfg.get('rl', True):
            cfg_mod.RL_AGGREGATION_METHOD = 'dual_attention'
            cfg_mod.RL_WARMUP_ROUNDS      = 9999
        else:
            cfg_mod.RL_AGGREGATION_METHOD = 'hybrid'
            cfg_mod.RL_WARMUP_ROUNDS      = 5
            cfg_mod.RL_RAMP_UP_ROUNDS     = 10
        cfg_mod.AGGREGATION_METHOD    = 'fedbn_fedprox'
        cfg_mod.GRADIENT_COMBINATION_METHOD = 'fedbn_fedprox'


def _run_ablation_single(attack_type, ablation_cfg, seed, num_rounds, sigma=15.0):
    """
    Run one ablation configuration for a given attack type and seed.
    Returns a result dict.
    """
    import federated_learning.config.config as cfg_mod
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds

    cfg_name = ablation_cfg.get('name', 'unknown')
    logger.info(f"    [{cfg_name}] seed={seed} attack={attack_type}")

    set_all_seeds(seed)

    configure_for_dataset(
        'ALZHEIMER', num_clients=10,
        non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
        aggregation_method='fedbn_fedprox'
    )
    cfg_mod.RANDOM_SEED = seed
    _apply_alzheimer_ablation_config(cfg_mod, ablation_cfg, num_rounds)
    _sync_config_to_modules()

    set_random_seeds(seed)
    start_t = time.time()

    try:
        train_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=True, num_workers=0)

        server = Server()
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()

        _, client_datasets = create_client_datasets(
            train_dataset=train_dataset, num_clients=10, iid=False, alpha=0.1)

        malicious_indices = np.random.choice(10, 4, replace=False)
        clients = []
        for i in range(10):
            is_mal = int(i) in malicious_indices.tolist()
            c = Client(client_id=i, dataset=client_datasets[i], is_malicious=is_mal)
            if is_mal:
                c.set_attack_parameters(
                    attack_type=attack_type,
                    scaling_factor=20.0,
                    sigma=sigma,
                    partial_percent=cfg_mod.PARTIAL_SCALING_PERCENT,
                    noise_factor=cfg_mod.NOISE_FACTOR,
                    flip_probability=cfg_mod.FLIP_PROBABILITY,
                )
            clients.append(c)
        server.add_clients(clients)

        root_gradients = server._collect_root_gradients()
        server.root_gradients = root_gradients   # enables root-cosine & sign-consistency features
        if getattr(cfg_mod, 'ENABLE_VAE', True):
            server.vae = server.train_vae(root_gradients, vae_epochs=cfg_mod.VAE_EPOCHS)
        else:
            server.vae = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=False, num_workers=0)
        server.test_loader = test_loader

        # Run training, applying optional feature-zeroing patch
        zero_feats = ablation_cfg.get('_zero_features')
        if zero_feats:
            with _patch_feature_zeroing(zero_feats):
                _, round_metrics = server.train(num_rounds=cfg_mod.GLOBAL_EPOCHS)
        else:
            _, round_metrics = server.train(num_rounds=cfg_mod.GLOBAL_EPOCHS)

        final_acc = server.evaluate_model()

        # Aggregate detection metrics over all rounds
        total_tp = total_fp = total_fn = total_tn = 0
        for rd in round_metrics.values():
            det = rd.get('detection_results', {})
            total_tp += det.get('true_positives',  0)
            total_fp += det.get('false_positives', 0)
            total_fn += det.get('false_negatives', 0)
            total_tn += det.get('true_negatives',  0)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        result = {
            'config_name':     cfg_name,
            'seed':            seed,
            'attack_type':     attack_type,
            'final_accuracy':  float(final_acc),
            'precision':       float(precision),
            'recall':          float(recall),
            'f1':              float(f1),
            'total_time':      time.time() - start_t,
            'status':          'completed',
        }
        logger.success(
            f"      → Acc={final_acc*100:.2f}%  "
            f"Prec={precision*100:.2f}  Rec={recall*100:.2f}  F1={f1*100:.2f}"
        )
        return result

    except Exception as exc:
        logger.error(f"      → FAILED: {str(exc)[:120]}")
        traceback.print_exc()
        return {
            'config_name': cfg_name,
            'seed':        seed,
            'attack_type': attack_type,
            'status':      'failed',
            'error':       str(exc),
            'total_time':  time.time() - start_t,
        }


def _save_ablation_table(results, filename_base, attack_label):
    """Save CSV + print a formatted text table for ablation results."""
    completed = [r for r in results if r.get('status') == 'completed']

    # CSV
    csv_path = os.path.join(RESULTS_DIR, f'{filename_base}.csv')
    fields = ['config_name', 'seed', 'attack_type',
              'final_accuracy', 'precision', 'recall', 'f1', 'total_time', 'status']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    logger.info(f"  Saved CSV → {csv_path}")

    # JSON
    json_path = os.path.join(RESULTS_DIR, f'{filename_base}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Saved JSON → {json_path}")

    # Pretty table
    header = (f"\n{'='*80}\n"
              f"ABLATION TABLE — {attack_label}\n"
              f"{'Configuration':<26} {'Acc (%)':>9} {'ΔAcc':>7} "
              f"{'Recall (%)':>11} {'F1 (%)':>8}\n"
              f"{'-'*80}")
    logger.info(header)

    # Compute full-system accuracy as baseline for delta
    full_acc = None
    for r in completed:
        if r['config_name'] == 'full':
            full_acc = r['final_accuracy']
            break

    for r in completed:
        acc   = r['final_accuracy'] * 100
        delta = ((r['final_accuracy'] - full_acc) * 100
                 if full_acc is not None and r['config_name'] != 'full' else 0.0)
        delta_str = f"{delta:+.2f}" if r['config_name'] != 'full' else "  --"
        is_fedavg = 'fedavg' in r['config_name']
        recall_str = f"{r['recall']*100:.2f}" if not is_fedavg else "  --"
        f1_str     = f"{r['f1']*100:.2f}"     if not is_fedavg else "  --"
        logger.info(
            f"  {r['config_name']:<24} {acc:>9.2f} {delta_str:>7} "
            f"{recall_str:>11} {f1_str:>8}"
        )

    logger.info('=' * 80)
    return csv_path, json_path


# ===========================================================================
# EXPERIMENT 1A — GAUSSIAN NOISE INJECTION ABLATION
# ===========================================================================

def run_exp1a_noise_ablation(seeds=None, dry_run=False):
    """
    Experiment 1A: Ablation under Gaussian Noise Injection (sigma=15.0).
    Validates the VAE fingerprinting component.
    """
    logger.info('=' * 70)
    logger.info('EXPERIMENT 1A: ABLATION UNDER GAUSSIAN NOISE INJECTION')
    logger.info('  Dataset: Alzheimer MRI | Clients: 10 | Malicious: 40%')
    logger.info('  Attack: gaussian_noise_injection sigma=15.0 | alpha=0.1 | 25 rounds')
    logger.info('=' * 70)

    if seeds is None:
        seeds = [42]
    if dry_run:
        seeds = [42]
    num_rounds = 2 if dry_run else 25

    all_results = []

    configs_to_run = ABLATION_CONFIGS.copy()

    for cfg in configs_to_run:
        logger.info(f"\n  Config: {cfg['name']}")
        seed_results = []
        for seed in seeds:
            r = _run_ablation_single(
                attack_type='gaussian_noise_injection',
                ablation_cfg=cfg,
                seed=seed,
                num_rounds=num_rounds,
                sigma=15.0,
            )
            all_results.append(r)
            if r['status'] == 'completed':
                seed_results.append(r)

        if len(seed_results) > 1:
            accs = [r['final_accuracy'] for r in seed_results]
            logger.info(f"    Mean Acc: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

    _save_ablation_table(all_results, 'exp1a_noise_ablation',
                         'Gaussian Noise Injection (sigma=15.0) | Alzheimer MRI')
    logger.info('\nExperiment 1A complete.')
    return all_results


# ===========================================================================
# EXPERIMENT 1B — SIGN-FLIPPING ABLATION
# ===========================================================================

def run_exp1b_signflip_ablation(seeds=None, dry_run=False, include_optional=True):
    """
    Experiment 1B: Ablation under Sign-Flipping (lambda=-1).
    Validates cosine-similarity and sign-consistency components.
    """
    logger.info('=' * 70)
    logger.info('EXPERIMENT 1B: ABLATION UNDER SIGN-FLIPPING (lambda=-1)')
    logger.info('  Dataset: Alzheimer MRI | Clients: 10 | Malicious: 40%')
    logger.info('  Attack: sign_flipping_attack | alpha=0.1 | 25 rounds')
    logger.info('=' * 70)

    if seeds is None:
        seeds = [42]
    if dry_run:
        seeds = [42]
    num_rounds = 2 if dry_run else 25

    all_results = []
    configs_to_run = ABLATION_CONFIGS.copy()
    if include_optional and not dry_run:
        configs_to_run += SIGNFLIP_EXTRA_CONFIGS

    for cfg in configs_to_run:
        logger.info(f"\n  Config: {cfg['name']}")
        seed_results = []
        for seed in seeds:
            r = _run_ablation_single(
                attack_type='sign_flipping_attack',
                ablation_cfg=cfg,
                seed=seed,
                num_rounds=num_rounds,
            )
            all_results.append(r)
            if r['status'] == 'completed':
                seed_results.append(r)

        if len(seed_results) > 1:
            accs = [r['final_accuracy'] for r in seed_results]
            logger.info(f"    Mean Acc: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

    _save_ablation_table(all_results, 'exp1b_signflip_ablation',
                         'Sign-Flipping (lambda=-1) | Alzheimer MRI')
    logger.info('\nExperiment 1B complete.')
    return all_results


# ===========================================================================
# EXPERIMENT 2 — DYNAMIC ATTACK SCHEDULE (RL JUSTIFICATION)
# ===========================================================================

EXP2_CONFIGS = [
    {'name': 'full',          'vae': True,  'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_rl',         'vae': True,  'shapley': True,  'dual_attention': True,  'rl': False},
    {'name': 'fedavg_no_def', 'vae': False, 'shapley': False, 'dual_attention': False, 'rl': False,
     '_fedavg': True},
]


def _run_exp2_single(cfg, seed, num_rounds, phase_transition_round=12):
    """
    Run one configuration of Experiment 2 (dynamic attack, phase-transition schedule).

    Phase transition:
      rounds 1–12  (round_idx 0–11):  malicious clients behave BENIGNLY
      rounds 13–25 (round_idx 12–24): malicious clients launch scaling x20

    Returns dict with per-round accuracy + detection, and final summary metrics.
    """
    import federated_learning.config.config as cfg_mod
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds

    cfg_name = cfg.get('name', 'unknown')
    logger.info(f"    [Exp2 | {cfg_name}] seed={seed}")

    set_all_seeds(seed)

    configure_for_dataset(
        'ALZHEIMER', num_clients=10,
        non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
        aggregation_method='fedbn_fedprox'
    )
    cfg_mod.RANDOM_SEED = seed
    _apply_alzheimer_ablation_config(cfg_mod, cfg, num_rounds)
    # For Exp 2, use RL in ACTIVE training mode (not frozen)
    # RL must be actively training during all 25 rounds
    if not cfg.get('_fedavg', False) and cfg.get('rl', True):
        cfg_mod.RL_AGGREGATION_METHOD = 'hybrid'
        cfg_mod.RL_WARMUP_ROUNDS      = 5
        cfg_mod.RL_RAMP_UP_ROUNDS     = 10
    _sync_config_to_modules()

    set_random_seeds(seed)
    start_t = time.time()

    try:
        train_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=True, num_workers=0)

        server = Server()
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()

        _, client_datasets = create_client_datasets(
            train_dataset=train_dataset, num_clients=10, iid=False, alpha=0.1)

        malicious_indices = np.random.choice(10, 4, replace=False)
        clients = []
        for i in range(10):
            is_mal = int(i) in malicious_indices.tolist()
            c = Client(client_id=i, dataset=client_datasets[i], is_malicious=is_mal)
            if is_mal:
                c.set_attack_parameters(
                    attack_type='scaling_attack',
                    scaling_factor=20.0,
                    partial_percent=cfg_mod.PARTIAL_SCALING_PERCENT,
                    noise_factor=cfg_mod.NOISE_FACTOR,
                    flip_probability=cfg_mod.FLIP_PROBABILITY,
                )
                # Phase-transition schedule: benign for rounds 0..(T-1), then attack
                c.dynamic_attack_schedule = 'phase_transition'
                c.phase_transition_round  = phase_transition_round   # 0-indexed

            clients.append(c)
        server.add_clients(clients)

        root_gradients = server._collect_root_gradients()
        server.root_gradients = root_gradients   # enables root-cosine & sign-consistency features
        if getattr(cfg_mod, 'ENABLE_VAE', True):
            server.vae = server.train_vae(root_gradients, vae_epochs=cfg_mod.VAE_EPOCHS)
        else:
            server.vae = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=False, num_workers=0)
        server.test_loader = test_loader

        _, round_metrics = server.train(num_rounds=cfg_mod.GLOBAL_EPOCHS)
        final_acc = server.evaluate_model()

        # Build per-round metrics (round_metrics[r]['test_accuracy'] = accuracy
        # measured BEFORE the update in round r+1, i.e. after completing r rounds)
        per_round = {}
        for r_idx, rd in round_metrics.items():
            acc_before_update = rd.get('test_accuracy', 0.0)
            det = rd.get('detection_results', {})
            tp  = det.get('true_positives',  0)
            fp  = det.get('false_positives', 0)
            fn  = det.get('false_negatives', 0)
            recall_r    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision_r = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1_r = (2 * precision_r * recall_r / (precision_r + recall_r)
                    if (precision_r + recall_r) > 0 else 0.0)
            per_round[int(r_idx)] = {
                'accuracy':          float(acc_before_update),
                'detection_recall':  float(recall_r),
                'detection_f1':      float(f1_r),
                'is_attack_phase':   int(r_idx) >= phase_transition_round,
            }

        # Summary key rounds (1-indexed in paper):
        # r13 = just before attack, r18 = 5 rds after, r25 = final
        def _get_acc(round_1indexed):
            r0 = round_1indexed - 1    # convert to 0-indexed round_metrics key
            if r0 in per_round:
                return per_round[r0]['accuracy']
            return None

        def _get_det_f1_range(from_round, to_round):
            """Mean detection F1 over a range of rounds (1-indexed)."""
            vals = [per_round[r-1]['detection_f1']
                    for r in range(from_round, to_round+1)
                    if (r-1) in per_round]
            return float(np.mean(vals)) if vals else 0.0

        acc_r13 = _get_acc(13)
        acc_r18 = _get_acc(18)
        acc_r25 = float(final_acc)    # after all 25 rounds

        det_f1_post_attack = _get_det_f1_range(13, 25)

        result = {
            'config_name':         cfg_name,
            'seed':                seed,
            'per_round':           per_round,
            'acc_r13':             acc_r13,
            'acc_r18':             acc_r18,
            'acc_r25':             acc_r25,
            'det_f1_r13_to_r25':   det_f1_post_attack,
            'final_accuracy':      float(final_acc),
            'total_time':          time.time() - start_t,
            'status':              'completed',
            'malicious_indices':   malicious_indices.tolist(),
        }

        def _pct(v):
            return f"{v*100:.2f}%" if v is not None else "--"

        logger.success(
            f"      → Acc@r13={_pct(acc_r13)}  "
            f"Acc@r18={_pct(acc_r18)}  "
            f"Acc@r25={_pct(acc_r25)}  "
            f"DetF1(r13-25)={det_f1_post_attack:.3f}"
        )
        return result

    except Exception as exc:
        logger.error(f"      → FAILED: {str(exc)[:120]}")
        traceback.print_exc()
        return {
            'config_name': cfg_name,
            'seed':        seed,
            'status':      'failed',
            'error':       str(exc),
            'total_time':  time.time() - start_t,
        }


def run_exp2_dynamic_attack(seeds=None, dry_run=False):
    """
    Experiment 2 (CRITICAL / NON-NEGOTIABLE):
    Dynamic attack schedule — phase-transition for RL justification.

    Phase transition:
      Rounds 1–12:  malicious clients behave BENIGNLY  (build false trust)
      Rounds 13–25: malicious clients switch to SCALING x20

    Outputs:
      - Summary table (CSV + JSON): acc@r13, acc@r18, acc@r25, detection F1
      - Per-round CSV: for convergence figure in the paper
    """
    logger.info('=' * 70)
    logger.info('EXPERIMENT 2 (CRITICAL): DYNAMIC ATTACK — PHASE TRANSITION')
    logger.info('  Dataset: Alzheimer MRI | Clients: 10 | Malicious: 40%')
    logger.info('  Rounds 1-12: BENIGN  →  Rounds 13-25: Scaling x20')
    logger.info('  Validates: RL temporal adaptivity advantage')
    logger.info('=' * 70)

    if seeds is None:
        seeds = [42, 123, 456]
    if dry_run:
        seeds = [42]
    num_rounds       = 3 if dry_run else 25
    phase_transition = 1 if dry_run else 12   # 0-indexed transition round

    all_results   = {}   # cfg_name → list of per-seed results
    all_per_round = []   # rows for per-round CSV

    for cfg in EXP2_CONFIGS:
        cfg_name = cfg['name']
        all_results[cfg_name] = []
        logger.info(f"\n  Config: {cfg_name}")

        for seed in seeds:
            r = _run_exp2_single(cfg, seed, num_rounds, phase_transition)
            all_results[cfg_name].append(r)

            if r['status'] == 'completed':
                for r_idx, rd in r['per_round'].items():
                    all_per_round.append({
                        'round':            r_idx + 1,    # 1-indexed for paper
                        'seed':             seed,
                        'config':           cfg_name,
                        'accuracy':         rd['accuracy'],
                        'detection_recall': rd['detection_recall'],
                        'detection_f1':     rd['detection_f1'],
                        'is_attack_phase':  int(rd['is_attack_phase']),
                    })

    # ---------- Per-round CSV ----------
    csv_pr_path = os.path.join(RESULTS_DIR, 'exp2_dynamic_per_round.csv')
    if all_per_round:
        fields_pr = ['round', 'seed', 'config', 'accuracy',
                     'detection_recall', 'detection_f1', 'is_attack_phase']
        with open(csv_pr_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields_pr)
            w.writeheader()
            w.writerows(all_per_round)
        logger.info(f"\n  Saved per-round CSV → {csv_pr_path}")

    # ---------- Summary table ----------
    logger.info('\n' + '=' * 80)
    logger.info('EXPERIMENT 2 SUMMARY TABLE')
    logger.info(f"{'Configuration':<22} {'Acc@r13 (%)':>12} {'Acc@r18 (%)':>12} "
                f"{'Acc@r25 (%)':>12} {'DetF1 r13-25':>14}")
    logger.info('-' * 80)

    summary_rows = []
    for cfg in EXP2_CONFIGS:
        cfg_name  = cfg['name']
        completed = [r for r in all_results[cfg_name] if r.get('status') == 'completed']
        if not completed:
            logger.warning(f"  {cfg_name}: No completed runs.")
            continue

        accs13 = [r['acc_r13'] for r in completed if r.get('acc_r13') is not None]
        accs18 = [r['acc_r18'] for r in completed if r.get('acc_r18') is not None]
        accs25 = [r['acc_r25'] for r in completed]
        f1s    = [r['det_f1_r13_to_r25'] for r in completed]

        def _fmt(vals):
            if not vals:
                return '  --'
            m = np.mean(vals) * 100
            s = np.std(vals)  * 100
            return f"{m:.2f}±{s:.2f}" if len(vals) > 1 else f"{m:.2f}"

        is_fedavg = cfg.get('_fedavg', False)
        f1_str = '  --' if is_fedavg else _fmt(f1s)

        row_str = (f"  {cfg_name:<20} {_fmt(accs13):>12} {_fmt(accs18):>12} "
                   f"{_fmt(accs25):>12} {f1_str:>14}")
        logger.info(row_str)

        summary_rows.append({
            'config_name':          cfg_name,
            'seeds':                seeds,
            'acc_r13_mean':         float(np.mean(accs13)) if accs13 else None,
            'acc_r13_std':          float(np.std(accs13))  if accs13 else None,
            'acc_r18_mean':         float(np.mean(accs18)) if accs18 else None,
            'acc_r18_std':          float(np.std(accs18))  if accs18 else None,
            'acc_r25_mean':         float(np.mean(accs25)) if accs25 else None,
            'acc_r25_std':          float(np.std(accs25))  if accs25 else None,
            'det_f1_r13_25_mean':   float(np.mean(f1s))   if f1s   else None,
            'det_f1_r13_25_std':    float(np.std(f1s))    if f1s   else None,
        })

    logger.info('=' * 80)

    # Save summary CSV + JSON
    csv_sum_path = os.path.join(RESULTS_DIR, 'exp2_dynamic_summary.csv')
    if summary_rows:
        with open(csv_sum_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()),
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(summary_rows)
        logger.info(f"  Saved summary CSV → {csv_sum_path}")

    json_path = os.path.join(RESULTS_DIR, 'exp2_dynamic_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  Saved JSON → {json_path}")

    logger.info('\nExperiment 2 complete.')
    return all_results, all_per_round


# ===========================================================================
# EXPERIMENT 3 — TRUST SCORE VISUALIZATION
# ===========================================================================

def run_exp3_trust_visualization(seed=42, dry_run=False):
    """
    Experiment 3: Qualitative trust score visualization.

    Runs a full OptiGradTrust experiment under scaling x20 and extracts
    per-client, per-round trust scores to generate:
      Option A: Trust Score Time Series  (PREFERRED — required)
      Option B: Trust Score Box Plot
      Option C: Shapley Value Distribution  (if Shapley enabled)

    All at ≥ 300 DPI, plus a raw-data CSV.
    """
    logger.info('=' * 70)
    logger.info('EXPERIMENT 3: TRUST SCORE / SHAPLEY VISUALIZATION')
    logger.info('  Dataset: Alzheimer MRI | Attack: Scaling x20')
    logger.info('  Configuration: Full OptiGradTrust | Seed=42 | 25 rounds')
    logger.info('=' * 70)

    import federated_learning.config.config as cfg_mod
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds

    num_rounds = 2 if dry_run else 25

    set_all_seeds(seed)

    configure_for_dataset(
        'ALZHEIMER', num_clients=10,
        non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
        aggregation_method='fedbn_fedprox'
    )
    cfg_mod.RANDOM_SEED       = seed
    cfg_mod.GLOBAL_EPOCHS     = num_rounds
    cfg_mod.FRACTION_MALICIOUS = 0.4
    cfg_mod.NUM_MALICIOUS     = 4
    cfg_mod.BATCH_SIZE        = 16
    cfg_mod.LR                = 1e-4
    cfg_mod.WEIGHT_DECAY      = 5e-5
    cfg_mod.ENABLE_VAE        = True
    cfg_mod.ENABLE_SHAPLEY    = True
    cfg_mod.ENABLE_DUAL_ATTENTION = True
    cfg_mod.RL_AGGREGATION_METHOD = 'hybrid'
    cfg_mod.RL_WARMUP_ROUNDS      = 5
    cfg_mod.RL_RAMP_UP_ROUNDS     = 10
    cfg_mod.AGGREGATION_METHOD    = 'fedbn_fedprox'
    cfg_mod.GRADIENT_COMBINATION_METHOD = 'fedbn_fedprox'
    cfg_mod.SCALING_FACTOR    = 20.0
    _sync_config_to_modules()

    set_random_seeds(seed)

    try:
        train_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=True, num_workers=0)

        server = Server()
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()

        _, client_datasets = create_client_datasets(
            train_dataset=train_dataset, num_clients=10, iid=False, alpha=0.1)

        malicious_indices = np.random.choice(10, 4, replace=False)
        clients = []
        client_is_malicious = {}
        for i in range(10):
            is_mal = int(i) in malicious_indices.tolist()
            client_is_malicious[i] = is_mal
            c = Client(client_id=i, dataset=client_datasets[i], is_malicious=is_mal)
            if is_mal:
                c.set_attack_parameters(
                    attack_type='scaling_attack',
                    scaling_factor=20.0,
                    partial_percent=cfg_mod.PARTIAL_SCALING_PERCENT,
                    noise_factor=cfg_mod.NOISE_FACTOR,
                    flip_probability=cfg_mod.FLIP_PROBABILITY,
                )
            clients.append(c)
        server.add_clients(clients)

        root_gradients = server._collect_root_gradients()
        server.root_gradients = root_gradients   # enables root-cosine & sign-consistency features
        server.vae = server.train_vae(root_gradients, vae_epochs=cfg_mod.VAE_EPOCHS)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=False, num_workers=0)
        server.test_loader = test_loader

        _, round_metrics = server.train(num_rounds=cfg_mod.GLOBAL_EPOCHS)
        final_acc = server.evaluate_model()
        logger.info(f"  Final accuracy: {final_acc*100:.2f}%")

        # ---- Extract trust scores and Shapley values ----
        trust_data = []   # list of dicts: round, client_id, is_malicious, trust_score, shapley_value

        for r_idx, rd in round_metrics.items():
            trust_scores   = rd.get('trust_scores',  {})
            features_dict  = rd.get('features',      {})    # features[client_id] = [f0..f5]

            for client_id in range(10):
                ts = trust_scores.get(client_id, None)
                if ts is None:
                    continue
                shap = None
                if client_id in features_dict:
                    feats = features_dict[client_id]
                    if isinstance(feats, list) and len(feats) >= 6:
                        shap = float(feats[5])

                trust_data.append({
                    'round':        int(r_idx) + 1,
                    'client_id':    client_id,
                    'is_malicious': int(client_is_malicious.get(client_id, False)),
                    'trust_score':  float(ts),
                    'shapley_value': shap,
                })

        if not trust_data:
            logger.warning("  No trust scores found in round_metrics. Skipping plots.")
            return {}

        # Save raw CSV
        csv_path = os.path.join(RESULTS_DIR, 'exp3_trust_scores.csv')
        fields = ['round', 'client_id', 'is_malicious', 'trust_score', 'shapley_value']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(trust_data)
        logger.info(f"  Saved trust data CSV → {csv_path}")

        # ---- Build per-client time series ----
        rounds_list   = sorted(set(d['round'] for d in trust_data))
        client_ts     = {}   # client_id → {round: trust_score}
        client_shap   = {}   # client_id → [shapley values across rounds]

        for d in trust_data:
            cid = d['client_id']
            if cid not in client_ts:
                client_ts[cid]   = {}
                client_shap[cid] = []
            client_ts[cid][d['round']] = d['trust_score']
            if d['shapley_value'] is not None:
                client_shap[cid].append(d['shapley_value'])

        # Benign/malicious split
        benign_ids   = [cid for cid, m in client_is_malicious.items() if not m]
        malicious_ids = [cid for cid, m in client_is_malicious.items() if m]

        # ---- Option A: Trust Score Time Series ----
        fig_a, ax_a = plt.subplots(figsize=(10, 5))
        for cid in benign_ids:
            ts_vals = [client_ts[cid].get(r, np.nan) for r in rounds_list]
            ax_a.plot(rounds_list, ts_vals, color='#2166ac', linewidth=1.5,
                      alpha=0.75, label='Benign' if cid == benign_ids[0] else '')
        for cid in malicious_ids:
            ts_vals = [client_ts[cid].get(r, np.nan) for r in rounds_list]
            ax_a.plot(rounds_list, ts_vals, color='#d6604d', linewidth=1.5,
                      linestyle='--', alpha=0.85,
                      label='Malicious' if cid == malicious_ids[0] else '')

        patch_b = mpatches.Patch(color='#2166ac', label='Benign clients')
        patch_m = mpatches.Patch(color='#d6604d', label='Malicious clients')
        ax_a.legend(handles=[patch_b, patch_m], fontsize=11)
        ax_a.set_xlabel('Round', fontsize=12)
        ax_a.set_ylabel('Trust Score', fontsize=12)
        ax_a.set_ylim(-0.05, 1.05)
        ax_a.set_title('Trust Score Time Series: Benign vs. Malicious Clients\n'
                        '(Alzheimer MRI, Scaling ×20, 40% Malicious, Seed=42)', fontsize=12)
        ax_a.grid(True, alpha=0.3)
        fig_a.tight_layout()
        fig_a_path = os.path.join(RESULTS_DIR, 'exp3_trust_timeseries.png')
        fig_a.savefig(fig_a_path, dpi=300, bbox_inches='tight')
        plt.close(fig_a)
        logger.success(f"  Saved Option A (time series) → {fig_a_path}")

        # ---- Option B: Trust Score Box Plot ----
        benign_scores   = [client_ts[cid].get(r, np.nan)
                           for cid in benign_ids for r in rounds_list
                           if not np.isnan(client_ts[cid].get(r, np.nan))]
        malicious_scores = [client_ts[cid].get(r, np.nan)
                            for cid in malicious_ids for r in rounds_list
                            if not np.isnan(client_ts[cid].get(r, np.nan))]

        if benign_scores and malicious_scores:
            fig_b, ax_b = plt.subplots(figsize=(6, 5))
            bp = ax_b.boxplot(
                [benign_scores, malicious_scores],
                labels=['Benign Clients', 'Malicious Clients'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
            )
            bp['boxes'][0].set_facecolor('#a6cee3')
            if len(bp['boxes']) > 1:
                bp['boxes'][1].set_facecolor('#fb9a99')
            ax_b.set_ylabel('Average Trust Score', fontsize=12)
            ax_b.set_title('Trust Score Distribution\n'
                           '(Benign vs. Malicious Clients)', fontsize=12)
            ax_b.grid(True, axis='y', alpha=0.3)
            fig_b.tight_layout()
            fig_b_path = os.path.join(RESULTS_DIR, 'exp3_trust_boxplot.png')
            fig_b.savefig(fig_b_path, dpi=300, bbox_inches='tight')
            plt.close(fig_b)
            logger.success(f"  Saved Option B (box plot) → {fig_b_path}")

        # ---- Option C: Shapley Value Distribution (violin) ----
        benign_shap   = [v for cid in benign_ids   for v in client_shap.get(cid, [])]
        malicious_shap = [v for cid in malicious_ids for v in client_shap.get(cid, [])]

        if benign_shap and malicious_shap:
            fig_c, ax_c = plt.subplots(figsize=(6, 5))
            parts = ax_c.violinplot(
                [benign_shap, malicious_shap],
                positions=[1, 2],
                showmedians=True,
            )
            for pc, color in zip(parts['bodies'], ['#a6cee3', '#fb9a99']):
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
            ax_c.set_xticks([1, 2])
            ax_c.set_xticklabels(['Benign Clients', 'Malicious Clients'])
            ax_c.set_ylabel('Shapley Value', fontsize=12)
            ax_c.set_title('Shapley Value Distribution\n'
                           '(Benign vs. Malicious Clients)', fontsize=12)
            ax_c.grid(True, axis='y', alpha=0.3)
            fig_c.tight_layout()
            fig_c_path = os.path.join(RESULTS_DIR, 'exp3_shapley_violin.png')
            fig_c.savefig(fig_c_path, dpi=300, bbox_inches='tight')
            plt.close(fig_c)
            logger.success(f"  Saved Option C (Shapley violin) → {fig_c_path}")

        logger.info('\nExperiment 3 complete.')
        return {'trust_data': trust_data, 'final_accuracy': float(final_acc)}

    except Exception as exc:
        logger.error(f"  Experiment 3 FAILED: {str(exc)[:200]}")
        traceback.print_exc()
        return {'status': 'failed', 'error': str(exc)}


# ===========================================================================
# EXPERIMENT 4 — OPTIMIZER COMPARISON UNDER ADVERSARIAL CONDITIONS
# ===========================================================================

# Optimizers to compare, in priority order (run as many as time allows)
EXP4_OPTIMIZERS = [
    {'name': 'FedAvg',    'method': 'fedavg',         'optigradtrust': False},
    {'name': 'FedBN',     'method': 'fedbn',           'optigradtrust': False},
    {'name': 'FedBN-P',   'method': 'fedbn_fedprox',   'optigradtrust': True},   # ours
    {'name': 'FedProx',   'method': 'fedprox',         'optigradtrust': False},
    {'name': 'FedNova',   'method': 'fednova',         'optigradtrust': False},
    {'name': 'SCAFFOLD',  'method': 'scaffold',        'optigradtrust': False},
    {'name': 'FedDWA',    'method': 'feddwa',          'optigradtrust': False},
    {'name': 'FedADMM',   'method': 'fedadmm',         'optigradtrust': False},
]


def _run_exp4_single_optimizer(opt_cfg, seed, num_rounds):
    """Run one optimizer for Experiment 4 (adversarial conditions)."""
    import federated_learning.config.config as cfg_mod
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds

    opt_name   = opt_cfg['name']
    opt_method = opt_cfg['method']
    use_trust  = opt_cfg['optigradtrust']

    logger.info(f"    [{opt_name}] seed={seed}")

    set_all_seeds(seed)

    # Configure
    configure_for_dataset(
        'ALZHEIMER', num_clients=10,
        non_iid_config={'enable': False, 'type': 'iid', 'alpha': None},
        aggregation_method=opt_method
    )
    cfg_mod.RANDOM_SEED        = seed
    cfg_mod.GLOBAL_EPOCHS      = num_rounds
    cfg_mod.FRACTION_MALICIOUS = 0.3
    cfg_mod.NUM_MALICIOUS      = 3
    cfg_mod.BATCH_SIZE         = 16
    cfg_mod.LR                 = 1e-4
    cfg_mod.WEIGHT_DECAY       = 5e-5
    cfg_mod.SCALING_FACTOR     = 10.0

    # For FedBN-P (our method), enable full OptiGradTrust trust mechanism
    # For others, use their plain aggregation (baseline mode — no trust)
    if use_trust:
        cfg_mod.ENABLE_VAE            = True
        cfg_mod.ENABLE_SHAPLEY        = True
        cfg_mod.ENABLE_DUAL_ATTENTION = True
        cfg_mod.RL_AGGREGATION_METHOD = 'hybrid'
        cfg_mod.RL_WARMUP_ROUNDS      = 5
        cfg_mod.RL_RAMP_UP_ROUNDS     = 10
        cfg_mod.AGGREGATION_METHOD    = opt_method
        cfg_mod.GRADIENT_COMBINATION_METHOD = opt_method
    # else: configure_for_dataset already set baseline mode (VAE/Shapley/DA disabled)

    _sync_config_to_modules()
    set_random_seeds(seed)
    start_t = time.time()

    try:
        train_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=True, num_workers=0)

        server = Server()
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()

        _, client_datasets = create_client_datasets(
            train_dataset=train_dataset, num_clients=10, iid=True, alpha=None)

        malicious_indices = np.random.choice(10, 3, replace=False)
        clients = []
        for i in range(10):
            is_mal = int(i) in malicious_indices.tolist()
            c = Client(client_id=i, dataset=client_datasets[i], is_malicious=is_mal)
            if is_mal:
                c.set_attack_parameters(
                    attack_type='scaling_attack',
                    scaling_factor=10.0,
                    partial_percent=cfg_mod.PARTIAL_SCALING_PERCENT,
                    noise_factor=cfg_mod.NOISE_FACTOR,
                    flip_probability=cfg_mod.FLIP_PROBABILITY,
                )
            clients.append(c)
        server.add_clients(clients)

        root_gradients = server._collect_root_gradients()
        server.root_gradients = root_gradients   # enables root-cosine & sign-consistency features
        if getattr(cfg_mod, 'ENABLE_VAE', False):
            server.vae = server.train_vae(root_gradients, vae_epochs=cfg_mod.VAE_EPOCHS)
        else:
            server.vae = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg_mod.BATCH_SIZE, shuffle=False, num_workers=0)
        server.test_loader = test_loader

        _, round_metrics = server.train(num_rounds=cfg_mod.GLOBAL_EPOCHS)
        final_acc = server.evaluate_model()

        # Per-round accuracy
        per_round_acc = {}
        for r_idx, rd in round_metrics.items():
            per_round_acc[int(r_idx) + 1] = float(rd.get('test_accuracy', 0.0))

        result = {
            'optimizer_name':   opt_name,
            'method':           opt_method,
            'seed':             seed,
            'final_accuracy':   float(final_acc),
            'per_round_acc':    per_round_acc,
            'total_time':       time.time() - start_t,
            'status':           'completed',
        }
        logger.success(f"      → Final Acc={final_acc*100:.2f}%")
        return result

    except Exception as exc:
        logger.error(f"      → FAILED: {str(exc)[:120]}")
        traceback.print_exc()
        return {
            'optimizer_name': opt_name,
            'method':         opt_method,
            'seed':           seed,
            'status':         'failed',
            'error':          str(exc),
            'total_time':     time.time() - start_t,
        }


def run_exp4_optimizer_adversarial(seed=42, dry_run=False, n_optimizers=4):
    """
    Experiment 4: Optimizer comparison under adversarial conditions.
    IID, 30% malicious, scaling x10, 25 rounds, seed=42.

    Generates:
      - Summary table (CSV): final accuracy per optimizer
      - Per-round CSV: accuracy vs round (for convergence figure)
      - Convergence figure (PNG)
    """
    logger.info('=' * 70)
    logger.info('EXPERIMENT 4: OPTIMIZER COMPARISON UNDER ADVERSARIAL CONDITIONS')
    logger.info('  Dataset: Alzheimer MRI | IID | Malicious: 30% | Scaling x10')
    logger.info('  Compares: FedAvg, FedBN, FedBN-P, FedProx [+ FedNova, SCAFFOLD, FedDWA, FedADMM]')
    logger.info('=' * 70)

    num_rounds = 2 if dry_run else 25
    # Limit to n_optimizers (priority order); if dry_run, just top 3
    opts_to_run = EXP4_OPTIMIZERS[:3] if dry_run else EXP4_OPTIMIZERS[:n_optimizers]

    all_results   = []
    per_round_rows = []

    for opt_cfg in opts_to_run:
        logger.info(f"\n  Optimizer: {opt_cfg['name']}")
        r = _run_exp4_single_optimizer(opt_cfg, seed, num_rounds)
        all_results.append(r)

        if r['status'] == 'completed':
            for round_num, acc in r['per_round_acc'].items():
                per_round_rows.append({
                    'round':          round_num,
                    'optimizer':      opt_cfg['name'],
                    'accuracy':       acc,
                    'is_our_method':  int(opt_cfg['optigradtrust']),
                })

    # ---- Per-round CSV ----
    csv_pr_path = os.path.join(RESULTS_DIR, 'exp4_optimizer_adversarial_per_round.csv')
    if per_round_rows:
        with open(csv_pr_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['round', 'optimizer', 'accuracy', 'is_our_method'])
            w.writeheader()
            w.writerows(per_round_rows)
        logger.info(f"\n  Saved per-round CSV → {csv_pr_path}")

    # ---- Summary table ----
    logger.info('\n' + '=' * 70)
    logger.info('EXPERIMENT 4 SUMMARY: Optimizer Final Accuracy under Adversarial Conditions')
    logger.info(f"{'Optimizer':<12} {'Final Acc (%)':>14}  {'Defense':>10}")
    logger.info('-' * 50)
    summary_rows = []
    for r in all_results:
        if r['status'] != 'completed':
            continue
        has_trust = next((o['optigradtrust'] for o in opts_to_run
                          if o['name'] == r['optimizer_name']), False)
        defense_str = 'OptiGradTrust' if has_trust else 'None'
        logger.info(f"  {r['optimizer_name']:<10} {r['final_accuracy']*100:>14.2f}  {defense_str:>10}")
        summary_rows.append({
            'optimizer':      r['optimizer_name'],
            'method':         r['method'],
            'final_accuracy': r['final_accuracy'],
            'has_defense':    int(has_trust),
        })
    logger.info('=' * 70)

    csv_sum_path = os.path.join(RESULTS_DIR, 'exp4_optimizer_adversarial_summary.csv')
    if summary_rows:
        with open(csv_sum_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        logger.info(f"  Saved summary CSV → {csv_sum_path}")

    json_path = os.path.join(RESULTS_DIR, 'exp4_optimizer_adversarial_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  Saved JSON → {json_path}")

    # ---- Convergence figure ----
    if per_round_rows:
        try:
            opt_names = list(dict.fromkeys(r['optimizer'] for r in per_round_rows))
            colors    = plt.cm.tab10(np.linspace(0, 1, len(opt_names)))
            fig, ax   = plt.subplots(figsize=(10, 5))
            for opt_name, color in zip(opt_names, colors):
                rows = [(r['round'], r['accuracy'])
                        for r in per_round_rows if r['optimizer'] == opt_name]
                if not rows:
                    continue
                rows.sort(key=lambda x: x[0])
                xs = [r[0] for r in rows]
                ys = [r[1] * 100 for r in rows]
                lw = 2.5 if opt_name == 'FedBN-P' else 1.5
                ls = '-' if opt_name == 'FedBN-P' else '--'
                ax.plot(xs, ys, label=opt_name, color=color, linewidth=lw, linestyle=ls)

            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Test Accuracy (%)', fontsize=12)
            ax.set_title('Optimizer Comparison under Adversarial Conditions\n'
                         '(Alzheimer MRI, IID, 30% Malicious, Scaling ×10)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = os.path.join(RESULTS_DIR, 'exp4_optimizer_adversarial_convergence.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.success(f"  Saved convergence figure → {fig_path}")
        except Exception as exc:
            logger.warning(f"  Could not generate convergence figure: {exc}")

    logger.info('\nExperiment 4 complete.')
    return all_results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OptiGradTrust Round-2 Revision Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Priority order (run in this order if time is limited):
  1. exp2   — Dynamic attack / RL justification  [NON-NEGOTIABLE]
  2. exp1a  — Noise ablation / VAE validation     [CRITICAL]
  3. exp3   — Trust visualization                  [REQUIRED]
  4. exp1b  — Sign-flip ablation                  [CRITICAL]
  5. exp4   — Optimizer adversarial comparison     [RECOMMENDED]

Examples:
  python run_r2_experiments.py --experiment exp2 --seeds 42 123 456
  python run_r2_experiments.py --experiment exp1a --seeds 42 123 456 789 1024
  python run_r2_experiments.py --experiment all --seeds 42
  python run_r2_experiments.py --experiment exp2 --dry-run
        """
    )
    parser.add_argument(
        '--experiment',
        choices=['exp1a', 'exp1b', 'exp2', 'exp3', 'exp4', 'all'],
        default='all',
        help='Which experiment to run'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+',
        default=None,
        help='Random seeds (default: single seed=42 for most; 42 123 456 for exp2)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Quick 2-3 round sanity check (do not use for real results)'
    )
    parser.add_argument(
        '--n-optimizers', type=int, default=4,
        help='Number of optimizers to run in Experiment 4 (1-8, default 4)'
    )
    args = parser.parse_args()

    logger.info('=' * 70)
    logger.info('OptiGradTrust — Round 2 Revision Experiments')
    logger.info(f'Experiment: {args.experiment}  |  Dry-run: {args.dry_run}')
    logger.info(f'Seeds: {args.seeds}')
    logger.info(f'Output dir: {RESULTS_DIR}')
    logger.info('=' * 70)

    exp = args.experiment
    dry = args.dry_run

    if exp in ('exp1a', 'all'):
        seeds = args.seeds or ([42] if dry else [42])
        run_exp1a_noise_ablation(seeds=seeds, dry_run=dry)

    if exp in ('exp1b', 'all'):
        seeds = args.seeds or ([42] if dry else [42])
        run_exp1b_signflip_ablation(seeds=seeds, dry_run=dry)

    if exp in ('exp2', 'all'):
        seeds = args.seeds or ([42] if dry else [42, 123, 456])
        run_exp2_dynamic_attack(seeds=seeds, dry_run=dry)

    if exp in ('exp3', 'all'):
        seed = (args.seeds[0] if args.seeds else 42)
        run_exp3_trust_visualization(seed=seed, dry_run=dry)

    if exp in ('exp4', 'all'):
        seed = (args.seeds[0] if args.seeds else 42)
        run_exp4_optimizer_adversarial(seed=seed, dry_run=dry,
                                        n_optimizers=args.n_optimizers)

    logger.info('\n' + '=' * 70)
    logger.info('All requested R2 experiments finished.')
    logger.info(f'Results saved in: {RESULTS_DIR}')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
