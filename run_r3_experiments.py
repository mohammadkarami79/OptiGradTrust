#!/usr/bin/env python3
"""
=============================================================================
OptiGradTrust -- Third Revision (R3) Experiment Runner
=============================================================================

Six core experiments requested by Reviewer 1 in the third revision:

  R3-A  Attack budget failure-mode sweep
        (scaling {x1,x10,x50,x100} + sign-flip fraction {10,20,40,60}%)
  R3-B  Computational / communication overhead comparison
        (FedAvg, FedBN, Krum, FLTrust, OptiGradTrust)
  R3-C  Multi-seed re-run of headline accuracy table
        (2 distributions x 3 attacks x 4 methods x 3 seeds = 72 runs)
  R3-D  Adaptive attacker (VAE-aware PGD)
  R3-E  RL vs rule-based aggregator ablation
  R3-F  Two additional baselines (RFA, SignGuard)

USAGE:
    python run_r3_experiments.py --experiment r3a --seeds 42 123 456
    python run_r3_experiments.py --experiment r3b --seeds 42
    python run_r3_experiments.py --experiment r3c --seeds 42 123 456
    python run_r3_experiments.py --experiment r3d --seeds 42 123 456
    python run_r3_experiments.py --experiment r3e --seeds 42 123 456
    python run_r3_experiments.py --experiment r3f --seeds 42 123 456
    python run_r3_experiments.py --experiment all
    python run_r3_experiments.py --experiment r3a --dry-run

Author: OptiGradTrust Team
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import time
import csv
import argparse
import traceback
import warnings
import copy
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from run_all_experiments import (
    set_all_seeds, configure_for_dataset,
    Logger, _sync_config_to_modules,
)

# ---------------------------------------------------------------------------
# Output directory & logger
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'r3_revision')
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    RESULTS_DIR,
    f'r3_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
)
logger = Logger(LOG_FILE)


# ---------------------------------------------------------------------------
# Common config knobs for R3 Alzheimer runs
# ---------------------------------------------------------------------------

def _apply_alzheimer_r3_config(cfg_mod, num_rounds, malicious_fraction=0.3,
                                fedprox_mu=0.01):
    """
    Shared Alzheimer-MRI configuration used by all R3 experiments.

    Mirrors R2 hyper-parameters: 10 clients, 30% malicious by default,
    batch=16, LR=1e-4, weight-decay=5e-5.
    """
    cfg_mod.GLOBAL_EPOCHS      = num_rounds
    cfg_mod.FRACTION_MALICIOUS = float(malicious_fraction)
    cfg_mod.NUM_MALICIOUS      = max(1, int(round(10 * malicious_fraction)))
    cfg_mod.BATCH_SIZE         = 16
    cfg_mod.LR                 = 1e-4
    cfg_mod.LEARNING_RATE      = 1e-4
    cfg_mod.WEIGHT_DECAY       = 5e-5
    cfg_mod.FEDPROX_MU         = fedprox_mu


def _apply_method_config(cfg_mod, method_name, num_rounds,
                          malicious_fraction=0.3, fedprox_mu=0.01,
                          use_rule_based_trust=False, measure_overhead=False):
    """
    Configure cfg_mod for a given aggregation method.

    method_name is one of:
      'optigradtrust', 'fedavg', 'fedbn', 'fedprox', 'krum', 'fltrust',
      'rfa', 'signguard', 'optigradtrust_rulebased', 'optigradtrust_norl'.

    Calls configure_for_dataset first (which sets the pure-baseline defences
    correctly) and then overrides what we need.
    """
    # map pretty name -> aggregation_method used by configure_for_dataset
    agg_map = {
        'optigradtrust':            'fedbn_fedprox',
        'optigradtrust_rulebased':  'fedbn_fedprox',
        'optigradtrust_norl':       'fedbn_fedprox',
        'fedavg':                   'fedavg',
        'fedbn':                    'fedbn',
        'fedprox':                  'fedprox',
        'krum':                     'krum',
        'fltrust':                  'fltrust',
        'rfa':                      'rfa',
        'signguard':                'signguard',
    }
    if method_name not in agg_map:
        raise ValueError(f"Unknown method_name: {method_name}")
    aggregation_method = agg_map[method_name]

    configure_for_dataset(
        'ALZHEIMER', num_clients=10,
        non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
        aggregation_method=aggregation_method
    )
    _apply_alzheimer_r3_config(cfg_mod, num_rounds, malicious_fraction, fedprox_mu)

    # OptiGradTrust variants: ensure defences ON (configure_for_dataset
    # already enables hybrid RL for fedbn_fedprox, but be explicit)
    if method_name == 'optigradtrust':
        cfg_mod.ENABLE_VAE            = True
        cfg_mod.ENABLE_SHAPLEY        = True
        cfg_mod.ENABLE_DUAL_ATTENTION = True
        cfg_mod.RL_AGGREGATION_METHOD = 'hybrid'
        cfg_mod.RL_WARMUP_ROUNDS      = 5
        cfg_mod.RL_RAMP_UP_ROUNDS     = 10
        cfg_mod.USE_RULE_BASED_TRUST  = False
    elif method_name == 'optigradtrust_rulebased':
        cfg_mod.ENABLE_VAE            = True
        cfg_mod.ENABLE_SHAPLEY        = True
        cfg_mod.ENABLE_DUAL_ATTENTION = True  # present but bypassed when rule-based is on
        cfg_mod.RL_AGGREGATION_METHOD = 'dual_attention'
        cfg_mod.RL_WARMUP_ROUNDS      = 9999
        cfg_mod.USE_RULE_BASED_TRUST  = True
    elif method_name == 'optigradtrust_norl':
        cfg_mod.ENABLE_VAE            = True
        cfg_mod.ENABLE_SHAPLEY        = True
        cfg_mod.ENABLE_DUAL_ATTENTION = True
        cfg_mod.RL_AGGREGATION_METHOD = 'dual_attention'
        cfg_mod.RL_WARMUP_ROUNDS      = 9999
        cfg_mod.USE_RULE_BASED_TRUST  = False
    else:
        # pure baselines
        cfg_mod.USE_RULE_BASED_TRUST = False

    # overhead measurement
    cfg_mod.MEASURE_OVERHEAD = bool(measure_overhead)


# ---------------------------------------------------------------------------
# Shared training skeleton
# ---------------------------------------------------------------------------

def _build_and_train(
    method_name: str,
    seed: int,
    num_rounds: int,
    attack_type: str,
    attack_params: Optional[dict] = None,
    malicious_fraction: float = 0.3,
    non_iid_alpha: Optional[float] = 0.1,
    dynamic_schedule: Optional[str] = None,
    phase_transition_round: int = 12,
    measure_overhead: bool = False,
) -> Tuple[dict, dict]:
    """
    One full Alzheimer run.
    Returns (result_dict, round_metrics).
    If the run fails, result_dict['status'] == 'failed' and round_metrics == {}.
    """
    import federated_learning.config.config as cfg_mod
    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds

    set_all_seeds(seed)
    iid = (non_iid_alpha is None)
    use_rule_based = (method_name == 'optigradtrust_rulebased')

    _apply_method_config(
        cfg_mod, method_name, num_rounds,
        malicious_fraction=malicious_fraction,
        measure_overhead=measure_overhead,
    )
    cfg_mod.RANDOM_SEED = seed
    if iid:
        cfg_mod.ENABLE_NON_IID = False
    else:
        cfg_mod.ENABLE_NON_IID = True
        cfg_mod.DIRICHLET_ALPHA = float(non_iid_alpha)
    _sync_config_to_modules()
    set_random_seeds(seed)

    start_t = time.time()

    try:
        train_dataset, test_dataset = load_dataset()
        root_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg_mod.BATCH_SIZE,
            shuffle=True, num_workers=0,
        )

        server = Server()
        # Hint num_byzantine for Krum
        server._num_malicious_hint = cfg_mod.NUM_MALICIOUS
        server.set_datasets(root_loader, test_dataset)
        server._pretrain_global_model()

        _, client_datasets = create_client_datasets(
            train_dataset=train_dataset, num_clients=10,
            iid=iid, alpha=(non_iid_alpha if non_iid_alpha is not None else 0.1),
        )

        num_mal = cfg_mod.NUM_MALICIOUS
        malicious_indices = np.random.choice(10, num_mal, replace=False)
        clients = []
        ap = dict(attack_params) if attack_params else {}
        for i in range(10):
            is_mal = int(i) in malicious_indices.tolist()
            c = Client(client_id=i, dataset=client_datasets[i], is_malicious=is_mal)
            if is_mal:
                c.set_attack_parameters(
                    attack_type=attack_type,
                    scaling_factor=ap.get('scaling_factor',
                                          getattr(cfg_mod, 'SCALING_FACTOR', 10.0)),
                    sigma=ap.get('sigma', 15.0),
                    partial_percent=getattr(cfg_mod, 'PARTIAL_SCALING_PERCENT', 0.3),
                    noise_factor=getattr(cfg_mod, 'NOISE_FACTOR', 0.5),
                    flip_probability=getattr(cfg_mod, 'FLIP_PROBABILITY', 1.0),
                )
                # forward all remaining attack_params onto the client
                if not hasattr(c, 'attack_params') or c.attack_params is None:
                    c.attack_params = {}
                for k, v in ap.items():
                    c.attack_params[k] = v
                c.attack.attack_params = c.attack_params
                if dynamic_schedule:
                    c.dynamic_attack_schedule = dynamic_schedule
                    c.phase_transition_round = phase_transition_round
            clients.append(c)
        server.add_clients(clients)

        # Root gradients for FLTrust + feature computation
        root_gradients = server._collect_root_gradients()
        server.root_gradients = root_gradients

        # Train VAE (only if enabled for this method)
        if getattr(cfg_mod, 'ENABLE_VAE', False):
            try:
                server.vae = server.train_vae(
                    root_gradients,
                    vae_epochs=getattr(cfg_mod, 'VAE_EPOCHS', 5),
                )
            except Exception as _e:
                logger.warning(f"  VAE training failed ({_e}); disabling VAE")
                server.vae = None
        else:
            server.vae = None

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg_mod.BATCH_SIZE,
            shuffle=False, num_workers=0,
        )
        server.test_loader = test_loader

        _ = use_rule_based  # flag is read by server via cfg_mod.USE_RULE_BASED_TRUST

        _, round_metrics = server.train(num_rounds=cfg_mod.GLOBAL_EPOCHS)
        final_acc = server.evaluate_model()

        # Aggregate detection metrics over all rounds
        total_tp = total_fp = total_fn = total_tn = 0
        agg_times, comm_vols, peak_mems = [], [], []
        for rd in round_metrics.values():
            det = rd.get('detection_results', {})
            total_tp += det.get('true_positives',  0)
            total_fp += det.get('false_positives', 0)
            total_fn += det.get('false_negatives', 0)
            total_tn += det.get('true_negatives',  0)
            oh = rd.get('overhead', None)
            if oh:
                agg_times.append(oh.get('agg_wall_time_s', 0.0))
                comm_vols.append(oh.get('comm_volume_mb',  0.0))
                peak_mems.append(oh.get('peak_mem_mb',     0.0))

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # per-round accuracy (BEFORE update of that round)
        per_round_acc = []
        for r_idx in sorted(round_metrics.keys()):
            per_round_acc.append(float(round_metrics[r_idx].get('test_accuracy', 0.0)))

        result = {
            'method':          method_name,
            'seed':            seed,
            'attack_type':     attack_type,
            'attack_params':   ap,
            'malicious_fraction': malicious_fraction,
            'non_iid_alpha':   (None if iid else float(non_iid_alpha)),
            'final_accuracy':  float(final_acc),
            'per_round_accuracy': per_round_acc,
            'precision':       float(precision),
            'recall':          float(recall),
            'f1':              float(f1),
            'tp': int(total_tp), 'fp': int(total_fp),
            'fn': int(total_fn), 'tn': int(total_tn),
            'total_time_s':    float(time.time() - start_t),
            'status':          'completed',
        }
        if agg_times:
            result['overhead'] = {
                'mean_agg_wall_time_s': float(np.mean(agg_times)),
                'std_agg_wall_time_s':  float(np.std(agg_times)),
                'mean_comm_volume_mb':  float(np.mean(comm_vols)) if comm_vols else 0.0,
                'mean_peak_mem_mb':     float(np.mean(peak_mems)) if peak_mems else 0.0,
                'num_rounds_measured':  len(agg_times),
            }
        logger.success(
            f"      -> Acc={final_acc*100:.2f}% "
            f"Prec={precision*100:.2f} Rec={recall*100:.2f} F1={f1*100:.2f} "
            f"t={result['total_time_s']:.1f}s"
        )
        return result, round_metrics

    except Exception as exc:
        logger.error(f"      -> FAILED: {str(exc)[:200]}")
        traceback.print_exc()
        return ({
            'method':       method_name,
            'seed':         seed,
            'attack_type':  attack_type,
            'status':       'failed',
            'error':        str(exc),
            'total_time_s': time.time() - start_t,
        }, {})


# ---------------------------------------------------------------------------
# CSV / JSON helpers
# ---------------------------------------------------------------------------

def _save_csv(rows, path, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in fields}
            # flatten dicts into json strings
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    row[k] = json.dumps(v)
            w.writerow(row)


def _save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ===========================================================================
# R3-A  Attack budget failure-mode sweep
# ===========================================================================

R3A_SCALING_LEVELS     = [1.0, 10.0, 50.0, 100.0]
R3A_SIGNFLIP_FRACTIONS = [0.10, 0.20, 0.40, 0.60]


def run_r3a(seeds: List[int], dry_run: bool = False):
    logger.info('=' * 70)
    logger.info('R3-A  ATTACK BUDGET FAILURE-MODE SWEEP')
    logger.info('  Alzheimer MRI | 10 clients | 25 rounds | OptiGradTrust only')
    logger.info(f'  Seeds: {seeds}   dry_run={dry_run}')
    logger.info('=' * 70)

    out_dir = os.path.join(RESULTS_DIR, 'r3a_attack_budget')
    os.makedirs(out_dir, exist_ok=True)
    num_rounds = 2 if dry_run else 25
    all_rows = []

    # ---- scaling sweep --------------------------------------------------
    logger.info('\n[R3-A | scaling sweep]')
    for factor in R3A_SCALING_LEVELS:
        for seed in seeds:
            logger.info(f"  scaling x{factor}  seed={seed}")
            res, _ = _build_and_train(
                method_name='optigradtrust', seed=seed, num_rounds=num_rounds,
                attack_type='scaling_attack',
                attack_params={'scaling_factor': float(factor)},
                malicious_fraction=0.3,
            )
            res['sweep'] = 'scaling'
            res['intensity'] = float(factor)
            all_rows.append(res)

    # ---- sign-flip sweep (vary malicious fraction) ----------------------
    logger.info('\n[R3-A | sign-flip sweep]')
    for frac in R3A_SIGNFLIP_FRACTIONS:
        for seed in seeds:
            logger.info(f"  sign-flip mal_frac={int(frac*100)}%  seed={seed}")
            res, _ = _build_and_train(
                method_name='optigradtrust', seed=seed, num_rounds=num_rounds,
                attack_type='sign_flipping_attack',
                malicious_fraction=float(frac),
            )
            res['sweep'] = 'signflip'
            res['intensity'] = float(frac)
            all_rows.append(res)

    # ---- save -----------------------------------------------------------
    fields = ['method', 'sweep', 'intensity', 'seed', 'attack_type',
              'final_accuracy', 'precision', 'recall', 'f1',
              'tp', 'fp', 'fn', 'tn', 'total_time_s', 'status']
    csv_path = os.path.join(out_dir, 'r3a_attack_budget.csv')
    _save_csv(all_rows, csv_path, fields)
    _save_json(all_rows, os.path.join(out_dir, 'r3a_attack_budget.json'))
    logger.info(f"  Saved CSV -> {csv_path}")

    # ---- plots (one per sweep) ------------------------------------------
    try:
        _plot_r3a(all_rows, out_dir)
    except Exception as _e:
        logger.warning(f"  Plotting failed ({_e})")

    _print_r3a_table(all_rows)
    logger.info('R3-A complete.')
    return all_rows


def _plot_r3a(rows, out_dir):
    # accuracy vs intensity with error bars (mean ± std across seeds)
    for sweep in ('scaling', 'signflip'):
        rows_s = [r for r in rows if r.get('sweep') == sweep and r.get('status') == 'completed']
        if not rows_s:
            continue
        intensities = sorted({r['intensity'] for r in rows_s})
        means, stds = [], []
        for intens in intensities:
            accs = [r['final_accuracy'] for r in rows_s if r['intensity'] == intens]
            means.append(np.mean(accs))
            stds.append(np.std(accs))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(intensities, np.array(means) * 100, yerr=np.array(stds) * 100,
                    marker='o', capsize=4, linewidth=1.5)
        ax.set_xlabel('Scaling factor' if sweep == 'scaling' else 'Malicious fraction')
        ax.set_ylabel('Final accuracy (%)')
        ax.set_title(f'R3-A — {sweep} sweep (mean ± std over seeds)')
        if sweep == 'scaling':
            ax.set_xscale('log')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f'r3a_{sweep}_accuracy.png')
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logger.info(f"  Plot -> {path}")


def _print_r3a_table(rows):
    completed = [r for r in rows if r.get('status') == 'completed']
    logger.info('\n' + '=' * 74)
    logger.info(f"{'Sweep':<10} {'Intensity':>10} {'mean Acc':>10} {'std Acc':>8} "
                f"{'mean F1':>10} {'runs':>5}")
    logger.info('-' * 74)
    for sweep in ('scaling', 'signflip'):
        rows_s = [r for r in completed if r.get('sweep') == sweep]
        intensities = sorted({r['intensity'] for r in rows_s})
        for intens in intensities:
            g = [r for r in rows_s if r['intensity'] == intens]
            accs = [r['final_accuracy'] for r in g]
            f1s  = [r['f1'] for r in g]
            logger.info(f"{sweep:<10} {intens:>10.2f} {np.mean(accs)*100:>10.2f} "
                        f"{np.std(accs)*100:>8.2f} {np.mean(f1s)*100:>10.2f} {len(g):>5}")
    logger.info('=' * 74)


# ===========================================================================
# R3-B  Computational / communication overhead
# ===========================================================================

R3B_METHODS = ['fedavg', 'fedbn', 'krum', 'fltrust', 'optigradtrust']


def run_r3b(seeds: List[int], dry_run: bool = False):
    logger.info('=' * 70)
    logger.info('R3-B  COMPUTATIONAL / COMMUNICATION OVERHEAD COMPARISON')
    logger.info('  Alzheimer MRI | 10 clients | 25 rounds | IID | 30% scaling')
    logger.info('  Methods: ' + ', '.join(R3B_METHODS))
    logger.info(f'  Seeds: {seeds}   dry_run={dry_run}')
    logger.info('=' * 70)

    out_dir = os.path.join(RESULTS_DIR, 'r3b_overhead')
    os.makedirs(out_dir, exist_ok=True)
    num_rounds = 2 if dry_run else 25
    all_rows = []

    for method in R3B_METHODS:
        for seed in seeds:
            logger.info(f"  method={method}  seed={seed}")
            res, _ = _build_and_train(
                method_name=method, seed=seed, num_rounds=num_rounds,
                attack_type='scaling_attack',
                attack_params={'scaling_factor': 10.0},
                malicious_fraction=0.3,
                non_iid_alpha=None,          # IID
                measure_overhead=True,
            )
            all_rows.append(res)

    # CSV + JSON
    fields = ['method', 'seed', 'final_accuracy', 'precision', 'recall', 'f1',
              'total_time_s', 'status']
    csv_path = os.path.join(out_dir, 'r3b_overhead.csv')
    _save_csv(all_rows, csv_path, fields)
    _save_json(all_rows, os.path.join(out_dir, 'r3b_overhead_full.json'))
    logger.info(f"  Saved CSV -> {csv_path}")

    # Build a focused overhead table
    summary_rows = []
    for method in R3B_METHODS:
        sel = [r for r in all_rows if r.get('method') == method and r.get('status') == 'completed']
        if not sel:
            continue
        agg_mean = np.mean([r.get('overhead', {}).get('mean_agg_wall_time_s', 0.0) for r in sel])
        comm_mean = np.mean([r.get('overhead', {}).get('mean_comm_volume_mb', 0.0) for r in sel])
        mem_mean = np.mean([r.get('overhead', {}).get('mean_peak_mem_mb', 0.0) for r in sel])
        tot_time = np.mean([r.get('total_time_s', 0.0) for r in sel])
        acc_mean = np.mean([r.get('final_accuracy', 0.0) for r in sel])
        summary_rows.append({
            'method':                 method,
            'mean_agg_wall_time_ms':  float(agg_mean * 1000.0),
            'mean_comm_volume_mb':    float(comm_mean),
            'mean_peak_mem_mb':       float(mem_mean),
            'mean_total_time_s':      float(tot_time),
            'mean_final_accuracy':    float(acc_mean),
            'num_seeds':              len(sel),
        })
    sum_csv = os.path.join(out_dir, 'r3b_overhead_summary.csv')
    _save_csv(summary_rows, sum_csv,
              ['method', 'mean_agg_wall_time_ms', 'mean_comm_volume_mb',
               'mean_peak_mem_mb', 'mean_total_time_s', 'mean_final_accuracy',
               'num_seeds'])
    logger.info(f"  Saved overhead summary -> {sum_csv}")

    # Bar chart (agg time, log scale)
    try:
        methods = [r['method'] for r in summary_rows]
        times_ms = [r['mean_agg_wall_time_ms'] for r in summary_rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(methods, times_ms)
        ax.set_yscale('log')
        ax.set_ylabel('Mean aggregation wall-time per round (ms)')
        ax.set_title('R3-B — Aggregation overhead (log scale)')
        for i, v in enumerate(times_ms):
            ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        fig.tight_layout()
        plt.xticks(rotation=20)
        plot_path = os.path.join(out_dir, 'r3b_overhead_bar.png')
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"  Plot -> {plot_path}")
    except Exception as _e:
        logger.warning(f"  Plotting failed ({_e})")

    # Pretty-print the summary
    logger.info('\n' + '=' * 90)
    logger.info(f"{'Method':<16} {'Agg (ms)':>10} {'Comm (MB)':>11} "
                f"{'Peak mem MB':>12} {'Tot (s)':>10} {'Acc (%)':>9}")
    logger.info('-' * 90)
    for r in summary_rows:
        logger.info(f"{r['method']:<16} {r['mean_agg_wall_time_ms']:>10.2f} "
                    f"{r['mean_comm_volume_mb']:>11.2f} {r['mean_peak_mem_mb']:>12.2f} "
                    f"{r['mean_total_time_s']:>10.1f} {r['mean_final_accuracy']*100:>9.2f}")
    logger.info('=' * 90)
    logger.info('R3-B complete.')
    return all_rows


# ===========================================================================
# R3-C  Multi-seed headline table
# ===========================================================================

R3C_METHODS       = ['fedavg', 'fedbn', 'fltrust', 'optigradtrust']
R3C_DISTRIBUTIONS = [('IID', None), ('NonIID_a0.1', 0.1)]
R3C_ATTACKS = [
    ('scaling_x20', 'scaling_attack',         {'scaling_factor': 20.0}),
    ('sign_flip',   'sign_flipping_attack',   {}),
    ('gauss_s15',   'gaussian_noise_injection', {'sigma': 15.0}),
]


def run_r3c(seeds: List[int], dry_run: bool = False):
    logger.info('=' * 70)
    logger.info('R3-C  MULTI-SEED HEADLINE TABLE')
    logger.info('  Alzheimer | 10 clients | 25 rounds')
    logger.info(f'  Methods: {R3C_METHODS}')
    logger.info(f'  Distributions: {[d[0] for d in R3C_DISTRIBUTIONS]}')
    logger.info(f'  Attacks: {[a[0] for a in R3C_ATTACKS]}')
    logger.info(f'  Seeds: {seeds}   dry_run={dry_run}')
    logger.info('=' * 70)

    out_dir = os.path.join(RESULTS_DIR, 'r3c_headline')
    os.makedirs(out_dir, exist_ok=True)
    num_rounds = 2 if dry_run else 25
    all_rows = []

    for dist_name, alpha in R3C_DISTRIBUTIONS:
        for atk_name, atk_type, atk_params in R3C_ATTACKS:
            for method in R3C_METHODS:
                for seed in seeds:
                    logger.info(f"  [{dist_name}|{atk_name}|{method}]  seed={seed}")
                    res, _ = _build_and_train(
                        method_name=method, seed=seed, num_rounds=num_rounds,
                        attack_type=atk_type, attack_params=atk_params,
                        malicious_fraction=0.3, non_iid_alpha=alpha,
                    )
                    res['distribution'] = dist_name
                    res['attack_name']  = atk_name
                    all_rows.append(res)

    fields = ['distribution', 'attack_name', 'method', 'seed', 'attack_type',
              'final_accuracy', 'precision', 'recall', 'f1',
              'tp', 'fp', 'fn', 'tn', 'total_time_s', 'status']
    csv_path = os.path.join(out_dir, 'r3c_headline_raw.csv')
    _save_csv(all_rows, csv_path, fields)
    _save_json(all_rows, os.path.join(out_dir, 'r3c_headline_raw.json'))
    logger.info(f"  Saved raw -> {csv_path}")

    # Headline summary (mean ± std per cell) + 95% CI on F1
    summary = []
    for dist_name, _ in R3C_DISTRIBUTIONS:
        for atk_name, _, _ in R3C_ATTACKS:
            for method in R3C_METHODS:
                cell = [r for r in all_rows
                        if r.get('distribution') == dist_name
                        and r.get('attack_name') == atk_name
                        and r.get('method') == method
                        and r.get('status') == 'completed']
                if not cell:
                    continue
                accs = np.array([r['final_accuracy'] for r in cell], dtype=float)
                f1s  = np.array([r['f1'] for r in cell], dtype=float)
                ci_lo, ci_hi = _ci_95(f1s)
                summary.append({
                    'distribution': dist_name,
                    'attack_name':  atk_name,
                    'method':       method,
                    'n_seeds':      len(cell),
                    'mean_acc':     float(accs.mean()),
                    'std_acc':      float(accs.std()),
                    'mean_f1':      float(f1s.mean()),
                    'std_f1':       float(f1s.std()),
                    'f1_ci95_low':  float(ci_lo),
                    'f1_ci95_high': float(ci_hi),
                })
    sum_csv = os.path.join(out_dir, 'r3c_headline_summary.csv')
    _save_csv(summary, sum_csv,
              ['distribution', 'attack_name', 'method', 'n_seeds',
               'mean_acc', 'std_acc', 'mean_f1', 'std_f1',
               'f1_ci95_low', 'f1_ci95_high'])
    logger.info(f"  Saved summary -> {sum_csv}")

    # Paired stats: OptiGradTrust vs each baseline, per (dist, attack) cell
    stats_rows = []
    for dist_name, _ in R3C_DISTRIBUTIONS:
        for atk_name, _, _ in R3C_ATTACKS:
            # Collect matched-seed data for each method
            data = {}
            for method in R3C_METHODS:
                pool = [r for r in all_rows
                        if r.get('distribution') == dist_name
                        and r.get('attack_name') == atk_name
                        and r.get('method') == method
                        and r.get('status') == 'completed']
                by_seed = {r['seed']: r for r in pool}
                data[method] = by_seed

            ref = 'optigradtrust'
            if ref not in data or len(data[ref]) == 0:
                continue
            shared_seeds = set(data[ref].keys())
            for m in R3C_METHODS:
                if m == ref:
                    continue
                seeds_m = set(data[m].keys())
                paired = sorted(list(shared_seeds & seeds_m))
                if len(paired) < 2:
                    continue
                x = np.array([data[ref][s]['final_accuracy'] for s in paired])
                y = np.array([data[m][s]['final_accuracy']   for s in paired])
                p_val, cohens_d = _paired_stats(x, y)
                stats_rows.append({
                    'distribution': dist_name,
                    'attack_name':  atk_name,
                    'baseline':     m,
                    'n_pairs':      len(paired),
                    'mean_opti':    float(x.mean()),
                    'mean_baseline': float(y.mean()),
                    'mean_delta':   float((x - y).mean()),
                    'wilcoxon_p':   float(p_val) if p_val is not None else float('nan'),
                    'cohens_d':     float(cohens_d) if cohens_d is not None else float('nan'),
                })
    stats_csv = os.path.join(out_dir, 'r3c_paired_stats.csv')
    _save_csv(stats_rows, stats_csv,
              ['distribution', 'attack_name', 'baseline', 'n_pairs',
               'mean_opti', 'mean_baseline', 'mean_delta',
               'wilcoxon_p', 'cohens_d'])
    logger.info(f"  Saved paired stats -> {stats_csv}")

    # Pretty-print summary
    logger.info('\n' + '=' * 100)
    logger.info(f"{'Dist':<12} {'Attack':<12} {'Method':<18} {'n':>3} "
                f"{'Acc (%)':>14} {'F1 (%)':>14} {'F1 95% CI':>20}")
    logger.info('-' * 100)
    for s in summary:
        acc_s = f"{s['mean_acc']*100:.2f} ± {s['std_acc']*100:.2f}"
        f1_s  = f"{s['mean_f1']*100:.2f} ± {s['std_f1']*100:.2f}"
        ci_s  = f"[{s['f1_ci95_low']*100:.2f}, {s['f1_ci95_high']*100:.2f}]"
        logger.info(f"{s['distribution']:<12} {s['attack_name']:<12} "
                    f"{s['method']:<18} {s['n_seeds']:>3} {acc_s:>14} "
                    f"{f1_s:>14} {ci_s:>20}")
    logger.info('=' * 100)
    logger.info('R3-C complete.')
    return all_rows


def _ci_95(arr: np.ndarray) -> Tuple[float, float]:
    """95% CI via t-distribution (small-sample safe) using scipy if present."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    mean = arr.mean()
    sem = arr.std(ddof=1) / np.sqrt(arr.size)
    try:
        from scipy import stats as spstats
        t = spstats.t.ppf(0.975, df=arr.size - 1)
    except Exception:
        # z≈1.96 fallback
        t = 1.96
    return (float(mean - t * sem), float(mean + t * sem))


def _paired_stats(x: np.ndarray, y: np.ndarray):
    """Return (wilcoxon p-value, Cohen's d_z for paired diff)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    diff = x - y
    try:
        from scipy import stats as spstats
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(spstats.wilcoxon(x, y, zero_method='wilcox').pvalue)
    except Exception:
        p = None
    if diff.std(ddof=1) > 0:
        d = float(diff.mean() / diff.std(ddof=1))
    else:
        d = 0.0
    return p, d


# ===========================================================================
# R3-D  Adaptive attack (VAE-aware PGD) vs static scaling
# ===========================================================================

def run_r3d(seeds: List[int], dry_run: bool = False):
    logger.info('=' * 70)
    logger.info('R3-D  ADAPTIVE (VAE-AWARE) ATTACKER vs STATIC SCALING')
    logger.info('  Alzheimer | 10 clients | 30% malicious | 25 rounds')
    logger.info(f'  Seeds: {seeds}   dry_run={dry_run}')
    logger.info('=' * 70)

    out_dir = os.path.join(RESULTS_DIR, 'r3d_adaptive')
    os.makedirs(out_dir, exist_ok=True)
    num_rounds = 2 if dry_run else 25
    all_rows = []

    # Condition 1: static scaling ×20 (reference point)
    # Condition 2: adaptive VAE attack
    conditions = [
        ('static_scaling_x20', 'scaling_attack',      {'scaling_factor': 20.0}),
        ('adaptive_vae',       'adaptive_vae_attack', {'scaling_factor': 20.0,
                                                       'stealth_ratio':  0.5,
                                                       'adaptive_steps': 20,
                                                       'adaptive_step_size': 0.05,
                                                       'alpha_vae': 1.0,
                                                       'beta_dir':  0.5}),
    ]

    for cond_name, atk_type, atk_params in conditions:
        for seed in seeds:
            logger.info(f"  [{cond_name}]  seed={seed}")
            res, _ = _build_and_train(
                method_name='optigradtrust', seed=seed, num_rounds=num_rounds,
                attack_type=atk_type, attack_params=atk_params,
                malicious_fraction=0.3,
            )
            res['condition'] = cond_name
            all_rows.append(res)

    fields = ['condition', 'method', 'seed', 'attack_type',
              'final_accuracy', 'precision', 'recall', 'f1',
              'tp', 'fp', 'fn', 'tn', 'total_time_s', 'status']
    csv_path = os.path.join(out_dir, 'r3d_adaptive.csv')
    _save_csv(all_rows, csv_path, fields)
    _save_json(all_rows, os.path.join(out_dir, 'r3d_adaptive_full.json'))
    logger.info(f"  Saved CSV -> {csv_path}")

    # Summary + paired stats: adaptive vs static
    summary = []
    for cond, _, _ in conditions:
        sel = [r for r in all_rows if r.get('condition') == cond and r.get('status') == 'completed']
        if not sel:
            continue
        accs = np.array([r['final_accuracy'] for r in sel])
        f1s  = np.array([r['f1'] for r in sel])
        summary.append({
            'condition': cond,
            'n_seeds':   len(sel),
            'mean_acc':  float(accs.mean()),
            'std_acc':   float(accs.std()),
            'mean_f1':   float(f1s.mean()),
            'std_f1':    float(f1s.std()),
        })
    _save_csv(summary, os.path.join(out_dir, 'r3d_summary.csv'),
              ['condition', 'n_seeds', 'mean_acc', 'std_acc', 'mean_f1', 'std_f1'])

    # paired acc drop: static - adaptive
    try:
        by_seed = {'static_scaling_x20': {}, 'adaptive_vae': {}}
        for r in all_rows:
            if r.get('status') == 'completed':
                by_seed.setdefault(r['condition'], {})[r['seed']] = r
        shared = sorted(set(by_seed['static_scaling_x20']) & set(by_seed['adaptive_vae']))
        if shared:
            x = np.array([by_seed['static_scaling_x20'][s]['final_accuracy'] for s in shared])
            y = np.array([by_seed['adaptive_vae'][s]['final_accuracy']        for s in shared])
            p, d = _paired_stats(x, y)
            logger.info(f"\n  Paired (static - adaptive): n={len(shared)}, "
                        f"mean_delta={(x-y).mean()*100:+.2f}pp, "
                        f"Wilcoxon p={p}, Cohen's d={d:.3f}")
    except Exception as _e:
        logger.warning(f"  Paired analysis failed ({_e})")

    logger.info('\n' + '=' * 60)
    logger.info(f"{'Condition':<22} {'n':>3} {'Acc (%)':>16} {'F1 (%)':>16}")
    logger.info('-' * 60)
    for s in summary:
        acc_s = f"{s['mean_acc']*100:.2f} ± {s['std_acc']*100:.2f}"
        f1_s  = f"{s['mean_f1']*100:.2f} ± {s['std_f1']*100:.2f}"
        logger.info(f"{s['condition']:<22} {s['n_seeds']:>3} {acc_s:>16} {f1_s:>16}")
    logger.info('=' * 60)
    logger.info('R3-D complete.')
    return all_rows


# ===========================================================================
# R3-E  RL vs rule-based ablation under dynamic attack
# ===========================================================================

R3E_METHODS = [
    ('optigradtrust',           'full_RL'),
    ('optigradtrust_rulebased', 'rule_based'),
    ('optigradtrust_norl',      'no_RL'),
]


def run_r3e(seeds: List[int], dry_run: bool = False):
    logger.info('=' * 70)
    logger.info('R3-E  RL vs RULE-BASED AGGREGATOR ABLATION')
    logger.info('  Alzheimer | 10 clients | 40% malicious | 25 rounds')
    logger.info('  Dynamic attack: phase-transition at round 13 (scaling x20)')
    logger.info(f'  Seeds: {seeds}   dry_run={dry_run}')
    logger.info('=' * 70)

    out_dir = os.path.join(RESULTS_DIR, 'r3e_rl_vs_rulebased')
    os.makedirs(out_dir, exist_ok=True)
    num_rounds = 2 if dry_run else 25
    all_rows = []

    for method_name, label in R3E_METHODS:
        for seed in seeds:
            logger.info(f"  [{label}]  seed={seed}")
            res, rm = _build_and_train(
                method_name=method_name, seed=seed, num_rounds=num_rounds,
                attack_type='scaling_attack',
                attack_params={'scaling_factor': 20.0},
                malicious_fraction=0.4,
                dynamic_schedule='phase_transition',
                phase_transition_round=12,  # 0-indexed -> round 13
            )
            res['label'] = label

            # Per-round detection F1 for pre/post attack analysis
            pre_f1s, post_f1s = [], []
            for r_idx, rd in rm.items():
                det = rd.get('detection_results', {})
                tp = det.get('true_positives',  0)
                fp = det.get('false_positives', 0)
                fn = det.get('false_negatives', 0)
                pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_r = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
                if int(r_idx) < 12:
                    pre_f1s.append(f1_r)
                else:
                    post_f1s.append(f1_r)
            res['mean_f1_pre_attack']  = float(np.mean(pre_f1s)) if pre_f1s else 0.0
            res['mean_f1_post_attack'] = float(np.mean(post_f1s)) if post_f1s else 0.0

            all_rows.append(res)

    fields = ['label', 'method', 'seed',
              'final_accuracy', 'precision', 'recall', 'f1',
              'mean_f1_pre_attack', 'mean_f1_post_attack',
              'total_time_s', 'status']
    csv_path = os.path.join(out_dir, 'r3e_rl_vs_rulebased.csv')
    _save_csv(all_rows, csv_path, fields)
    _save_json(all_rows, os.path.join(out_dir, 'r3e_full.json'))
    logger.info(f"  Saved CSV -> {csv_path}")

    # Summary
    summary = []
    for _, label in R3E_METHODS:
        sel = [r for r in all_rows if r.get('label') == label and r.get('status') == 'completed']
        if not sel:
            continue
        accs = np.array([r['final_accuracy'] for r in sel])
        f1s  = np.array([r['f1'] for r in sel])
        f1_post = np.array([r.get('mean_f1_post_attack', 0.0) for r in sel])
        summary.append({
            'label':       label,
            'n_seeds':     len(sel),
            'mean_acc':    float(accs.mean()),
            'std_acc':     float(accs.std()),
            'mean_f1':     float(f1s.mean()),
            'std_f1':      float(f1s.std()),
            'mean_f1_post': float(f1_post.mean()),
            'std_f1_post':  float(f1_post.std()),
        })
    _save_csv(summary, os.path.join(out_dir, 'r3e_summary.csv'),
              ['label', 'n_seeds', 'mean_acc', 'std_acc',
               'mean_f1', 'std_f1', 'mean_f1_post', 'std_f1_post'])

    logger.info('\n' + '=' * 80)
    logger.info(f"{'Variant':<16} {'n':>3} {'Acc (%)':>16} "
                f"{'F1 overall (%)':>18} {'F1 post-atk (%)':>20}")
    logger.info('-' * 80)
    for s in summary:
        logger.info(f"{s['label']:<16} {s['n_seeds']:>3} "
                    f"{s['mean_acc']*100:>8.2f} ± {s['std_acc']*100:>5.2f}   "
                    f"{s['mean_f1']*100:>8.2f} ± {s['std_f1']*100:>5.2f}   "
                    f"{s['mean_f1_post']*100:>8.2f} ± {s['std_f1_post']*100:>5.2f}")
    logger.info('=' * 80)
    logger.info('R3-E complete.')
    return all_rows


# ===========================================================================
# R3-F  Additional baselines: RFA + SignGuard
# ===========================================================================

R3F_METHODS = ['rfa', 'signguard']


def run_r3f(seeds: List[int], dry_run: bool = False):
    logger.info('=' * 70)
    logger.info('R3-F  ADDITIONAL BASELINES (RFA + SignGuard)')
    logger.info('  Alzheimer | 10 clients | 30% scaling x10 | 25 rounds')
    logger.info(f'  Seeds: {seeds}   dry_run={dry_run}')
    logger.info('=' * 70)

    out_dir = os.path.join(RESULTS_DIR, 'r3f_baselines')
    os.makedirs(out_dir, exist_ok=True)
    num_rounds = 2 if dry_run else 25
    all_rows = []

    for method in R3F_METHODS:
        for seed in seeds:
            logger.info(f"  method={method}  seed={seed}")
            res, _ = _build_and_train(
                method_name=method, seed=seed, num_rounds=num_rounds,
                attack_type='scaling_attack',
                attack_params={'scaling_factor': 10.0},
                malicious_fraction=0.3,
            )
            all_rows.append(res)

    fields = ['method', 'seed', 'attack_type',
              'final_accuracy', 'precision', 'recall', 'f1',
              'tp', 'fp', 'fn', 'tn', 'total_time_s', 'status']
    csv_path = os.path.join(out_dir, 'r3f_baselines.csv')
    _save_csv(all_rows, csv_path, fields)
    _save_json(all_rows, os.path.join(out_dir, 'r3f_full.json'))
    logger.info(f"  Saved CSV -> {csv_path}")

    # Summary
    summary = []
    for method in R3F_METHODS:
        sel = [r for r in all_rows if r.get('method') == method and r.get('status') == 'completed']
        if not sel:
            continue
        accs = np.array([r['final_accuracy'] for r in sel])
        f1s  = np.array([r['f1'] for r in sel])
        summary.append({
            'method':    method,
            'n_seeds':   len(sel),
            'mean_acc':  float(accs.mean()),
            'std_acc':   float(accs.std()),
            'mean_f1':   float(f1s.mean()),
            'std_f1':    float(f1s.std()),
        })
    _save_csv(summary, os.path.join(out_dir, 'r3f_summary.csv'),
              ['method', 'n_seeds', 'mean_acc', 'std_acc', 'mean_f1', 'std_f1'])

    logger.info('\n' + '=' * 60)
    logger.info(f"{'Method':<14} {'n':>3} {'Acc (%)':>16} {'F1 (%)':>16}")
    logger.info('-' * 60)
    for s in summary:
        acc_s = f"{s['mean_acc']*100:.2f} ± {s['std_acc']*100:.2f}"
        f1_s  = f"{s['mean_f1']*100:.2f} ± {s['std_f1']*100:.2f}"
        logger.info(f"{s['method']:<14} {s['n_seeds']:>3} {acc_s:>16} {f1_s:>16}")
    logger.info('=' * 60)
    logger.info('R3-F complete.')
    return all_rows


# ===========================================================================
# Main dispatch
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run R3 revision experiments for OptiGradTrust',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--experiment', '-e', required=True,
        choices=['r3a', 'r3b', 'r3c', 'r3d', 'r3e', 'r3f', 'all'],
        help='Which R3 experiment to run',
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Short 2-round sanity run for each config')
    args = parser.parse_args()

    seeds = args.seeds
    dry = args.dry_run
    exp = args.experiment

    logger.info('#' * 74)
    logger.info(f"#  R3 REVISION RUNNER  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info(f"#  experiment={exp}  seeds={seeds}  dry_run={dry}")
    logger.info(f"#  results dir: {RESULTS_DIR}")
    logger.info(f"#  log file:    {LOG_FILE}")
    logger.info('#' * 74)

    t0 = time.time()
    try:
        if exp == 'r3a' or exp == 'all':
            run_r3a(seeds, dry_run=dry)
        if exp == 'r3b' or exp == 'all':
            run_r3b(seeds=[seeds[0]] if exp == 'r3b' and len(seeds) > 1 else seeds,
                    dry_run=dry)
        if exp == 'r3c' or exp == 'all':
            run_r3c(seeds, dry_run=dry)
        if exp == 'r3d' or exp == 'all':
            run_r3d(seeds, dry_run=dry)
        if exp == 'r3e' or exp == 'all':
            run_r3e(seeds, dry_run=dry)
        if exp == 'r3f' or exp == 'all':
            run_r3f(seeds, dry_run=dry)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"UNCAUGHT ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info('\n' + '#' * 74)
    logger.info(f"#  ALL DONE.  elapsed={elapsed/3600:.2f} h  "
                f"log: {LOG_FILE}")
    if dry:
        logger.info("#  DRY-RUN COMPLETE")
    logger.info('#' * 74)


if __name__ == '__main__':
    main()
