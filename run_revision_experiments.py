#!/usr/bin/env python3
"""
=============================================================================
OptiGradTrust -- Revision Experiments
=============================================================================

Experiments for the paper revision:

  Experiment 1: Per-component timing breakdown (original config, Alzheimer MRI)
  Experiment 2: Combined component ablation   (ablation / strengthened config)
  Table 9:      Single-component ablation + FedAvg  (Alzheimer, ablation config)
  Table 8:      Single-component ablation           (OASIS, strengthened config)

This script is *additive*: it imports from the existing codebase and does NOT
modify any existing file.

USAGE:
    python run_revision_experiments.py --experiment timing     # Exp 1 only
    python run_revision_experiments.py --experiment ablation   # Exp 2 only
    python run_revision_experiments.py --experiment table9     # Table 9 (Alzheimer)
    python run_revision_experiments.py --experiment table8     # Table 8 (OASIS)
    python run_revision_experiments.py --experiment tables     # Table 8 + Table 9
    python run_revision_experiments.py --experiment all        # Everything
    python run_revision_experiments.py --experiment table9 --dry-run

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
import argparse
import traceback
import csv
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from run_all_experiments import (
    run_single_experiment, set_all_seeds, configure_for_dataset,
    compute_statistics, Logger
)

# ── Output directory ────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'revision2')
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(RESULTS_DIR, f'revision2_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
logger = Logger(LOG_FILE)

# ── Seeds ───────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]

# =============================================================================
# EXPERIMENT 1 — PER-COMPONENT TIMING BREAKDOWN
# =============================================================================
# Original config: Alzheimer MRI, IID, 10 clients, 30 % malicious, scaling ×10,
#                  25 rounds, seed = 42


def run_timing_experiment(num_rounds: int = 25, seed: int = 42, dry_run: bool = False):
    """
    Instrument the OptiGradTrust pipeline to measure wall-time for each of the
    10 trust-scoring components.  Also runs FedAvg as a timing baseline.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: PER-COMPONENT TIMING BREAKDOWN")
    logger.info("=" * 70)

    if dry_run:
        num_rounds = 2
        logger.info("[DRY-RUN] Limiting to 2 rounds")

    # ── helpers ──────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()

    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    def _timed(fn, *a, **kw):
        """Run *fn* and return (result, elapsed_ms)."""
        _sync()
        t0 = time.perf_counter()
        result = fn(*a, **kw)
        _sync()
        return result, (time.perf_counter() - t0) * 1000.0

    # ── configure (original config) ──────────────────────────────────────
    set_all_seeds(seed)
    import federated_learning.config.config as config

    cfg = configure_for_dataset('ALZHEIMER', num_clients=10,
                                non_iid_config={'enable': False, 'type': 'iid', 'alpha': None},
                                aggregation_method='fedbn_fedprox')
    cfg.GLOBAL_EPOCHS = num_rounds
    cfg.RANDOM_SEED = seed
    cfg.FRACTION_MALICIOUS = 0.3
    cfg.NUM_MALICIOUS = 3
    cfg.SCALING_FACTOR = 10.0

    from federated_learning.training.server import Server
    from federated_learning.training.client import Client
    from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
    from federated_learning.utils.model_utils import set_random_seeds
    from federated_learning.training.training_utils import test

    set_random_seeds(seed)

    # ── data ─────────────────────────────────────────────────────────────
    train_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # ── server ───────────────────────────────────────────────────────────
    server = Server()
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    initial_accuracy = server.evaluate_model()
    logger.info(f"Initial accuracy: {initial_accuracy:.4f}")

    # ── clients ──────────────────────────────────────────────────────────
    root_dataset, client_datasets = create_client_datasets(
        train_dataset=train_dataset, num_clients=10, iid=True, alpha=None)

    clients = []
    malicious_indices = np.random.choice(10, 3, replace=False)
    for i in range(10):
        is_mal = i in malicious_indices
        c = Client(client_id=i, dataset=client_datasets[i], is_malicious=is_mal)
        if is_mal:
            c.set_attack_parameters(
                attack_type='scaling_attack',
                scaling_factor=config.SCALING_FACTOR,
                partial_percent=config.PARTIAL_SCALING_PERCENT,
                noise_factor=config.NOISE_FACTOR,
                flip_probability=config.FLIP_PROBABILITY)
        clients.append(c)
    server.add_clients(clients)

    # ── VAE training ─────────────────────────────────────────────────────
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)

    # ── test loader ──────────────────────────────────────────────────────
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    server.test_loader = test_loader

    # ── timing storage ───────────────────────────────────────────────────
    component_names = [
        'local_training', 'vae_fingerprinting', 'cosine_similarity',
        'peer_consensus', 'l2_norm', 'sign_consistency',
        'shapley_values', 'dual_attention', 'ddqn_rl_scoring',
        'trust_weighted_aggregation'
    ]
    per_round_timings: List[Dict] = []

    # ── instrumented training loop ───────────────────────────────────────
    logger.info(f"Starting instrumented training for {num_rounds} rounds ...")

    for round_idx in range(num_rounds):
        logger.info(f"--- Round {round_idx + 1}/{num_rounds} ---")
        timings: Dict[str, float] = {c: 0.0 for c in component_names}

        # (a) select clients — same logic as Server._select_clients
        from federated_learning.config.config import CLIENT_SELECTION_RATIO
        n_sel = max(1, int(len(server.clients) * CLIENT_SELECTION_RATIO))
        selected_clients = np.random.choice(len(server.clients), n_sel, replace=False).tolist()

        # (b) local training
        all_gradients = []
        client_indices = []
        _sync()
        t_train_start = time.perf_counter()

        for cidx in selected_clients:
            client = server.clients[cidx]
            try:
                result = client.train(server.global_model, round_idx)
                if isinstance(result, tuple) and len(result) == 2:
                    gradient, _ = result
                else:
                    gradient = result
                if gradient is None:
                    continue
                gradient = gradient.to(server.device)
                all_gradients.append(gradient)
                client_indices.append(cidx)
            except Exception:
                traceback.print_exc()

        _sync()
        timings['local_training'] = (time.perf_counter() - t_train_start) * 1000.0

        if not all_gradients:
            logger.warning(f"Round {round_idx+1}: no gradients collected, skipping")
            continue

        # Prepare root gradient reference
        root_grad = server.root_gradients[0] if server.root_gradients else None

        # (c-f) per-client feature computation (VAE, cosine, L2, sign)
        num_cl = len(all_gradients)
        from federated_learning.config.config import ENABLE_SHAPLEY
        feature_dim = 6 if ENABLE_SHAPLEY else 5
        features = torch.zeros((num_cl, feature_dim), device=server.device)

        for i, grad in enumerate(all_gradients):
            # VAE fingerprinting
            _sync(); t0 = time.perf_counter()
            if server.vae is not None:
                try:
                    with torch.no_grad():
                        recon, _, _ = server.vae(grad.unsqueeze(0))
                        recon_err = F.mse_loss(recon.squeeze(0), grad)
                        features[i, 0] = torch.clamp(recon_err / (recon_err + 1.0), 0.0, 1.0)
                except Exception:
                    features[i, 0] = 0.5
            else:
                features[i, 0] = 0.5
            _sync()
            timings['vae_fingerprinting'] += (time.perf_counter() - t0) * 1000.0

            # Cosine similarity to root
            _sync(); t0 = time.perf_counter()
            if root_grad is not None:
                rg = root_grad.to(server.device) if root_grad.device != server.device else root_grad
                cos_sim = F.cosine_similarity(grad.unsqueeze(0), rg.unsqueeze(0))
                features[i, 1] = (cos_sim + 1.0) / 2.0
            else:
                features[i, 1] = 0.5
            _sync()
            timings['cosine_similarity'] += (time.perf_counter() - t0) * 1000.0

            # L2 norm
            _sync(); t0 = time.perf_counter()
            gn = torch.norm(grad).item()
            if hasattr(server, 'root_gradients') and server.root_gradients:
                root_norms = [torch.norm(g).item() for g in server.root_gradients]
                norm_ref = max(np.median(root_norms) * 3.0, 50.0)
            else:
                norm_ref = 500.0
            features[i, 3] = min(np.log1p(gn) / np.log1p(norm_ref), 1.0)
            _sync()
            timings['l2_norm'] += (time.perf_counter() - t0) * 1000.0

            # Sign consistency
            _sync(); t0 = time.perf_counter()
            if root_grad is not None:
                rg = root_grad.to(server.device) if root_grad.device != server.device else root_grad
                features[i, 4] = (torch.sign(grad) == torch.sign(rg)).float().mean()
            else:
                features[i, 4] = 0.5
            _sync()
            timings['sign_consistency'] += (time.perf_counter() - t0) * 1000.0

        # Peer consensus (pairwise cosine)
        _sync(); t0 = time.perf_counter()
        cos_mat = torch.zeros((num_cl, num_cl), device=server.device)
        for i in range(num_cl):
            for j in range(num_cl):
                if i != j:
                    cos_mat[i, j] = (F.cosine_similarity(
                        all_gradients[i].unsqueeze(0),
                        all_gradients[j].unsqueeze(0)) + 1) / 2
        for i in range(num_cl):
            features[i, 2] = cos_mat[i].sum() / max(num_cl - 1, 1)
        _sync()
        timings['peer_consensus'] = (time.perf_counter() - t0) * 1000.0

        # (g) Shapley values
        _sync(); t0 = time.perf_counter()
        shapley_values = server._compute_shapley_values(all_gradients, client_indices)
        _sync()
        timings['shapley_values'] = (time.perf_counter() - t0) * 1000.0

        if shapley_values is not None and features.size(1) >= 6:
            for i, val in enumerate(shapley_values):
                features[i, 5] = val

        # (h) Dual Attention
        _sync(); t0 = time.perf_counter()
        global_context = torch.mean(features, dim=0, keepdim=True)
        try:
            server.dual_attention.eval()
            with torch.no_grad():
                trust_tensor, confidence = server.dual_attention(features, global_context)
            client_malicious_scores = trust_tensor.cpu().numpy().tolist()
            client_trust_scores = [1.0 - s for s in client_malicious_scores]
            server.trust_scores = torch.tensor(client_trust_scores, device=server.device)
            server.malicious_scores = trust_tensor
            server.confidence_scores = confidence
        except Exception:
            traceback.print_exc()
            client_trust_scores = [1.0] * num_cl
            server.trust_scores = torch.ones(num_cl, device=server.device)
            server.confidence_scores = torch.ones(num_cl, device=server.device)

        # get_gradient_weights
        try:
            weights_tensor, malicious_idx = server.dual_attention.get_gradient_weights(
                features, server.malicious_scores, server.confidence_scores)
            weights = weights_tensor.cpu().numpy().tolist()
            server.dual_attention_weights = weights_tensor
        except Exception:
            traceback.print_exc()
            weights = [max(0.01, 1.0 - s) for s in client_trust_scores]
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
            else:
                weights = [1.0 / num_cl] * num_cl

        server.weights = torch.tensor(weights, device=server.device)
        _sync()
        timings['dual_attention'] = (time.perf_counter() - t0) * 1000.0

        # (i) DDQN / RL scoring
        from federated_learning.config.config import (
            AGGREGATION_METHOD, RL_AGGREGATION_METHOD, RL_WARMUP_ROUNDS, RL_RAMP_UP_ROUNDS
        )
        warmup = RL_WARMUP_ROUNDS if 'RL_WARMUP_ROUNDS' in dir(config) else 5
        ramp = RL_RAMP_UP_ROUNDS if 'RL_RAMP_UP_ROUNDS' in dir(config) else 10
        use_rl = (RL_AGGREGATION_METHOD in ('rl_actor_critic', 'hybrid') and
                  round_idx >= warmup)

        _sync(); t0 = time.perf_counter()
        server.current_round_gradients = all_gradients.copy()
        if use_rl:
            aggregated_gradient = server._aggregate_rl(all_gradients, features, client_indices)
            if RL_AGGREGATION_METHOD == 'hybrid' and round_idx < warmup + ramp:
                blend = (round_idx - warmup) / ramp
                da_w = getattr(server, 'dual_attention_weights', server.weights)
                da_grad = server._aggregate_fedavg(all_gradients, da_w)
                aggregated_gradient = blend * aggregated_gradient + (1 - blend) * da_grad
        else:
            aggregated_gradient = None
        _sync()
        timings['ddqn_rl_scoring'] = (time.perf_counter() - t0) * 1000.0

        # (j) Trust-weighted aggregation
        _sync(); t0 = time.perf_counter()
        if aggregated_gradient is None:
            aggregated_gradient = server._aggregate_fedavg(all_gradients, server.weights)
        old_state = copy.deepcopy(server.global_model)
        server._update_global_model(aggregated_gradient)
        _sync()
        timings['trust_weighted_aggregation'] = (time.perf_counter() - t0) * 1000.0

        # RL feedback update (matches original loop)
        if use_rl:
            post_acc, post_err = test(server.global_model, test_loader)
            server._update_rl_model(round_idx, post_err, post_acc, features, client_indices)

        # evaluate for logging
        rd_acc, rd_err = test(server.global_model, test_loader)
        logger.info(f"  Round {round_idx+1} accuracy: {rd_acc:.4f}")

        # record
        row = {'round': round_idx + 1}
        row.update(timings)
        per_round_timings.append(row)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── summarize (rounds 6-25, or all if dry-run) ───────────────────────
    skip = 0 if dry_run else 5
    analysis_rows = per_round_timings[skip:]

    summary: Dict[str, Dict] = {}
    for comp in component_names:
        vals = [r[comp] for r in analysis_rows]
        summary[comp] = {
            'mean_ms': float(np.mean(vals)),
            'std_ms': float(np.std(vals)),
            'median_ms': float(np.median(vals)),
        }

    total_mean = sum(v['mean_ms'] for v in summary.values())
    for comp in component_names:
        summary[comp]['pct'] = summary[comp]['mean_ms'] / total_mean * 100 if total_mean > 0 else 0

    # Compute overhead: trust components = everything except local_training
    local_training_mean = summary['local_training']['mean_ms']
    trust_overhead_ms = total_mean - local_training_mean
    overhead_pct = (trust_overhead_ms / local_training_mean * 100
                    if local_training_mean > 0 else float('inf'))

    # ── FedAvg baseline timing ───────────────────────────────────────────
    # Run FedAvg under the same settings to get an independent wall-clock
    # comparison. Use separate config to avoid state contamination.
    logger.info("\nRunning FedAvg baseline for timing comparison ...")
    set_all_seeds(seed)
    fedavg_result = run_single_experiment(
        dataset='ALZHEIMER', attack_type='scaling_attack', seed=seed,
        num_clients=10,
        non_iid_config={'enable': False, 'type': 'iid', 'alpha': None},
        aggregation_method='fedavg', epochs=num_rounds,
        malicious_ratio=0.3, scaling_factor=10.0)
    fedavg_total_time_ms = fedavg_result.get('training_time', 0) * 1000.0
    fedavg_ms_per_round = fedavg_total_time_ms / max(num_rounds, 1)
    fedavg_accuracy = fedavg_result.get('final_accuracy', 0)

    # ── save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'timing_breakdown.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round'] + component_names)
        writer.writeheader()
        for row in per_round_timings:
            writer.writerow(row)
    logger.info(f"Saved per-round timings -> {csv_path}")

    # ── print table ──────────────────────────────────────────────────────
    instrumented_acc = rd_acc if 'rd_acc' in locals() else None

    logger.info("\n" + "=" * 70)
    logger.info("TIMING BREAKDOWN SUMMARY (averaged over rounds %d-%d)" %
                (skip + 1, num_rounds))
    logger.info("%-35s %12s %10s" % ("Component", "Time (ms)", "% Total"))
    logger.info("-" * 60)
    for comp in component_names:
        s = summary[comp]
        logger.info("%-35s %9.2f ± %-6.2f %7.1f%%" %
                    (comp, s['mean_ms'], s['std_ms'], s['pct']))
    logger.info("-" * 60)
    logger.info("%-35s %9.2f %10s" % ("OptiGradTrust TOTAL", total_mean, "100%"))
    logger.info("%-35s %9.2f" % ("  of which: local training", local_training_mean))
    logger.info("%-35s %9.2f %9.1f%%" % ("  of which: trust overhead",
                                          trust_overhead_ms, overhead_pct))
    logger.info("")
    logger.info("%-35s %9.2f" % ("FedAvg round time (wall-clock)", fedavg_ms_per_round))
    logger.info("%-35s %9s %9.1f%%" % ("Trust overhead vs local training", "", overhead_pct))
    logger.info("=" * 70)
    if instrumented_acc is not None:
        logger.info(f"Instrumented loop final accuracy: {instrumented_acc:.4f}")
    logger.info(f"FedAvg baseline accuracy:         {fedavg_accuracy:.4f}")

    # ── save JSON summary ────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, 'timing_breakdown_summary.json')
    json_obj = {
        'config': {
            'dataset': 'ALZHEIMER', 'distribution': 'IID', 'num_clients': 10,
            'malicious_ratio': 0.3, 'scaling_factor': 10.0,
            'num_rounds': num_rounds, 'seed': seed,
            'warmup_skipped': skip,
        },
        'per_component': summary,
        'optigradtrust_total_ms': total_mean,
        'local_training_ms': local_training_mean,
        'trust_overhead_ms': trust_overhead_ms,
        'trust_overhead_pct': overhead_pct,
        'fedavg_wall_clock_ms_per_round': fedavg_ms_per_round,
        'fedavg_accuracy': fedavg_accuracy,
        'instrumented_accuracy': instrumented_acc,
    }
    with open(json_path, 'w') as f:
        json.dump(json_obj, f, indent=2)
    logger.info(f"Saved JSON summary -> {json_path}")

    return json_obj


# =============================================================================
# EXPERIMENT 2 — COMBINED COMPONENT ABLATION
# =============================================================================

COMBINED_ABLATION_CONFIGS = [
    # A: w/o VAE + Shapley
    {'name': 'no_vae_no_shapley',
     'vae': False, 'shapley': False, 'dual_attention': True, 'rl': True},
    # B: w/o VAE + Shapley + RL
    {'name': 'no_vae_no_shapley_no_rl',
     'vae': False, 'shapley': False, 'dual_attention': True, 'rl': False},
    # C: Cosine + L2 only  (peer consensus & sign consistency zeroed out)
    {'name': 'cosine_l2_only',
     'vae': False, 'shapley': False, 'dual_attention': True, 'rl': False,
     'zero_features': [2, 4]},  # indices: 2=peer consensus, 4=sign consistency
]


def _patch_feature_zeroing(zero_indices):
    """
    Context manager that monkey-patches Server._compute_all_gradient_features
    to zero out specific feature columns (e.g. peer consensus, sign consistency).
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from federated_learning.training.server import Server
        original = Server._compute_all_gradient_features

        def patched(self, client_gradients):
            features = original(self, client_gradients)
            if isinstance(features, torch.Tensor):
                for idx in zero_indices:
                    if idx < features.size(1):
                        features[:, idx] = 0.5  # neutral default
            return features

        Server._compute_all_gradient_features = patched
        try:
            yield
        finally:
            Server._compute_all_gradient_features = original

    return _ctx()


def _run_experiment_with_config(cfg, **kwargs):
    """Run a single experiment, applying feature zeroing if requested."""
    zero = cfg.get('zero_features')
    ablation = {k: v for k, v in cfg.items() if k not in ('zero_features',)}
    if zero:
        with _patch_feature_zeroing(zero):
            return run_single_experiment(ablation_config=ablation, **kwargs)
    else:
        return run_single_experiment(ablation_config=ablation, **kwargs)


def run_combined_ablation(dry_run: bool = False):
    """
    Run the 3 combined-removal ablation configurations on both Alzheimer MRI
    (ablation config, single seed) and OASIS (strengthened config, 5 seeds).
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: COMBINED COMPONENT ABLATION")
    logger.info("=" * 70)

    epochs_override = 2 if dry_run else None
    seeds_oasis = [42] if dry_run else SEEDS
    all_results: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # ALZHEIMER MRI — ablation config (matches paper Table 9)
    # 40 % malicious, scaling ×20, Dirichlet α=0.1, 25 rds, seed=42
    # NOTE: attack_type='scaling_attack' only applies gradient scaling;
    #       noise_factor and flip_probability have no effect for this type.
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n>>> ALZHEIMER MRI (ablation config) <<<")
    alz_epochs = epochs_override or 25
    for cfg in COMBINED_ABLATION_CONFIGS:
        logger.info(f"\n  Config: {cfg['name']}")
        result = _run_experiment_with_config(cfg,
            dataset='ALZHEIMER',
            attack_type='scaling_attack',
            seed=42,
            num_clients=10,
            non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
            aggregation_method='fedbn_fedprox',
            epochs=alz_epochs,
            malicious_ratio=0.4,
            scaling_factor=20.0,
        )
        result['experiment'] = 'combined_ablation'
        result['config_name'] = cfg['name']
        all_results.append(result)

        if result['status'] == 'completed':
            det = result.get('detection', {})
            logger.success(
                f"    Acc={result['final_accuracy']:.4f}  "
                f"P={det.get('precision',0):.4f}  R={det.get('recall',0):.4f}  "
                f"F1={det.get('f1_score',0):.4f}")
        else:
            logger.error(f"    FAILED: {result.get('error', 'unknown')}")

    # ──────────────────────────────────────────────────────────────────────
    # OASIS -- strengthened config (n = 5 seeds)
    # 40 % malicious, scaling x30, IID, 8 rounds
    # NOTE: noise_factor/flip_probability removed -- no effect for scaling_attack
    # ──────────────────────────────────────────────────────────────────────
    logger.info("\n>>> OASIS (strengthened config, n=%d seeds) <<<" % len(seeds_oasis))
    oasis_epochs = epochs_override or 8
    for cfg in COMBINED_ABLATION_CONFIGS:
        logger.info(f"\n  Config: {cfg['name']}")
        seed_results = []
        for seed in seeds_oasis:
            result = _run_experiment_with_config(cfg,
                dataset='OASIS',
                attack_type='scaling_attack',
                seed=seed,
                num_clients=10,
                non_iid_config={'enable': False, 'type': 'iid', 'alpha': None},
                aggregation_method='fedbn_fedprox',
                epochs=oasis_epochs,
                malicious_ratio=0.4,
                scaling_factor=30.0,
            )
            result['experiment'] = 'combined_ablation'
            result['config_name'] = cfg['name']
            all_results.append(result)

            if result['status'] == 'completed':
                seed_results.append(result)
                det = result.get('detection', {})
                logger.success(
                    f"    Seed {seed}: Acc={result['final_accuracy']:.4f}  "
                    f"F1={det.get('f1_score',0):.4f}")
            else:
                logger.error(f"    Seed {seed}: FAILED")

        if seed_results:
            accs = [r['final_accuracy'] for r in seed_results]
            logger.info(f"    Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # ── save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, 'combined_ablation_results.csv')
    csv_rows = []
    for r in all_results:
        det = r.get('detection', {})
        csv_rows.append({
            'dataset': r.get('dataset'),
            'config_name': r.get('config_name'),
            'seed': r.get('seed'),
            'accuracy': r.get('final_accuracy'),
            'precision': det.get('precision'),
            'recall': det.get('recall'),
            'f1_score': det.get('f1_score'),
            'tp': det.get('true_positives'),
            'fp': det.get('false_positives'),
            'fn': det.get('false_negatives'),
            'tn': det.get('true_negatives'),
            'status': r.get('status'),
            'time_s': r.get('total_time'),
        })

    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Saved raw results -> {csv_path}")

    # ── save JSON ────────────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, 'combined_ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved JSON results -> {json_path}")

    # ── print summary tables ─────────────────────────────────────────────
    _print_ablation_summary(all_results)

    return all_results


def _print_ablation_summary(all_results: List[Dict]):
    """Pretty-print the ablation result tables."""

    # Alzheimer
    alz = [r for r in all_results if r.get('dataset') == 'ALZHEIMER' and r.get('status') == 'completed']
    if alz:
        logger.info("\n" + "=" * 80)
        logger.info("ALZHEIMER MRI (ablation config, seed=42)")
        logger.info("%-30s %10s %10s %10s %10s" %
                    ("Configuration", "Acc(%)", "Precision", "Recall", "F1"))
        logger.info("-" * 72)
        for r in alz:
            det = r.get('detection', {})
            logger.info("%-30s %9.2f %9.2f %9.2f %9.2f" % (
                r['config_name'],
                r['final_accuracy'] * 100,
                det.get('precision', 0) * 100,
                det.get('recall', 0) * 100,
                det.get('f1_score', 0) * 100))
        logger.info("=" * 80)

    # OASIS (aggregate over seeds)
    oasis = [r for r in all_results if r.get('dataset') == 'OASIS' and r.get('status') == 'completed']
    if oasis:
        logger.info("\n" + "=" * 80)
        logger.info("OASIS (strengthened config, n=5 seeds)")
        logger.info("%-30s %15s %15s %15s %15s" %
                    ("Configuration", "Acc(%)", "Precision", "Recall", "F1"))
        logger.info("-" * 92)
        from itertools import groupby
        oasis_sorted = sorted(oasis, key=lambda x: x['config_name'])
        for cfg_name, group in groupby(oasis_sorted, key=lambda x: x['config_name']):
            items = list(group)
            accs = [r['final_accuracy'] * 100 for r in items]
            precs = [r['detection']['precision'] * 100 for r in items]
            recs = [r['detection']['recall'] * 100 for r in items]
            f1s = [r['detection']['f1_score'] * 100 for r in items]
            logger.info("%-30s %6.2f±%-6.2f %6.2f±%-6.2f %6.2f±%-6.2f %6.2f±%-6.2f" % (
                cfg_name,
                np.mean(accs), np.std(accs),
                np.mean(precs), np.std(precs),
                np.mean(recs), np.std(recs),
                np.mean(f1s), np.std(f1s)))
        logger.info("=" * 80)


# =============================================================================
# TABLE 9 — SINGLE-COMPONENT ABLATION + FedAvg (ALZHEIMER)
# =============================================================================
# Ablation config: 40% malicious, scaling x20, Dirichlet alpha=0.1,
#                  25 rounds, seed=42, fedbn_fedprox aggregation.

TABLE9_CONFIGS = [
    {'name': 'full',             'vae': True,  'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_vae',           'vae': False, 'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_shapley',       'vae': True,  'shapley': False, 'dual_attention': True,  'rl': True},
    {'name': 'no_rl',            'vae': True,  'shapley': True,  'dual_attention': True,  'rl': False},
    {'name': 'no_dual_attention','vae': True,  'shapley': True,  'dual_attention': False, 'rl': True},
]


def run_table9_alzheimer(dry_run: bool = False):
    """
    Run Table 9: single-component ablation (5 configs) + FedAvg baseline
    on Alzheimer MRI with the ablation attack config, seed=42.
    """
    logger.info("=" * 70)
    logger.info("TABLE 9: SINGLE-COMPONENT ABLATION + FedAvg (ALZHEIMER)")
    logger.info("=" * 70)

    epochs = 2 if dry_run else 25
    all_results: List[Dict] = []

    # --- 5 single-component ablation configs ---
    for cfg in TABLE9_CONFIGS:
        logger.info(f"\n  Config: {cfg['name']}")
        result = run_single_experiment(
            ablation_config=cfg,
            dataset='ALZHEIMER',
            attack_type='scaling_attack',
            seed=42,
            num_clients=10,
            non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
            aggregation_method='fedbn_fedprox',
            epochs=epochs,
            malicious_ratio=0.4,
            scaling_factor=20.0,
        )
        result['experiment'] = 'table9'
        result['config_name'] = cfg['name']
        all_results.append(result)

        if result['status'] == 'completed':
            det = result.get('detection', {})
            logger.success(
                f"    Acc={result['final_accuracy']:.4f}  "
                f"P={det.get('precision',0):.4f}  R={det.get('recall',0):.4f}  "
                f"F1={det.get('f1_score',0):.4f}")
        else:
            logger.error(f"    FAILED: {result.get('error', 'unknown')}")

    # --- FedAvg baseline (no defense components) ---
    logger.info(f"\n  Config: fedavg_baseline")
    fedavg_result = run_single_experiment(
        dataset='ALZHEIMER',
        attack_type='scaling_attack',
        seed=42,
        num_clients=10,
        non_iid_config={'enable': True, 'type': 'dirichlet', 'alpha': 0.1},
        aggregation_method='fedavg',
        epochs=epochs,
        malicious_ratio=0.4,
        scaling_factor=20.0,
    )
    fedavg_result['experiment'] = 'table9'
    fedavg_result['config_name'] = 'fedavg_baseline'
    all_results.append(fedavg_result)

    if fedavg_result['status'] == 'completed':
        det = fedavg_result.get('detection', {})
        logger.success(
            f"    Acc={fedavg_result['final_accuracy']:.4f}  "
            f"P={det.get('precision',0):.4f}  R={det.get('recall',0):.4f}  "
            f"F1={det.get('f1_score',0):.4f}")
    else:
        logger.error(f"    FAILED: {fedavg_result.get('error', 'unknown')}")

    # --- Save results ---
    csv_path = os.path.join(RESULTS_DIR, 'table9_results.csv')
    csv_rows = []
    for r in all_results:
        det = r.get('detection', {})
        csv_rows.append({
            'config_name': r.get('config_name'),
            'accuracy': r.get('final_accuracy'),
            'precision': det.get('precision'),
            'recall': det.get('recall'),
            'f1_score': det.get('f1_score'),
            'tp': det.get('true_positives'),
            'fp': det.get('false_positives'),
            'fn': det.get('false_negatives'),
            'tn': det.get('true_negatives'),
            'status': r.get('status'),
            'total_time': r.get('total_time'),
        })
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Saved CSV -> {csv_path}")

    json_path = os.path.join(RESULTS_DIR, 'table9_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved JSON -> {json_path}")

    # --- Summary table ---
    completed = [r for r in all_results if r.get('status') == 'completed']
    if completed:
        logger.info("\n" + "=" * 80)
        logger.info("TABLE 9: ALZHEIMER MRI ABLATION (40%% malicious, scaling x20, Dirichlet a=0.1)")
        logger.info("%-25s %10s %10s %10s %10s" %
                    ("Configuration", "Acc(%)", "Prec(%)", "Rec(%)", "F1(%)"))
        logger.info("-" * 67)
        for r in completed:
            det = r.get('detection', {})
            logger.info("%-25s %9.2f %9.2f %9.2f %9.2f" % (
                r['config_name'],
                r['final_accuracy'] * 100,
                det.get('precision', 0) * 100,
                det.get('recall', 0) * 100,
                det.get('f1_score', 0) * 100))
        logger.info("=" * 80)

    return all_results


# =============================================================================
# TABLE 8 — SINGLE-COMPONENT ABLATION (OASIS)
# =============================================================================
# Strengthened config: scaling x30, 40% malicious, IID, 8 rounds, n=5 seeds.

TABLE8_CONFIGS = [
    {'name': 'full',             'vae': True,  'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_vae',           'vae': False, 'shapley': True,  'dual_attention': True,  'rl': True},
    {'name': 'no_rl',            'vae': True,  'shapley': True,  'dual_attention': True,  'rl': False},
    {'name': 'no_dual_attention','vae': True,  'shapley': True,  'dual_attention': False, 'rl': True},
]


def run_table8_oasis(dry_run: bool = False):
    """
    Run Table 8: single-component ablation (4 configs x 5 seeds)
    on OASIS with the strengthened config (scaling, 40% malicious, IID).
    """
    logger.info("=" * 70)
    logger.info("TABLE 8: SINGLE-COMPONENT ABLATION (OASIS)")
    logger.info("=" * 70)

    epochs = 2 if dry_run else 8
    seeds = [42] if dry_run else SEEDS
    all_results: List[Dict] = []

    for cfg in TABLE8_CONFIGS:
        logger.info(f"\n  Config: {cfg['name']}")
        seed_results = []
        for seed in seeds:
            result = run_single_experiment(
                ablation_config=cfg,
                dataset='OASIS',
                attack_type='scaling_attack',
                seed=seed,
                num_clients=10,
                non_iid_config={'enable': False, 'type': 'iid', 'alpha': None},
                aggregation_method='fedbn_fedprox',
                epochs=epochs,
                malicious_ratio=0.4,
                scaling_factor=30.0,
            )
            result['experiment'] = 'table8'
            result['config_name'] = cfg['name']
            all_results.append(result)

            if result['status'] == 'completed':
                seed_results.append(result)
                det = result.get('detection', {})
                logger.success(
                    f"    Seed {seed}: Acc={result['final_accuracy']:.4f}  "
                    f"F1={det.get('f1_score',0):.4f}")
            else:
                logger.error(f"    Seed {seed}: FAILED")

        if seed_results:
            accs = [r['final_accuracy'] for r in seed_results]
            logger.info(f"    Mean Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # --- Save results ---
    csv_path = os.path.join(RESULTS_DIR, 'table8_results.csv')
    csv_rows = []
    for r in all_results:
        det = r.get('detection', {})
        csv_rows.append({
            'config_name': r.get('config_name'),
            'seed': r.get('seed'),
            'accuracy': r.get('final_accuracy'),
            'precision': det.get('precision'),
            'recall': det.get('recall'),
            'f1_score': det.get('f1_score'),
            'tp': det.get('true_positives'),
            'fp': det.get('false_positives'),
            'fn': det.get('false_negatives'),
            'tn': det.get('true_negatives'),
            'status': r.get('status'),
            'total_time': r.get('total_time'),
        })
    if csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"Saved CSV -> {csv_path}")

    json_path = os.path.join(RESULTS_DIR, 'table8_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved JSON -> {json_path}")

    # --- Summary table ---
    completed = [r for r in all_results if r.get('status') == 'completed']
    if completed:
        logger.info("\n" + "=" * 80)
        logger.info("TABLE 8: OASIS ABLATION (40%% malicious, scaling x30, IID, n=5 seeds)")
        logger.info("%-25s %15s %15s %15s %15s" %
                    ("Configuration", "Acc(%)", "Prec(%)", "Rec(%)", "F1(%)"))
        logger.info("-" * 87)
        from itertools import groupby
        sorted_results = sorted(completed, key=lambda x: x['config_name'])
        for cfg_name, group in groupby(sorted_results, key=lambda x: x['config_name']):
            items = list(group)
            accs = [r['final_accuracy'] * 100 for r in items]
            precs = [r['detection']['precision'] * 100 for r in items]
            recs = [r['detection']['recall'] * 100 for r in items]
            f1s = [r['detection']['f1_score'] * 100 for r in items]
            logger.info("%-25s %6.2f+/-%-6.2f %6.2f+/-%-6.2f %6.2f+/-%-6.2f %6.2f+/-%-6.2f" % (
                cfg_name,
                np.mean(accs), np.std(accs),
                np.mean(precs), np.std(precs),
                np.mean(recs), np.std(recs),
                np.mean(f1s), np.std(f1s)))
        logger.info("=" * 80)

    return all_results


# =============================================================================
# COMBINED SUMMARY
# =============================================================================

def generate_revision_summary(timing_result, ablation_results):
    """Write a markdown summary of both experiments."""
    md_path = os.path.join(RESULTS_DIR, 'revision_experiments_summary.md')
    lines = [
        "# Revision Experiments Summary",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]

    if timing_result:
        lines += [
            "## Experiment 1: Per-Component Timing Breakdown",
            "",
            "| Component | Time (ms/round) | %% of Total |",
            "|---|---|---|",
        ]
        for comp in [
            'local_training', 'vae_fingerprinting', 'cosine_similarity',
            'peer_consensus', 'l2_norm', 'sign_consistency',
            'shapley_values', 'dual_attention', 'ddqn_rl_scoring',
            'trust_weighted_aggregation'
        ]:
            s = timing_result['per_component'][comp]
            lines.append("| %s | %.2f ± %.2f | %.1f%% |" %
                        (comp, s['mean_ms'], s['std_ms'], s['pct']))
        lines += [
            "|---|---|---|",
            "| **OptiGradTrust TOTAL** | %.2f | 100%% |" % timing_result['optigradtrust_total_ms'],
            "| **Trust overhead** | %.2f | %.1f%% of local training |" % (
                timing_result['trust_overhead_ms'], timing_result['trust_overhead_pct']),
            "| **FedAvg round time (wall-clock)** | %.2f | |" % timing_result['fedavg_wall_clock_ms_per_round'],
            "",
        ]

    if ablation_results:
        completed = [r for r in ablation_results if r.get('status') == 'completed']

        alz = [r for r in completed if r.get('dataset') == 'ALZHEIMER']
        if alz:
            lines += [
                "## Experiment 2a: Combined Ablation — Alzheimer MRI",
                "",
                "| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |",
                "|---|---|---|---|---|",
            ]
            for r in alz:
                d = r.get('detection', {})
                lines.append("| %s | %.2f | %.2f | %.2f | %.2f |" % (
                    r['config_name'],
                    r['final_accuracy'] * 100,
                    d.get('precision', 0) * 100,
                    d.get('recall', 0) * 100,
                    d.get('f1_score', 0) * 100))
            lines.append("")

        oasis = [r for r in completed if r.get('dataset') == 'OASIS']
        if oasis:
            lines += [
                "## Experiment 2b: Combined Ablation — OASIS",
                "",
                "| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |",
                "|---|---|---|---|---|",
            ]
            from itertools import groupby
            oasis_sorted = sorted(oasis, key=lambda x: x['config_name'])
            for cfg_name, group in groupby(oasis_sorted, key=lambda x: x['config_name']):
                items = list(group)
                acc = np.mean([r['final_accuracy'] * 100 for r in items])
                acc_s = np.std([r['final_accuracy'] * 100 for r in items])
                pr = np.mean([r['detection']['precision'] * 100 for r in items])
                pr_s = np.std([r['detection']['precision'] * 100 for r in items])
                rc = np.mean([r['detection']['recall'] * 100 for r in items])
                rc_s = np.std([r['detection']['recall'] * 100 for r in items])
                f1 = np.mean([r['detection']['f1_score'] * 100 for r in items])
                f1_s = np.std([r['detection']['f1_score'] * 100 for r in items])
                lines.append("| %s | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f | %.2f ± %.2f |" %
                            (cfg_name, acc, acc_s, pr, pr_s, rc, rc_s, f1, f1_s))
            lines.append("")

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Summary written -> {md_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='OptiGradTrust Revision Experiments')
    parser.add_argument('--experiment',
                        choices=['timing', 'ablation', 'table9', 'table8', 'tables', 'all'],
                        default='all', help='Which experiment to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick sanity check (2 rounds, 1 seed)')
    args = parser.parse_args()

    logger.info("OptiGradTrust Revision Experiments")
    logger.info(f"Experiment: {args.experiment}  |  Dry-run: {args.dry_run}")
    logger.info(f"Output dir: {RESULTS_DIR}")

    timing_result = None
    ablation_results = None
    table9_results = None
    table8_results = None

    if args.experiment in ('timing', 'all'):
        timing_result = run_timing_experiment(dry_run=args.dry_run)

    if args.experiment in ('ablation', 'all'):
        ablation_results = run_combined_ablation(dry_run=args.dry_run)

    if args.experiment in ('table9', 'tables', 'all'):
        table9_results = run_table9_alzheimer(dry_run=args.dry_run)

    if args.experiment in ('table8', 'tables', 'all'):
        table8_results = run_table8_oasis(dry_run=args.dry_run)

    generate_revision_summary(timing_result, ablation_results)
    logger.info("\nAll revision experiments complete.")


if __name__ == '__main__':
    main()
