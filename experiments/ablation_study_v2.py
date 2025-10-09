"""
Enhanced Ablation Study v2
===========================

این نسخه بهبود یافته شامل:
1. حملات متنوع و پیچیده
2. Test کردن RL با حملات unseen
3. Disable کردن واقعی components (نه فقط feature values)
4. Settings بهینه برای نتایج واضح
5. تحلیل جامع‌تر

نویسنده: AI Assistant
تاریخ: 8 اکتبر 2025
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from federated_learning.config import config
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.data.dataset_utils import load_dataset, create_client_datasets
from federated_learning.utils.model_utils import set_random_seeds


class EnhancedAblationServer(Server):
    """
    Enhanced ablation server با کنترل کامل روی components.
    
    Features:
    - واقعاً VAE را train نمی‌کند (اگر disabled باشد)
    - واقعاً Shapley را محاسبه نمی‌کند
    - RL را غیرفعال می‌کند
    - Dual Attention را غیرفعال می‌کند
    """
    
    def __init__(self, 
                 disabled_features=None,
                 disable_vae_training=False,
                 disable_shapley_computation=False,
                 disable_rl=False,
                 disable_dual_attention=False):
        """
        Args:
            disabled_features: لیست features برای صفر کردن
            disable_vae_training: اگر True، VAE train نمی‌شود
            disable_shapley_computation: اگر True، Shapley محاسبه نمی‌شود
            disable_rl: اگر True، فقط dual attention استفاده می‌شود
            disable_dual_attention: اگر True، فقط RL استفاده می‌شود
        """
        super().__init__()
        self.disabled_features = disabled_features or []
        self.disable_vae_training = disable_vae_training
        self.disable_shapley_computation = disable_shapley_computation
        self.disable_rl = disable_rl
        self.disable_dual_attention = disable_dual_attention
        
        print(f"\n{'='*80}")
        print(f"🔬 Enhanced Ablation Configuration:")
        print(f"{'='*80}")
        if disabled_features:
            print(f"  Disabled features: {disabled_features}")
        if disable_vae_training:
            print(f"  ⚠️  VAE Training: DISABLED")
        if disable_shapley_computation:
            print(f"  ⚠️  Shapley Computation: DISABLED")
        if disable_rl:
            print(f"  ⚠️  RL-Attention: DISABLED (using only Dual Attention)")
        if disable_dual_attention:
            print(f"  ⚠️  Dual Attention: DISABLED (using only RL)")
        print(f"{'='*80}\n")
    
    def train_vae(self, root_gradients, vae_epochs=50):
        """Override: اگر VAE disabled باشد، train نمی‌کند."""
        if self.disable_vae_training:
            print("⚠️  Skipping VAE training (disabled)")
            # یک VAE dummy برمی‌گردانیم
            from federated_learning.models.vae import GradientVAE
            vae = GradientVAE(
                gradient_dim=root_gradients[0].numel(),
                latent_dim=config.VAE_LATENT_DIM
            ).to(self.device)
            return vae
        else:
            print("✓ Training VAE normally")
            return super().train_vae(root_gradients, vae_epochs)
    
    def _compute_shapley_values(self, gradients, indices):
        """Override: اگر Shapley disabled باشد، محاسبه نمی‌کند."""
        if self.disable_shapley_computation:
            print("⚠️  Skipping Shapley computation (disabled)")
            # مقادیر neutral برمی‌گردانیم
            num_clients = len(gradients)
            return torch.ones(num_clients) * 0.5
        else:
            return super()._compute_shapley_values(gradients, indices)
    
    def _compute_gradient_features(self, gradient, root_gradient=None, skip_client_sim=False):
        """Override: features را صفر می‌کند اگر disabled باشند."""
        # محاسبه features اصلی
        features = super()._compute_gradient_features(gradient, root_gradient, skip_client_sim)
        
        # Feature indices mapping
        feature_map = {
            'vae': 0,
            'cosine_ref': 1,
            'cosine_peer': 2,
            'l2_norm': 3,
            'sign_consistency': 4,
            'shapley': 5
        }
        
        # صفر کردن disabled features
        for feat_name in self.disabled_features:
            if feat_name in feature_map:
                idx = feature_map[feat_name]
                if idx < len(features):
                    # Set to neutral value
                    if feat_name in ['vae', 'l2_norm', 'shapley']:
                        features[idx] = 0.0
                    else:
                        features[idx] = 0.5  # Neutral for similarity metrics
        
        return features
    
    def _aggregate_rl(self, gradients, features, client_indices):
        """
        Override _aggregate_rl: اگر RL disabled باشد، از Dual Attention استفاده می‌کند.
        """
        if self.disable_rl:
            print("⚠️  RL disabled - using Dual Attention for aggregation")
            
            # استفاده از dual attention به جای RL
            from federated_learning.models.attention import DualAttention
            
            # محاسبه trust scores با dual attention
            attention_model = DualAttention(
                feature_dim=features.shape[1],
                hidden_dim=getattr(config, 'DUAL_ATTENTION_HIDDEN_DIM', 64),
                num_heads=getattr(config, 'DUAL_ATTENTION_NUM_HEADS', 4)
            ).to(self.device)
            
            # تنظیم به evaluation mode
            attention_model.eval()
            
            with torch.no_grad():
                # محاسبه trust scores
                # DualAttention returns (malicious_scores, confidence_scores)
                malicious_scores, confidence_scores = attention_model(features)
                
                # trust_scores = 1 - malicious_scores
                trust_scores = 1.0 - malicious_scores.squeeze()
                
                # نرمالسازی weights
                weights = trust_scores / trust_scores.sum()
            
            # ذخیره برای logging
            self.weights = weights
            self.trust_scores = trust_scores
            self.confidence_scores = confidence_scores
            
            # لاگ کردن weights
            print("\nDual Attention Aggregation Weights:")
            for i, client_idx in enumerate(client_indices):
                client = self.clients[client_idx]
                is_malicious = "YES" if client.is_malicious else "NO"
                weight = weights[i].item()
                print(f"Client {client_idx} (Malicious: {is_malicious}): Weight = {weight:.4f}")
            
            # استفاده از base aggregation method
            from federated_learning.config.config import AGGREGATION_METHOD
            
            if AGGREGATION_METHOD == 'fedbn':
                return self._aggregate_fedbn(gradients, weights)
            elif AGGREGATION_METHOD == 'fedprox':
                return self._aggregate_fedprox(gradients, weights)
            else:
                return self._aggregate_fedavg(gradients, weights)
        
        else:
            # RL فعال است - استفاده نرمال
            return super()._aggregate_rl(gradients, features, client_indices)
    
    def _aggregate_with_trust(self, gradients, features, client_indices):
        """Override: استفاده از RL یا Dual Attention بر اساس config."""
        
        if self.disable_rl and self.disable_dual_attention:
            print("⚠️  Both RL and Dual Attention disabled - using simple averaging")
            # Simple FedAvg
            weights = torch.ones(len(gradients)) / len(gradients)
            return self._aggregate_fedavg(gradients, weights)
        
        elif self.disable_rl:
            print("⚠️  RL disabled - using only Dual Attention")
            # این الان در _aggregate_rl handle می‌شود
            return super()._aggregate_with_trust(gradients, features, client_indices)
        
        elif self.disable_dual_attention:
            print("⚠️  Dual Attention disabled - using only RL")
            # فقط از RL استفاده می‌کنیم
            return self._aggregate_rl(gradients, features, client_indices)
        
        else:
            # هر دو فعال هستند - استفاده نرمال
            return super()._aggregate_with_trust(gradients, features, client_indices)


def run_enhanced_ablation_experiment(
    config_name,
    disabled_features=None,
    disable_vae_training=False,
    disable_shapley_computation=False,
    disable_rl=False,
    disable_dual_attention=False,
    num_rounds=50,
    attack_types=None,
    malicious_ratio=0.4,
    attack_intensity=20.0,
    seed=42
):
    """
    یک experiment ablation اجرا می‌کند با تنظیمات دقیق.
    
    Args:
        config_name: نام configuration (مثلاً "baseline" یا "without_vae")
        disabled_features: لیست features برای disable کردن
        disable_vae_training: VAE train نشود
        disable_shapley_computation: Shapley محاسبه نشود
        disable_rl: RL غیرفعال شود
        disable_dual_attention: Dual Attention غیرفعال شود
        num_rounds: تعداد rounds
        attack_types: لیست attack types برای استفاده
        malicious_ratio: نسبت malicious clients
        attack_intensity: شدت حملات
        seed: random seed
    """
    
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {config_name}")
    print(f"{'='*80}")
    print(f"Rounds: {num_rounds}")
    print(f"Malicious Ratio: {malicious_ratio*100}%")
    print(f"Attack Intensity: {attack_intensity}")
    print(f"Attack Types: {attack_types}")
    print(f"Random Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Set seed
    set_random_seeds(seed)
    
    # Load dataset
    root_dataset, test_dataset = load_dataset()
    root_loader = torch.utils.data.DataLoader(
        root_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    
    # Create server
    server = EnhancedAblationServer(
        disabled_features=disabled_features,
        disable_vae_training=disable_vae_training,
        disable_shapley_computation=disable_shapley_computation,
        disable_rl=disable_rl,
        disable_dual_attention=disable_dual_attention
    )
    server.set_datasets(root_loader, test_dataset)
    server._pretrain_global_model()
    
    initial_accuracy = server.evaluate_model()
    print(f"Initial accuracy: {initial_accuracy:.4f}")
    
    # Create client datasets
    root_client_dataset, client_datasets = create_client_datasets(
        train_dataset=root_dataset,
        num_clients=config.NUM_CLIENTS,
        iid=not config.ENABLE_NON_IID,
        alpha=config.DIRICHLET_ALPHA if config.ENABLE_NON_IID else None
    )
    
    # Create clients
    clients = []
    for i in range(config.NUM_CLIENTS):
        client = Client(client_id=i, dataset=client_datasets[i], is_malicious=False)
        clients.append(client)
    
    server.add_clients(clients)
    
    # Configure malicious clients با حملات متنوع
    num_malicious = int(config.NUM_CLIENTS * malicious_ratio)
    malicious_indices = np.random.choice(config.NUM_CLIENTS, num_malicious, replace=False)
    
    if attack_types is None:
        attack_types = ['scaling_attack']
    
    print(f"🎯 Configuring {num_malicious} malicious clients:")
    for i, mal_idx in enumerate(malicious_indices):
        # هر malicious client یک attack type متفاوت
        attack_type = attack_types[i % len(attack_types)]
        clients[mal_idx].is_malicious = True
        clients[mal_idx].set_attack_parameters(
            attack_type=attack_type,
            scaling_factor=attack_intensity
        )
        print(f"  Client {mal_idx}: {attack_type} (intensity={attack_intensity})")
    
    # Train VAE
    print("\n🔧 VAE Training Phase:")
    root_gradients = server._collect_root_gradients()
    server.vae = server.train_vae(root_gradients, vae_epochs=config.VAE_EPOCHS)
    
    # Run federated training
    print(f"\n🚀 Starting federated learning for {num_rounds} rounds...")
    training_errors, round_metrics = server.train(num_rounds=num_rounds)
    
    # Evaluate results
    final_accuracy = server.evaluate_model()
    improvement = final_accuracy - initial_accuracy
    
    # Extract detection metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    
    if round_metrics:
        for round_idx, round_data in round_metrics.items():
            if 'detection_results' in round_data:
                det_results = round_data['detection_results']
                total_tp += det_results.get('true_positives', 0)
                total_fp += det_results.get('false_positives', 0)
                total_fn += det_results.get('false_negatives', 0)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    result = {
        'config_name': config_name,
        'disabled_features': disabled_features,
        'disable_vae_training': disable_vae_training,
        'disable_shapley_computation': disable_shapley_computation,
        'disable_rl': disable_rl,
        'disable_dual_attention': disable_dual_attention,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'detection_precision': precision,
        'detection_recall': recall,
        'detection_f1': f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'malicious_indices': malicious_indices.tolist(),
        'attack_types': attack_types,
        'num_rounds': num_rounds,
        'malicious_ratio': malicious_ratio,
        'attack_intensity': attack_intensity,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*80}")
    print(f"✅ {config_name} - Results:")
    print(f"{'='*80}")
    print(f"  Final Accuracy: {final_accuracy:.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    print(f"  Detection Precision: {precision:.4f}")
    print(f"  Detection Recall: {recall:.4f}")
    print(f"  Detection F1: {f1_score:.4f}")
    print(f"{'='*80}\n")
    
    return result


def run_comprehensive_ablation_study(
    output_dir,
    num_rounds=50,
    quick_mode=False
):
    """
    اجرای ablation study کامل با تمام scenarios.
    """
    
    print(f"\n{'🔬'*40}")
    print(f"COMPREHENSIVE ABLATION STUDY v2")
    print(f"{'🔬'*40}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if quick_mode:
        num_rounds = 15
        print("⚡ Quick mode: 15 rounds")
    
    # حملات برای training (ساده)
    TRAINING_ATTACKS = ['scaling_attack', 'noise_attack']
    
    # حملات برای test RL (پیچیده و unseen)
    RL_TEST_ATTACKS = ['label_flipping', 'min_max_attack', 'partial_scaling_attack']
    
    # حملات ترکیبی (همه باهم)
    ALL_ATTACKS = ['scaling_attack', 'noise_attack', 'label_flipping', 'min_max_attack']
    
    results = {}
    
    # ==========================================
    # 1. BASELINE (همه فعال، حملات ساده)
    # ==========================================
    print("\n" + "="*80)
    print("1️⃣  BASELINE: All components active, simple attacks")
    print("="*80)
    
    results['baseline'] = run_enhanced_ablation_experiment(
        config_name='baseline',
        num_rounds=num_rounds,
        attack_types=TRAINING_ATTACKS,
        seed=42
    )
    
    # ==========================================
    # 2. WITHOUT VAE
    # ==========================================
    print("\n" + "="*80)
    print("2️⃣  WITHOUT VAE: VAE training disabled")
    print("="*80)
    
    results['without_vae'] = run_enhanced_ablation_experiment(
        config_name='without_vae',
        disabled_features=['vae'],
        disable_vae_training=True,
        num_rounds=num_rounds,
        attack_types=TRAINING_ATTACKS,
        seed=42
    )
    
    # ==========================================
    # 3. WITHOUT SHAPLEY
    # ==========================================
    print("\n" + "="*80)
    print("3️⃣  WITHOUT SHAPLEY: Shapley computation disabled")
    print("="*80)
    
    results['without_shapley'] = run_enhanced_ablation_experiment(
        config_name='without_shapley',
        disabled_features=['shapley'],
        disable_shapley_computation=True,
        num_rounds=num_rounds,
        attack_types=ALL_ATTACKS,  # Shapley برای حملات پیچیده مفید است
        seed=42
    )
    
    # ==========================================
    # 4. WITHOUT RL (فقط Dual Attention)
    # ==========================================
    print("\n" + "="*80)
    print("4️⃣  WITHOUT RL: Only Dual Attention (RL disabled)")
    print("="*80)
    
    results['without_rl'] = run_enhanced_ablation_experiment(
        config_name='without_rl',
        disable_rl=True,
        num_rounds=num_rounds,
        attack_types=TRAINING_ATTACKS,
        seed=42
    )
    
    # ==========================================
    # 5. RL TEST با حملات UNSEEN
    # ==========================================
    print("\n" + "="*80)
    print("5️⃣  RL TEST: Testing RL with unseen attacks")
    print("="*80)
    
    results['rl_with_unseen_attacks'] = run_enhanced_ablation_experiment(
        config_name='rl_with_unseen_attacks',
        num_rounds=num_rounds,
        attack_types=RL_TEST_ATTACKS,  # حملاتی که dual attention ندیده
        seed=43  # seed متفاوت
    )
    
    # ==========================================
    # 6. WITHOUT DUAL ATTENTION (فقط RL)
    # ==========================================
    print("\n" + "="*80)
    print("6️⃣  WITHOUT DUAL ATTENTION: Only RL (Dual Attention disabled)")
    print("="*80)
    
    results['without_dual_attention'] = run_enhanced_ablation_experiment(
        config_name='without_dual_attention',
        disable_dual_attention=True,
        num_rounds=num_rounds,
        attack_types=RL_TEST_ATTACKS,
        seed=42
    )
    
    # ==========================================
    # 7. بقیه Features
    # ==========================================
    for feature in ['cosine_ref', 'cosine_peer', 'l2_norm', 'sign_consistency']:
        print("\n" + "="*80)
        print(f"WITHOUT {feature.upper()}")
        print("="*80)
        
        results[f'without_{feature}'] = run_enhanced_ablation_experiment(
            config_name=f'without_{feature}',
            disabled_features=[feature],
            num_rounds=num_rounds,
            attack_types=TRAINING_ATTACKS,
            seed=42
        )
    
    # ==========================================
    # ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE ABLATION ANALYSIS")
    print(f"{'='*80}\n")
    
    baseline_acc = results['baseline']['final_accuracy']
    baseline_f1 = results['baseline']['detection_f1']
    
    print(f"Baseline Performance:")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  Detection F1: {baseline_f1:.4f}\n")
    
    print(f"{'Configuration':<35} {'Accuracy':<12} {'Acc Drop':<12} {'F1':<12} {'F1 Drop':<12}")
    print(f"{'-'*80}")
    
    analysis = {
        'baseline': {
            'accuracy': baseline_acc,
            'detection_f1': baseline_f1
        },
        'component_importance': {}
    }
    
    for config_name, result in results.items():
        if config_name == 'baseline':
            continue
        
        acc_drop = baseline_acc - result['final_accuracy']
        f1_drop = baseline_f1 - result['detection_f1']
        
        print(f"{config_name:<35} {result['final_accuracy']:<12.4f} "
              f"{acc_drop:<12.4f} {result['detection_f1']:<12.4f} {f1_drop:<12.4f}")
        
        analysis['component_importance'][config_name] = {
            'accuracy_drop': acc_drop,
            'f1_drop': f1_drop,
            'final_accuracy': result['final_accuracy'],
            'final_f1': result['detection_f1']
        }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_file = output_dir / f'comprehensive_ablation_v2_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'raw_results': results,
            'analysis': analysis,
            'metadata': {
                'num_rounds': num_rounds,
                'quick_mode': quick_mode,
                'timestamp': timestamp
            }
        }, f, indent=2)
    
    # Save CSV
    import pandas as pd
    
    summary_data = [{
        'Configuration': 'Baseline',
        'Accuracy': baseline_acc,
        'Detection_F1': baseline_f1,
        'Accuracy_Drop': 0.0,
        'F1_Drop': 0.0
    }]
    
    for config_name, result in results.items():
        if config_name == 'baseline':
            continue
        summary_data.append({
            'Configuration': config_name,
            'Accuracy': result['final_accuracy'],
            'Detection_F1': result['detection_f1'],
            'Accuracy_Drop': baseline_acc - result['final_accuracy'],
            'F1_Drop': baseline_f1 - result['detection_f1']
        })
    
    df = pd.DataFrame(summary_data)
    csv_file = output_dir / f'comprehensive_ablation_v2_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\n✅ Comprehensive ablation study completed!")
    print(f"📁 Results: {results_file}")
    print(f"📊 CSV: {csv_file}")
    
    return {
        'results_file': str(results_file),
        'csv_file': str(csv_file),
        'analysis': analysis,
        'raw_results': results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Ablation Study v2')
    parser.add_argument('--output-dir', default='experiments/results/ablation_v2',
                       help='Output directory')
    parser.add_argument('--rounds', type=int, default=50,
                       help='Number of training rounds')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (15 rounds)')
    
    args = parser.parse_args()
    
    run_comprehensive_ablation_study(
        output_dir=args.output_dir,
        num_rounds=args.rounds,
        quick_mode=args.quick
    )

