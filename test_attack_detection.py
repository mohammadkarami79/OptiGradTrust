import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import os
from datetime import datetime

from federated_learning.config.config import *
from federated_learning.models.vae import GradientVAE
from federated_learning.attacks.attack_utils import simulate_attack
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.utils.model_utils import get_gradient

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class TestAttackDetection:
    def __init__(self):
        print("\n=== Initializing Attack Detection Test ===")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create server for access to models
        self.server = Server()
        self.vae = self.server.vae
        self.dual_attention = self.server.dual_attention
        
        # Create output directory for plots
        self.output_dir = "test_results/plots"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_attack_gradients(self):
        """Generate honest and various attack gradients for testing"""
        print("\n=== Generating Test Gradients ===")
        
        # Create an honest gradient as baseline
        gradient_dim = GRADIENT_DIMENSION
        honest_gradient = torch.randn(gradient_dim).to(self.device)
        honest_gradient = honest_gradient / torch.norm(honest_gradient)
        
        # Create a set of attack gradients
        attack_types = [
            "scaling",
            "sign_flipping",
            "label_flipping",
            "min_max",
            "min_sum"
        ]
        
        attack_gradients = {}
        for attack_type in attack_types:
            # Simulate each attack type
            attack_gradient = simulate_attack(
                gradient=honest_gradient.clone(),
                attack_type=attack_type
            )
            attack_gradients[attack_type] = attack_gradient
        
        # Store all gradients
        all_gradients = {"honest": honest_gradient}
        all_gradients.update(attack_gradients)
        
        return all_gradients
    
    def test_vae_reconstruction(self, gradients):
        """Test VAE reconstruction of honest and attack gradients"""
        print("\n=== Testing VAE Reconstruction ===")
        
        self.vae.eval()
        results = {}
        
        plt.figure(figsize=(12, 8))
        
        for i, (name, gradient) in enumerate(gradients.items()):
            with torch.no_grad():
                # Get reconstruction
                reconstructed, mu, logvar = self.vae(gradient.unsqueeze(0))
                reconstructed = reconstructed.squeeze(0)
                
                # Calculate reconstruction error
                mse_loss = torch.nn.functional.mse_loss(reconstructed, gradient)
                cosine_sim = torch.nn.functional.cosine_similarity(reconstructed, gradient, dim=0)
                
                # Store results
                results[name] = {
                    'mse_loss': mse_loss.item(),
                    'cosine_sim': cosine_sim.item()
                }
                
                print(f"{name} - MSE: {mse_loss.item():.6f}, Cosine similarity: {cosine_sim.item():.6f}")
                
                # Plot small segment of the gradient for visualization
                segment_size = 50
                start_idx = 0
                
                # Plot
                plt.subplot(len(gradients), 1, i+1)
                plt.plot(gradient[start_idx:start_idx+segment_size].cpu().numpy(), 'b-', label='Original', alpha=0.7)
                plt.plot(reconstructed[start_idx:start_idx+segment_size].cpu().numpy(), 'r-', label='Reconstructed', alpha=0.7)
                plt.title(f"{name.capitalize()} Gradient")
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "vae_reconstruction.png"))
        plt.close()
        
        # Plot reconstruction errors
        plt.figure(figsize=(10, 6))
        names = list(results.keys())
        mse_values = [results[name]['mse_loss'] for name in names]
        
        plt.bar(range(len(names)), mse_values, color=['green' if name == 'honest' else 'red' for name in names])
        plt.xticks(range(len(names)), [name.capitalize() for name in names], rotation=45)
        plt.title('VAE Reconstruction Error by Gradient Type')
        plt.ylabel('MSE Loss')
        plt.axhline(y=results['honest']['mse_loss'], color='black', linestyle='--', label='Honest threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "reconstruction_errors.png"))
        plt.close()
        
        return results
    
    def test_feature_extraction(self, gradients):
        """Test feature extraction for gradients"""
        print("\n=== Testing Feature Extraction ===")
        
        # Extract features for all gradients
        features = {}
        feature_data = []
        
        for name, gradient in gradients.items():
            # Create dummy list of gradients (all the same for this test)
            dummy_gradients = [gradient.clone()]
            
            # Extract features
            gradient_features = self.server._compute_gradient_features(
                grad=gradient,
                raw_grad=gradient,
                client_gradients=dummy_gradients,
                all_raw_gradients=dummy_gradients
            )
            
            features[name] = gradient_features
            
            # Print features
            print(f"\n{name.capitalize()} features:")
            feature_names = ["Reconstruction", "Root Similarity", "Client Similarity", 
                           "Gradient Norm", "Consistency"]
            if ENABLE_SHAPLEY:
                feature_names.append("Shapley Value")
                
            for i, feat_name in enumerate(feature_names):
                value = gradient_features[i].item()
                feature_data.append({
                    'gradient_type': name,
                    'feature': feat_name,
                    'value': value
                })
                print(f"  {feat_name}: {value:.6f}")
        
        # Plot features as radar chart
        gradient_types = list(features.keys())
        honest_idx = gradient_types.index("honest")
        
        # Prepare radar chart
        feature_names = ["Reconstruction", "Root Similarity", "Client Similarity", 
                        "Gradient Norm", "Consistency"]
        if ENABLE_SHAPLEY:
            feature_names.append("Shapley Value")
            
        num_features = len(feature_names)
        angles = np.linspace(0, 2*np.pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, name in enumerate(gradient_types):
            # Get values and close the loop
            values = features[name].cpu().numpy().tolist()
            values += values[:1]
            
            color = 'green' if name == 'honest' else plt.cm.tab10(i)
            ax.plot(angles, values, color=color, linewidth=2, label=name.capitalize())
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)
        ax.set_title('Gradient Features Comparison', size=20)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_radar.png"))
        plt.close()
        
        # Bar chart for each feature
        plt.figure(figsize=(15, 10))
        
        # Group by feature
        feature_groups = {}
        for entry in feature_data:
            if entry['feature'] not in feature_groups:
                feature_groups[entry['feature']] = []
            feature_groups[entry['feature']].append({
                'gradient_type': entry['gradient_type'],
                'value': entry['value']
            })
        
        # Plot each feature as a subplot
        for i, (feature, data) in enumerate(feature_groups.items()):
            ax = plt.subplot(np.ceil(len(feature_groups)/2), 2, i+1)
            gradient_types = [d['gradient_type'] for d in data]
            values = [d['value'] for d in data]
            
            # Define colors (green for honest, red for others)
            colors = ['green' if gtype == 'honest' else 'red' for gtype in gradient_types]
            
            ax.bar(range(len(gradient_types)), values, color=colors)
            ax.set_xticks(range(len(gradient_types)))
            ax.set_xticklabels([gtype.capitalize() for gtype in gradient_types], rotation=45)
            ax.set_title(f'{feature}')
            ax.set_ylim(0, 1)
            
            # Add honest threshold line
            honest_value = next(d['value'] for d in data if d['gradient_type'] == 'honest')
            ax.axhline(y=honest_value, color='black', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_comparison.png"))
        plt.close()
        
        return features
    
    def test_multi_indicator_detection(self, gradients, features):
        """Test the multi-indicator attack detection system"""
        print("\n=== Testing Multi-Indicator Attack Detection ===")
        
        # Stack features for all gradients
        gradient_types = list(features.keys())
        feature_stack = torch.stack([features[name] for name in gradient_types])
        
        # Get trust scores from dual attention
        self.dual_attention.eval()
        with torch.no_grad():
            trust_scores, confidence_scores = self.dual_attention(feature_stack)
            weights, detected = self.dual_attention.get_gradient_weights(feature_stack)
        
        # Print results
        print("\nDetection results:")
        for i, name in enumerate(gradient_types):
            is_malicious = name != "honest"
            is_detected = i in detected
            
            print(f"{name.capitalize()} - Trust: {trust_scores[i].item():.4f}, " + 
                 f"Weight: {weights[i].item():.4f}, Detected: {is_detected}")
            
            # Check if detection matches ground truth
            if is_malicious == is_detected:
                print("  ✅ Correctly classified")
            else:
                print("  ❌ Incorrectly classified")
        
        # Plot trust scores
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(gradient_types)), trust_scores.cpu().numpy(),
               color=['green' if name == 'honest' else 'red' for name in gradient_types])
        plt.axhline(y=0.5, color='black', linestyle='--', label='Detection threshold')
        plt.xticks(range(len(gradient_types)), [name.capitalize() for name in gradient_types], rotation=45)
        plt.title('Trust Scores by Gradient Type')
        plt.ylabel('Trust Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "trust_scores.png"))
        plt.close()
        
        # Plot weights
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(gradient_types)), weights.cpu().numpy(),
               color=['green' if name == 'honest' else 'red' for name in gradient_types])
        plt.xticks(range(len(gradient_types)), [name.capitalize() for name in gradient_types], rotation=45)
        plt.title('Aggregation Weights by Gradient Type')
        plt.ylabel('Weight')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "weights.png"))
        plt.close()
        
        # Calculate detection metrics
        true_malicious = [i for i, name in enumerate(gradient_types) if name != "honest"]
        detected_malicious = [i for i in detected]
        
        true_positives = len([i for i in detected_malicious if i in true_malicious])
        false_positives = len([i for i in detected_malicious if i not in true_malicious])
        true_negatives = len([i for i in range(len(gradient_types)) if i not in detected_malicious and i not in true_malicious])
        false_negatives = len([i for i in true_malicious if i not in detected_malicious])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(gradient_types)
        
        print("\nDetection metrics:")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return {
            'trust_scores': trust_scores.cpu().numpy(),
            'weights': weights.cpu().numpy(),
            'detected': detected,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            }
        }
    
    def run_all_tests(self):
        """Run all attack detection tests"""
        # Generate test gradients
        gradients = self.generate_attack_gradients()
        
        # Test VAE reconstruction
        vae_results = self.test_vae_reconstruction(gradients)
        
        # Test feature extraction
        features = self.test_feature_extraction(gradients)
        
        # Test multi-indicator detection
        detection_results = self.test_multi_indicator_detection(gradients, features)
        
        print("\n=== Attack Detection Test Summary ===")
        print(f"Attack detection accuracy: {detection_results['metrics']['accuracy']:.4f}")
        print(f"Attack detection precision: {detection_results['metrics']['precision']:.4f}")
        print(f"Attack detection recall: {detection_results['metrics']['recall']:.4f}")
        
        if detection_results['metrics']['accuracy'] >= 0.8:
            print("\n✅ OVERALL TEST PASSED: Attack detection system is working effectively!")
        else:
            print("\n❌ OVERALL TEST FAILED: Attack detection system needs improvement.")
        
        return detection_results

if __name__ == "__main__":
    print(f"=== ATTACK DETECTION SYSTEM TEST ===")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test = TestAttackDetection()
    results = test.run_all_tests() 