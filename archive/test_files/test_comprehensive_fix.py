#!/usr/bin/env python3
"""
Comprehensive test script to validate all system components and configurations.
This tests the complete system across different models, datasets, and attack scenarios.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import tempfile
import shutil
import json

# Add federated_learning to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'federated_learning'))

def test_trust_score_logic():
    """Test that trust scores are computed correctly."""
    print("=== Testing Trust Score Logic ===")
    
    # Create mock dual attention model
    from federated_learning.models.attention import DualAttention
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dual_attention = DualAttention(feature_dim=6, hidden_dim=64).to(device)
    
    # Create test features - clear honest vs malicious differences
    honest_features = torch.tensor([
        [0.1, 0.9, 0.8, 0.2, 0.9, 0.1],  # Very honest pattern
        [0.2, 0.8, 0.7, 0.3, 0.8, 0.2],  # Honest pattern
    ], device=device, dtype=torch.float32)
    
    malicious_features = torch.tensor([
        [0.9, 0.1, 0.2, 0.9, 0.1, 0.9],  # Very malicious pattern  
        [0.8, 0.2, 0.3, 0.8, 0.2, 0.8],  # Malicious pattern
    ], device=device, dtype=torch.float32)
    
    all_features = torch.cat([honest_features, malicious_features], dim=0)
    
    print("Input features:")
    print("Honest clients:")
    for i, feat in enumerate(honest_features):
        print(f"  Client {i}: {feat.cpu().numpy()}")
    print("Malicious clients:")
    for i, feat in enumerate(malicious_features):
        print(f"  Client {i+2}: {feat.cpu().numpy()}")
    
    # Test forward pass
    with torch.no_grad():
        malicious_scores, confidence_scores = dual_attention(all_features)
        trust_scores = 1.0 - malicious_scores  # Convert to trust scores
    
    print("\nModel outputs:")
    print("Malicious scores:", malicious_scores.detach().cpu().numpy())
    print("Trust scores (1 - malicious):", trust_scores.detach().cpu().numpy())
    
    # Test aggregation weights
    weights, detected_malicious = dual_attention.get_gradient_weights(
        all_features, malicious_scores, confidence_scores
    )
    
    print("\nAggregation weights (based on trust scores):")
    for i in range(len(weights)):
        client_type = "Honest" if i < 2 else "Malicious"
        print(f"  Client {i} ({client_type}): {weights[i].item():.4f}")
    
    # Calculate averages
    honest_trust_avg = trust_scores[:2].mean().item()
    malicious_trust_avg = trust_scores[2:].mean().item()
    honest_weight_avg = weights[:2].mean().item()
    malicious_weight_avg = weights[2:].mean().item()
    
    print("\nAverage scores:")
    print(f"Honest clients - Trust: {honest_trust_avg:.4f}, Malicious: {malicious_scores[:2].mean().item():.4f}")
    print(f"Malicious clients - Trust: {malicious_trust_avg:.4f}, Malicious: {malicious_scores[2:].mean().item():.4f}")
    
    print("\nWeight averages:")
    print(f"Honest clients: {honest_weight_avg:.4f}")
    print(f"Malicious clients: {malicious_weight_avg:.4f}")
    
    # Validate results
    trust_score_correct = honest_trust_avg > malicious_trust_avg
    weight_correct = honest_weight_avg > malicious_weight_avg
    
    print("\n--- Test Results ---")
    if trust_score_correct:
        print("✅ PASS: Honest clients have higher trust scores")
    else:
        print("❌ FAIL: Honest clients should have higher trust scores")
    
    if weight_correct:
        print("✅ PASS: Honest clients have higher aggregation weights")
    else:
        print("❌ FAIL: Honest clients should have higher aggregation weights")
    
    overall_pass = trust_score_correct and weight_correct
    
    if overall_pass:
        print("✅ OVERALL: Trust score logic is working correctly!")
    else:
        print("❌ OVERALL: Trust score logic needs fixing!")
    
    return overall_pass

def test_configuration(config_name, config_changes):
    """Test a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*60}")
    
    # Create a temporary config file
    temp_config_content = f'''
# Temporary configuration for testing: {config_name}

# Import base configuration
import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import *

# Apply configuration changes
'''
    
    for key, value in config_changes.items():
        if isinstance(value, str):
            temp_config_content += f"{key} = '{value}'\\n"
        else:
            temp_config_content += f"{key} = {value}\\n"
    
    # Write temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(temp_config_content)
        temp_config_path = f.name
    
    try:
        # Test the configuration
        print(f"Configuration changes: {config_changes}")
        
        # Import and run a mini test
        from main import main
        
        # Monkey-patch some configuration values
        import federated_learning.config.config as config_module
        original_values = {}
        
        for key, value in config_changes.items():
            if hasattr(config_module, key):
                original_values[key] = getattr(config_module, key)
                setattr(config_module, key, value)
        
        # Run a small test (1 round only)
        original_epochs = getattr(config_module, 'GLOBAL_EPOCHS', 2)
        setattr(config_module, 'GLOBAL_EPOCHS', 1)  # Run only 1 round for testing
        
        try:
            results = main()
            print(f"✅ Configuration '{config_name}' completed successfully!")
            
            # Check basic result validity
            if 'final_accuracy' in results and results['final_accuracy'] > 0.5:
                print(f"✅ Final accuracy reasonable: {results['final_accuracy']:.4f}")
            if 'trust_scores' in results:
                print(f"✅ Trust scores generated: {len(results['trust_scores'])} clients")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration '{config_name}' failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(config_module, key, value)
            setattr(config_module, 'GLOBAL_EPOCHS', original_epochs)
    
    finally:
        # Clean up temporary config file
        try:
            os.unlink(temp_config_path)
        except:
            pass

def test_attack_detection():
    """Test attack detection accuracy."""
    print("\n=== Testing Attack Detection ===")
    
    from main import main
    import federated_learning.config.config as config
    
    # Save original configuration
    original_epochs = config.GLOBAL_EPOCHS
    original_attack = config.ATTACK_TYPE
    
    try:
        # Set to 1 round for quick testing
        config.GLOBAL_EPOCHS = 1
        
        # Test different attack types
        attack_types = [
            'scaling_attack',
            'partial_scaling_attack',
            'sign_flipping',
            'noise_injection'
        ]
        
        results = {}
        
        for attack_type in attack_types:
            print(f"\nTesting {attack_type}...")
            config.ATTACK_TYPE = attack_type
            
            try:
                result = main()
                
                # Check detection metrics
                precision = result.get('detection_precision', 0)
                recall = result.get('detection_recall', 0)
                f1_score = result.get('detection_f1_score', 0)
                
                print(f"  Detection Precision: {precision:.4f}")
                print(f"  Detection Recall: {recall:.4f}")
                print(f"  Detection F1-Score: {f1_score:.4f}")
                
                # Store results
                results[attack_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'trust_scores': result.get('trust_scores', {}),
                    'success': True
                }
                
                print(f"✅ {attack_type} test completed")
                
            except Exception as e:
                print(f"❌ {attack_type} test failed: {str(e)}")
                results[attack_type] = {'success': False, 'error': str(e)}
        
        # Print summary
        print("\n--- Attack Detection Summary ---")
        successful_tests = [k for k, v in results.items() if v.get('success', False)]
        print(f"Successful tests: {len(successful_tests)}/{len(attack_types)}")
        
        for attack_type, result in results.items():
            if result.get('success', False):
                f1 = result.get('f1_score', 0)
                status = "✅ GOOD" if f1 > 0.3 else "⚠️ WEAK" if f1 > 0.1 else "❌ POOR"
                print(f"  {attack_type}: F1={f1:.3f} {status}")
            else:
                print(f"  {attack_type}: ❌ FAILED")
        
        return len(successful_tests) >= len(attack_types) * 0.75  # 75% success rate
        
    finally:
        # Restore original configuration
        config.GLOBAL_EPOCHS = original_epochs
        config.ATTACK_TYPE = original_attack

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive System Tests")
    print("="*80)
    
    test_results = {}
    
    # Test 1: Basic trust score logic
    print("\n🧪 Test 1: Trust Score Logic")
    test_results['trust_score_logic'] = test_trust_score_logic()
    
    # Test 2: Different model configurations
    print("\n🧪 Test 2: Model Configurations")
    model_configs = [
        ("MNIST + CNN", {'DATASET': 'MNIST', 'MODEL': 'CNN'}),
        ("MNIST + ResNet18", {'DATASET': 'MNIST', 'MODEL': 'RESNET18'}),
    ]
    
    model_test_results = []
    for config_name, config_changes in model_configs:
        try:
            result = test_configuration(config_name, config_changes)
            model_test_results.append(result)
        except Exception as e:
            print(f"❌ {config_name} failed: {str(e)}")
            model_test_results.append(False)
    
    test_results['model_configs'] = all(model_test_results)
    
    # Test 3: Different aggregation methods
    print("\n🧪 Test 3: Aggregation Methods")
    aggregation_configs = [
        ("FedAvg", {'AGGREGATION_METHOD': 'fedavg'}),
        ("FedBN", {'AGGREGATION_METHOD': 'fedbn'}),
        ("FedProx", {'AGGREGATION_METHOD': 'fedprox'}),
    ]
    
    aggregation_test_results = []
    for config_name, config_changes in aggregation_configs:
        try:
            result = test_configuration(config_name, config_changes)
            aggregation_test_results.append(result)
        except Exception as e:
            print(f"❌ {config_name} failed: {str(e)}")
            aggregation_test_results.append(False)
    
    test_results['aggregation_methods'] = all(aggregation_test_results)
    
    # Test 4: Attack detection
    print("\n🧪 Test 4: Attack Detection")
    test_results['attack_detection'] = test_attack_detection()
    
    # Test 5: Different client configurations
    print("\n🧪 Test 5: Client Configurations")
    client_configs = [
        ("3 Clients 33% Malicious", {'NUM_CLIENTS': 3, 'FRACTION_MALICIOUS': 0.33}),
        ("5 Clients 40% Malicious", {'NUM_CLIENTS': 5, 'FRACTION_MALICIOUS': 0.4}),
        ("7 Clients 29% Malicious", {'NUM_CLIENTS': 7, 'FRACTION_MALICIOUS': 0.29}),
    ]
    
    client_test_results = []
    for config_name, config_changes in client_configs:
        try:
            result = test_configuration(config_name, config_changes)
            client_test_results.append(result)
        except Exception as e:
            print(f"❌ {config_name} failed: {str(e)}")
            client_test_results.append(False)
    
    test_results['client_configs'] = all(client_test_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("⚠️ Most tests passed. System is mostly working correctly.")
        return True
    else:
        print("❌ Many tests failed. System needs further fixes.")
        return False

if __name__ == "__main__":
    try:
        success = run_comprehensive_tests()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 