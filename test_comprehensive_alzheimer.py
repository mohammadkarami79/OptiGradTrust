#!/usr/bin/env python3
"""
Comprehensive test for ALZHEIMER + ResNet18 with all combinations
of hybrid/dual attention aggregation methods and attacks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure before imports
import federated_learning.config.config as config
import torch
import traceback
from datetime import datetime

def run_comprehensive_test():
    """Run comprehensive tests for all combinations."""
    print("COMPREHENSIVE ALZHEIMER + RESNET18 TEST")
    print("=" * 80)
    
    # Base configuration
    base_config = {
        'DATASET': 'ALZHEIMER',
        'MODEL': 'RESNET18',
        'NUM_CLIENTS': 4,
        'FRACTION_MALICIOUS': 0.25,  # 1 malicious client
        'GLOBAL_EPOCHS': 2,
        'LOCAL_EPOCHS_CLIENT': 1,
        'BATCH_SIZE': 8,
        'FAST_MODE': True,
        'AGGREGATION_METHOD': 'fedavg',
        'ENABLE_DUAL_ATTENTION': True,
        'ENABLE_VAE': True,
        'ENABLE_SHAPLEY': True,
    }
    
    # Test combinations
    test_combinations = [
        # Dual attention mode with different aggregation methods
        {
            'name': 'Dual Attention + FedAvg + No Attack',
            'rl_aggregation': 'dual_attention',
            'aggregation': 'fedavg',
            'attack_type': 'none'
        },
        {
            'name': 'Dual Attention + FedBN + Scaling Attack',
            'rl_aggregation': 'dual_attention',
            'aggregation': 'fedbn',
            'attack_type': 'scaling_attack'
        },
        {
            'name': 'Dual Attention + FedProx + Partial Scaling Attack',
            'rl_aggregation': 'dual_attention',
            'aggregation': 'fedprox',
            'attack_type': 'partial_scaling_attack'
        },
        {
            'name': 'Dual Attention + FedADMM + Sign Flipping Attack',
            'rl_aggregation': 'dual_attention',
            'aggregation': 'fedadmm',
            'attack_type': 'sign_flipping_attack'
        },
        
        # Hybrid mode with different aggregation methods
        {
            'name': 'Hybrid + FedAvg + Noise Attack',
            'rl_aggregation': 'hybrid',
            'aggregation': 'fedavg',
            'attack_type': 'noise_attack'
        },
        {
            'name': 'Hybrid + FedBN + Min Max Attack',
            'rl_aggregation': 'hybrid',
            'aggregation': 'fedbn',
            'attack_type': 'min_max_attack'
        },
        {
            'name': 'Hybrid + FedProx + Targeted Attack',
            'rl_aggregation': 'hybrid',
            'aggregation': 'fedprox',
            'attack_type': 'targeted_attack'
        },
        {
            'name': 'Hybrid + FedADMM + Backdoor Attack',
            'rl_aggregation': 'hybrid',
            'aggregation': 'fedadmm',
            'attack_type': 'backdoor_attack'
        },
        
        # RL Actor-Critic mode with different attacks
        {
            'name': 'RL Actor-Critic + FedAvg + Scaling Attack',
            'rl_aggregation': 'rl_actor_critic',
            'aggregation': 'fedavg',
            'attack_type': 'scaling_attack'
        },
        {
            'name': 'RL Actor-Critic + FedBN + Sign Flipping Attack',
            'rl_aggregation': 'rl_actor_critic',
            'aggregation': 'fedbn',
            'attack_type': 'sign_flipping_attack'
        }
    ]
    
    results = []
    successful_tests = 0
    total_tests = len(test_combinations)
    
    for i, test_config in enumerate(test_combinations, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{total_tests}: {test_config['name']}")
        print(f"{'='*80}")
        
        # Apply base configuration
        for key, value in base_config.items():
            setattr(config, key, value)
        
        # Apply test-specific configuration
        config.RL_AGGREGATION_METHOD = test_config['rl_aggregation']
        config.AGGREGATION_METHOD = test_config['aggregation']
        config.ATTACK_TYPE = test_config['attack_type']
        
        # Set hybrid-specific parameters
        if test_config['rl_aggregation'] == 'hybrid':
            config.RL_WARMUP_ROUNDS = 1
            config.RL_RAMP_UP_ROUNDS = 1
        
        print(f"Configuration:")
        print(f"  RL Aggregation: {config.RL_AGGREGATION_METHOD}")
        print(f"  Base Aggregation: {config.AGGREGATION_METHOD}")
        print(f"  Attack Type: {config.ATTACK_TYPE}")
        
        # Run the test
        test_start_time = datetime.now()
        try:
            # Import main after setting config
            import importlib
            if 'main' in sys.modules:
                importlib.reload(sys.modules['main'])
            else:
                import main
            
            # Clear old model files to ensure fresh start
            import shutil
            model_weights_dir = 'model_weights'
            if os.path.exists(model_weights_dir):
                try:
                    # Only delete files specific to ALZHEIMER/RESNET18
                    for filename in os.listdir(model_weights_dir):
                        if 'ALZHEIMER_RESNET18' in filename:
                            os.remove(os.path.join(model_weights_dir, filename))
                except:
                    pass  # Ignore deletion errors
            
            # Run the main function with current config
            test_errors, round_metrics = main.main()
            
            test_duration = (datetime.now() - test_start_time).total_seconds()
            
            # Check if test was successful
            if test_errors and len(test_errors) > 0:
                final_error = test_errors[-1]
                initial_error = test_errors[0]
                improvement = initial_error - final_error
                
                result = {
                    'name': test_config['name'],
                    'status': 'SUCCESS',
                    'duration': test_duration,
                    'initial_error': initial_error,
                    'final_error': final_error,
                    'improvement': improvement,
                    'rounds_completed': len(test_errors) - 1,
                    'details': f"Error reduced from {initial_error:.4f} to {final_error:.4f}"
                }
                
                successful_tests += 1
                print(f"✅ TEST PASSED in {test_duration:.1f}s")
                print(f"   Initial error: {initial_error:.4f}")
                print(f"   Final error: {final_error:.4f}")
                print(f"   Improvement: {improvement:.4f}")
                
            else:
                result = {
                    'name': test_config['name'],
                    'status': 'FAILED',
                    'duration': test_duration,
                    'error': 'No test errors returned',
                    'details': 'Training completed but no error metrics available'
                }
                print(f"❌ TEST FAILED: No error metrics returned")
            
        except Exception as e:
            test_duration = (datetime.now() - test_start_time).total_seconds()
            error_msg = str(e)
            
            result = {
                'name': test_config['name'],
                'status': 'ERROR',
                'duration': test_duration,
                'error': error_msg,
                'details': f"Exception during execution: {error_msg}"
            }
            
            print(f"❌ TEST ERROR in {test_duration:.1f}s: {error_msg}")
            # Print traceback for debugging
            traceback.print_exc()
        
        results.append(result)
        
        # Clean up GPU memory between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\nDetailed Results:")
    print("-" * 80)
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {result['name']}")
        print(f"   Status: {result['status']} ({result['duration']:.1f}s)")
        if result['status'] == 'SUCCESS':
            print(f"   {result['details']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        print()
    
    # Summary by category
    print("Summary by RL Aggregation Method:")
    print("-" * 40)
    
    rl_methods = {}
    for result in results:
        # Extract RL method from test name
        if 'Dual Attention' in result['name']:
            method = 'Dual Attention'
        elif 'Hybrid' in result['name']:
            method = 'Hybrid'
        elif 'RL Actor-Critic' in result['name']:
            method = 'RL Actor-Critic'
        else:
            method = 'Unknown'
        
        if method not in rl_methods:
            rl_methods[method] = {'total': 0, 'success': 0}
        
        rl_methods[method]['total'] += 1
        if result['status'] == 'SUCCESS':
            rl_methods[method]['success'] += 1
    
    for method, stats in rl_methods.items():
        success_rate = stats['success'] / stats['total'] * 100
        print(f"{method}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    print(f"\n{'='*80}")
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED! ALZHEIMER + ResNet18 system is fully working!")
        print("✅ Ready for high-parameter training and paper writing.")
    elif successful_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. System is mostly working with minor issues.")
        print("🔧 Consider investigating failed tests before high-parameter training.")
    else:
        print("❌ Multiple tests failed. System needs debugging before proceeding.")
        print("🛠️  Please fix the issues before running high-parameter experiments.")
    print(f"{'='*80}")
    
    return results, successful_tests == total_tests

if __name__ == "__main__":
    results, all_passed = run_comprehensive_test()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_test_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("COMPREHENSIVE ALZHEIMER + RESNET18 TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Test: {result['name']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            if result['status'] == 'SUCCESS':
                f.write(f"Details: {result['details']}\n")
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    print(f"\nTest results saved to: {results_file}")
    
    if all_passed:
        print("\n🚀 System ready for production training!")
        sys.exit(0)
    else:
        print("\n🔧 System needs fixes before production.")
        sys.exit(1) 