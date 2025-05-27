#!/usr/bin/env python3

import subprocess
import sys
import os
import time

def run_test(test_name, cmd, timeout=240):
    """Run a single test and return results."""
    print('='*60)
    print(f'RUNNING: {test_name}')
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f'✅ PASSED: {test_name} ({duration:.1f}s)')
            
            # Extract key metrics
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['Initial test error:', 'Final test error:', 'Error reduction:', 'Applied', 'False positives:', 'False negatives:']):
                    print(f'  {line.strip()}')
            return True
        else:
            print(f'❌ FAILED: {test_name}')
            print(f'Error output: {result.stderr[-300:]}')
            return False
            
    except subprocess.TimeoutExpired:
        print(f'❌ TIMEOUT: {test_name} (>{timeout}s)')
        return False
    except Exception as e:
        print(f'❌ EXCEPTION: {test_name} - {e}')
        return False

def main():
    """Run critical validation tests."""
    print("HYBRID AGGREGATION CRITICAL VALIDATION")
    print("="*60)
    
    tests = [
        # Test 1: Basic functionality (no attacks)
        ("Basic MNIST/CNN/FedAvg (no attacks)", 
         ['python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN', '--aggregation', 'fedavg', 
          '--rl_aggregation', 'hybrid', '--attack_type', 'none', '--fast_mode', '--global_epochs', '3']),
        
        # Test 2: Basic functionality with FedBN
        ("Basic MNIST/CNN/FedBN (no attacks)",
         ['python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN', '--aggregation', 'fedbn', 
          '--rl_aggregation', 'hybrid', '--attack_type', 'none', '--fast_mode', '--global_epochs', '3']),
        
        # Test 3: Scaling attack detection
        ("Scaling attack detection",
         ['python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN', '--aggregation', 'fedavg', 
          '--rl_aggregation', 'hybrid', '--attack_type', 'scaling_attack', '--fast_mode', '--global_epochs', '3']),
        
        # Test 4: Partial scaling attack
        ("Partial scaling attack",
         ['python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN', '--aggregation', 'fedbn', 
          '--rl_aggregation', 'hybrid', '--attack_type', 'partial_scaling_attack', '--fast_mode', '--global_epochs', '3']),
        
        # Test 5: FedADMM with sign flipping
        ("FedADMM with sign flipping",
         ['python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN', '--aggregation', 'fedadmm', 
          '--rl_aggregation', 'hybrid', '--attack_type', 'sign_flipping_attack', '--fast_mode', '--global_epochs', '3']),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, cmd in tests:
        if run_test(test_name, cmd):
            passed += 1
        print()  # Empty line for readability
    
    print("="*60)
    print(f"CRITICAL TESTS SUMMARY: {passed}/{total} PASSED")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL CRITICAL TESTS PASSED! System is working correctly.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed, but some issues detected.")
    else:
        print("❌ CRITICAL FAILURES detected. System needs attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 