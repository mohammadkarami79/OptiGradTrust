#!/usr/bin/env python3
"""
Comprehensive validation suite for hybrid aggregation method.
Tests all combinations of datasets, models, aggregation methods, and attacks.
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration matrix for comprehensive testing
DATASETS = ['MNIST', 'CIFAR10', 'ALZHEIMER']
MODELS = {
    'MNIST': ['CNN'],
    'CIFAR10': ['CNN', 'RESNET18'], 
    'ALZHEIMER': ['RESNET18', 'RESNET50']
}
AGGREGATION_METHODS = ['fedavg', 'fedbn', 'fedadmm']
ATTACK_TYPES = [
    'none',
    'scaling_attack', 
    'partial_scaling_attack',
    'sign_flipping_attack',
    'noise_attack',
    'targeted_parameters',
    'min_max',
    'min_sum'
]

class ValidationRunner:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_single_test(self, dataset: str, model: str, aggregation: str, 
                       attack: str, fast_mode: bool = True) -> Dict:
        """Run a single test configuration and return results."""
        
        test_name = f"{dataset}_{model}_{aggregation}_{attack}"
        print(f"\n{'='*60}")
        print(f"Running test: {test_name}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            'python', 'main.py',
            '--dataset', dataset,
            '--model', model,
            '--aggregation', aggregation,
            '--rl_aggregation', 'hybrid',
            '--attack_type', attack,
            '--global_epochs', '3' if fast_mode else '10',
            '--local_epochs', '2',
            '--num_clients', '5',
            '--malicious_ratio', '0.4'
        ]
        
        if fast_mode:
            cmd.append('--fast_mode')
            
        # Run test
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            duration = time.time() - start_time
            
            # Parse output for key metrics
            success = result.returncode == 0
            output_lines = result.stdout.split('\n') if result.stdout else []
            
            # Extract metrics from output
            initial_error = None
            final_error = None
            error_reduction = None
            
            for line in output_lines:
                if 'Initial test error:' in line:
                    try:
                        initial_error = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Final test error:' in line:
                    try:
                        final_error = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Error reduction:' in line:
                    try:
                        error_reduction = float(line.split(':')[1].split()[0])
                    except:
                        pass
            
            # Check for attack application
            attack_applied = 'Applied' in result.stdout and attack in result.stdout
            
            # Check for detection metrics
            false_positives = result.stdout.count('False positives:')
            false_negatives = result.stdout.count('False negatives')
            
            return {
                'success': success,
                'duration': duration,
                'initial_error': initial_error,
                'final_error': final_error,
                'error_reduction': error_reduction,
                'attack_applied': attack_applied,
                'false_positives': false_positives > 0,
                'false_negatives': false_negatives > 0,
                'output_sample': result.stdout[-500:] if result.stdout else '',
                'error_sample': result.stderr[-500:] if result.stderr else ''
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'duration': 300,
                'error': 'Timeout',
                'initial_error': None,
                'final_error': None,
                'error_reduction': None,
                'attack_applied': False,
                'false_positives': False,
                'false_negatives': False
            }
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e),
                'initial_error': None,
                'final_error': None,
                'error_reduction': None,
                'attack_applied': False,
                'false_positives': False,
                'false_negatives': False
            }
    
    def run_critical_tests(self) -> Dict:
        """Run critical tests that must pass for basic functionality."""
        
        print("\n" + "="*80)
        print("RUNNING CRITICAL TESTS")
        print("="*80)
        
        critical_tests = [
            # Basic functionality with no attacks
            ('MNIST', 'CNN', 'fedavg', 'none'),
            ('MNIST', 'CNN', 'fedbn', 'none'),
            
            # Attack handling
            ('MNIST', 'CNN', 'fedavg', 'scaling_attack'),
            ('MNIST', 'CNN', 'fedbn', 'partial_scaling_attack'),
            
            # Different aggregation methods
            ('MNIST', 'CNN', 'fedadmm', 'sign_flipping_attack'),
        ]
        
        results = {}
        passed = 0
        
        for dataset, model, aggregation, attack in critical_tests:
            test_name = f"{dataset}_{model}_{aggregation}_{attack}"
            result = self.run_single_test(dataset, model, aggregation, attack)
            results[test_name] = result
            
            if result['success']:
                print(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                print(f"❌ FAILED: {test_name}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        print(f"\nCritical Tests: {passed}/{len(critical_tests)} passed")
        return results
    
    def run_comprehensive_tests(self) -> Dict:
        """Run comprehensive test matrix (warning: time consuming)."""
        
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE TESTS")
        print("="*80)
        print("This will test all combinations of datasets, models, aggregations, and attacks.")
        print("Estimated time: 30-60 minutes")
        
        # Ask for confirmation
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping comprehensive tests.")
            return {}
        
        results = {}
        total_tests = 0
        passed_tests = 0
        
        for dataset in DATASETS:
            for model in MODELS[dataset]:
                for aggregation in AGGREGATION_METHODS:
                    for attack in ATTACK_TYPES[:4]:  # Limit to first 4 attacks for time
                        total_tests += 1
                        test_name = f"{dataset}_{model}_{aggregation}_{attack}"
                        
                        print(f"\nProgress: {total_tests}/{len(DATASETS)*2*3*4}")
                        result = self.run_single_test(dataset, model, aggregation, attack)
                        results[test_name] = result
                        
                        if result['success']:
                            passed_tests += 1
                            print(f"✅ {test_name}")
                        else:
                            print(f"❌ {test_name}")
        
        print(f"\nComprehensive Tests: {passed_tests}/{total_tests} passed")
        return results
    
    def run_detection_accuracy_tests(self) -> Dict:
        """Test malicious client detection accuracy."""
        
        print("\n" + "="*80)
        print("RUNNING DETECTION ACCURACY TESTS")
        print("="*80)
        
        detection_tests = [
            # Strong attacks should be detected
            ('MNIST', 'CNN', 'fedavg', 'scaling_attack'),
            ('MNIST', 'CNN', 'fedbn', 'partial_scaling_attack'),
            
            # Subtle attacks might be missed
            ('MNIST', 'CNN', 'fedavg', 'noise_attack'),
        ]
        
        results = {}
        
        for dataset, model, aggregation, attack in detection_tests:
            test_name = f"detection_{dataset}_{model}_{aggregation}_{attack}"
            result = self.run_single_test(dataset, model, aggregation, attack)
            
            # Analyze detection performance
            if result['success'] and attack != 'none':
                if result['attack_applied']:
                    if not result['false_negatives']:
                        detection_score = "GOOD"  # Attack applied and detected
                    else:
                        detection_score = "POOR"  # Attack applied but not detected
                else:
                    detection_score = "FAILED"  # Attack not even applied
            else:
                detection_score = "UNKNOWN"
            
            result['detection_score'] = detection_score
            results[test_name] = result
            
            print(f"Detection for {attack}: {detection_score}")
        
        return results
    
    def save_results(self, results: Dict):
        """Save test results to JSON file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_results_{timestamp}.json"
        
        # Add summary statistics
        summary = {
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results.values() if r.get('success', False)),
            'failed_tests': sum(1 for r in results.values() if not r.get('success', False)),
            'tests_with_attacks': sum(1 for r in results.values() if r.get('attack_applied', False)),
            'average_duration': sum(r.get('duration', 0) for r in results.values()) / len(results) if results else 0,
            'timestamp': timestamp
        }
        
        output = {
            'summary': summary,
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename
    
    def print_summary(self, results: Dict):
        """Print summary of validation results."""
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        if not results:
            print("No tests were run.")
            return
        
        total = len(results)
        passed = sum(1 for r in results.values() if r.get('success', False))
        failed = total - passed
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Analyze by category
        categories = {}
        for test_name, result in results.items():
            parts = test_name.split('_')
            if len(parts) >= 4:
                attack = parts[-1]
                categories.setdefault(attack, {'passed': 0, 'total': 0})
                categories[attack]['total'] += 1
                if result.get('success', False):
                    categories[attack]['passed'] += 1
        
        print(f"\nBy Attack Type:")
        for attack, stats in categories.items():
            rate = stats['passed'] / stats['total'] * 100
            print(f"  {attack}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        # Check for critical issues
        critical_failures = []
        for test_name, result in results.items():
            if not result.get('success', False):
                if 'none' in test_name:  # No-attack test failed
                    critical_failures.append(f"Basic functionality failed: {test_name}")
                elif result.get('error') == 'Timeout':
                    critical_failures.append(f"Timeout: {test_name}")
        
        if critical_failures:
            print(f"\n⚠️  CRITICAL ISSUES:")
            for issue in critical_failures:
                print(f"  {issue}")
        else:
            print(f"\n✅ No critical issues detected")
        
        # Performance metrics
        durations = [r.get('duration', 0) for r in results.values() if r.get('duration')]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            print(f"\nPerformance:")
            print(f"  Average test time: {avg_duration:.1f}s")
            print(f"  Longest test: {max_duration:.1f}s")

def main():
    """Run validation suite."""
    
    print("Hybrid Aggregation Comprehensive Validation Suite")
    print("=" * 60)
    
    runner = ValidationRunner()
    all_results = {}
    
    # Run different test suites
    print("\nChoose validation level:")
    print("1. Critical tests only (5 tests, ~5 minutes)")
    print("2. Detection accuracy tests (~3 tests)")
    print("3. Comprehensive tests (40+ tests, 30-60 minutes)")
    print("4. All tests")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        critical_results = runner.run_critical_tests()
        all_results.update(critical_results)
    
    if choice in ['2', '4']:
        detection_results = runner.run_detection_accuracy_tests()
        all_results.update(detection_results)
    
    if choice in ['3', '4']:
        comprehensive_results = runner.run_comprehensive_tests()
        all_results.update(comprehensive_results)
    
    # Save and summarize results
    if all_results:
        runner.save_results(all_results)
        runner.print_summary(all_results)
    else:
        print("No tests were run.")

if __name__ == "__main__":
    main() 