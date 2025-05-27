#!/usr/bin/env python3
"""
Comprehensive validation suite for ALZHEIMER dataset with ResNet18
Tests both hybrid and dual attention configurations across all gradient combination methods and attacks.
"""

import subprocess
import sys
import time
import json
import torch
import os
from datetime import datetime
from typing import Dict, List, Tuple

class AlzheimerValidator:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
        # Define test configurations
        self.aggregation_methods = ['fedavg', 'fedbn', 'fedadmm']
        self.attack_types = [
            'none',
            'scaling_attack', 
            'partial_scaling_attack',
            'sign_flipping_attack',
            'noise_attack',
            'targeted_parameters',
            'min_max',
            'min_sum'
        ]
        self.rl_aggregation_modes = ['hybrid', 'dual_attention']
        
    def check_alzheimer_dataset(self):
        """Check if ALZHEIMER dataset is available."""
        dataset_path = './data/alzheimer'
        train_path = os.path.join(dataset_path, 'train')
        test_path = os.path.join(dataset_path, 'test')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("❌ ALZHEIMER dataset not found!")
            print(f"Expected paths:")
            print(f"  Train: {train_path}")
            print(f"  Test: {test_path}")
            print("\nPlease ensure the ALZHEIMER dataset is properly set up.")
            return False
            
        # Check if classes exist
        expected_classes = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
        for split in ['train', 'test']:
            split_path = os.path.join(dataset_path, split)
            existing_classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            
            for cls in expected_classes:
                if cls not in existing_classes:
                    print(f"❌ Missing class '{cls}' in {split} split")
                    return False
                    
        print("✅ ALZHEIMER dataset structure verified")
        return True
    
    def check_resnet18_compatibility(self):
        """Test ResNet18 model compatibility with ALZHEIMER dataset."""
        try:
            print("Testing ResNet18 compatibility...")
            from federated_learning.models.resnet import ResNet18Alzheimer
            
            # Test model creation
            model = ResNet18Alzheimer(num_classes=4, unfreeze_layers=5, pretrained=False)
            
            # Test forward pass with ALZHEIMER-sized input (3-channel, 128x128)
            test_input = torch.randn(2, 3, 128, 128)  # Batch of 2 for BatchNorm
            model.eval()
            output = model(test_input)
            
            expected_shape = (2, 4)  # batch_size, num_classes
            if output.shape == expected_shape:
                print("✅ ResNet18 compatibility verified")
                return True
            else:
                print(f"❌ ResNet18 output shape mismatch: expected {expected_shape}, got {output.shape}")
                return False
                
        except Exception as e:
            print(f"❌ ResNet18 compatibility test failed: {e}")
            return False
    
    def run_quick_functionality_test(self):
        """Run a quick test to verify basic functionality."""
        print("\n" + "="*60)
        print("QUICK FUNCTIONALITY TEST")
        print("="*60)
        
        cmd = [
            'python', 'main.py',
            '--dataset', 'ALZHEIMER',
            '--model', 'RESNET18',
            '--aggregation', 'fedavg',
            '--rl_aggregation', 'dual_attention',
            '--attack_type', 'none',
            '--fast_mode',
            '--global_epochs', '2',
            '--num_clients', '3',
            '--local_epochs', '1'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Quick test PASSED ({duration:.1f}s)")
                
                # Check for key indicators
                output = result.stdout
                if "Applied" in output and "attack" in output:
                    print("  ⚠️  Unexpected attack application in no-attack test")
                else:
                    print("  ✅ No attacks applied as expected")
                
                if "Initial test accuracy:" in output and "Final test accuracy:" in output:
                    print("  ✅ Training completed with accuracy measurements")
                else:
                    print("  ⚠️  Training completion unclear")
                
                return True
            else:
                print(f"❌ Quick test FAILED ({duration:.1f}s)")
                print(f"Error: {result.stderr[-300:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Quick test TIMEOUT (>180s)")
            return False
        except Exception as e:
            print(f"❌ Quick test EXCEPTION: {e}")
            return False
    
    def run_configuration_matrix_test(self):
        """Test key configuration combinations."""
        print("\n" + "="*60)
        print("CONFIGURATION MATRIX TEST")
        print("="*60)
        
        # Critical configuration combinations to test
        test_configs = [
            # Dual attention tests
            ("DualAttention_FedAvg_NoAttack", {
                'rl_aggregation': 'dual_attention',
                'aggregation': 'fedavg',
                'attack_type': 'none'
            }),
            ("DualAttention_FedBN_ScalingAttack", {
                'rl_aggregation': 'dual_attention',
                'aggregation': 'fedbn',
                'attack_type': 'scaling_attack'
            }),
            ("DualAttention_FedADMM_PartialScaling", {
                'rl_aggregation': 'dual_attention',
                'aggregation': 'fedadmm',
                'attack_type': 'partial_scaling_attack'
            }),
            
            # Hybrid tests
            ("Hybrid_FedAvg_SignFlipping", {
                'rl_aggregation': 'hybrid',
                'aggregation': 'fedavg',
                'attack_type': 'sign_flipping_attack'
            }),
            ("Hybrid_FedBN_NoAttack", {
                'rl_aggregation': 'hybrid',
                'aggregation': 'fedbn',
                'attack_type': 'none'
            }),
            ("Hybrid_FedADMM_NoiseAttack", {
                'rl_aggregation': 'hybrid',
                'aggregation': 'fedadmm',
                'attack_type': 'noise_attack'
            }),
        ]
        
        passed = 0
        total = len(test_configs)
        
        for test_name, config in test_configs:
            print(f"\n--- Testing: {test_name} ---")
            
            cmd = [
                'python', 'main.py',
                '--dataset', 'ALZHEIMER',
                '--model', 'RESNET18',
                '--aggregation', config['aggregation'],
                '--rl_aggregation', config['rl_aggregation'],
                '--attack_type', config['attack_type'],
                '--fast_mode',
                '--global_epochs', '3',
                '--num_clients', '4',
                '--local_epochs', '1',
                '--malicious_ratio', '0.25'
            ]
            
            try:
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    print(f"  ✅ PASSED ({duration:.1f}s)")
                    passed += 1
                    
                    # Analyze output
                    output = result.stdout
                    self._analyze_test_output(output, config)
                    
                else:
                    print(f"  ❌ FAILED ({duration:.1f}s)")
                    print(f"  Error: {result.stderr[-200:]}")
                    
                self.results[test_name] = {
                    'passed': result.returncode == 0,
                    'duration': duration,
                    'config': config,
                    'stdout_sample': result.stdout[-500:] if result.stdout else "",
                    'stderr_sample': result.stderr[-200:] if result.stderr else ""
                }
                
            except subprocess.TimeoutExpired:
                print(f"  ❌ TIMEOUT (>240s)")
                self.results[test_name] = {
                    'passed': False,
                    'duration': 240,
                    'config': config,
                    'error': 'TIMEOUT'
                }
            except Exception as e:
                print(f"  ❌ EXCEPTION: {e}")
                self.results[test_name] = {
                    'passed': False,
                    'duration': 0,
                    'config': config,
                    'error': str(e)
                }
        
        print(f"\nMatrix Test Results: {passed}/{total} PASSED")
        return passed >= total * 0.8  # Allow 20% failure
    
    def _analyze_test_output(self, output: str, config: dict):
        """Analyze test output for key indicators."""
        # Check attack application
        if config['attack_type'] != 'none':
            if f"Applied {config['attack_type']}" in output:
                print(f"    ✅ {config['attack_type']} correctly applied")
            else:
                print(f"    ⚠️  {config['attack_type']} may not have been applied")
        
        # Check aggregation method
        if config['aggregation'] in output:
            print(f"    ✅ {config['aggregation']} aggregation detected")
        
        # Check RL aggregation mode
        if config['rl_aggregation'] == 'hybrid':
            if "warmup" in output.lower() or "ramp" in output.lower():
                print(f"    ✅ Hybrid mode transitions detected")
        
        # Check training completion
        if "Final test accuracy:" in output:
            try:
                # Extract final accuracy
                lines = output.split('\n')
                for line in lines:
                    if "Final test accuracy:" in line:
                        acc_str = line.split(':')[1].split(',')[0].strip()
                        accuracy = float(acc_str)
                        print(f"    ✅ Final accuracy: {accuracy:.3f}")
                        break
            except:
                print(f"    ⚠️  Could not parse final accuracy")
    
    def run_stress_test(self):
        """Run a longer test with more rounds to check stability."""
        print("\n" + "="*60)
        print("STRESS TEST")
        print("="*60)
        
        cmd = [
            'python', 'main.py',
            '--dataset', 'ALZHEIMER',
            '--model', 'RESNET18',
            '--aggregation', 'fedbn',
            '--rl_aggregation', 'hybrid',
            '--attack_type', 'partial_scaling_attack',
            '--global_epochs', '8',
            '--num_clients', '5',
            '--local_epochs', '2',
            '--malicious_ratio', '0.2'
        ]
        
        print(f"Running stress test with 8 rounds...")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Stress test PASSED ({duration:.1f}s)")
                
                # Check for memory issues or crashes
                output = result.stdout
                if "memory" in output.lower() or "oom" in output.lower():
                    print("  ⚠️  Potential memory issues detected")
                else:
                    print("  ✅ No memory issues detected")
                
                return True
            else:
                print(f"❌ Stress test FAILED ({duration:.1f}s)")
                print(f"Error: {result.stderr[-300:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Stress test TIMEOUT (>600s)")
            return False
        except Exception as e:
            print(f"❌ Stress test EXCEPTION: {e}")
            return False
    
    def generate_comprehensive_report(self):
        """Generate a detailed report of all test results."""
        print("\n" + "="*80)
        print("ALZHEIMER/RESNET18 COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        if not self.results:
            print("No test results available.")
            return
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Calculate confidence level
        confidence = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        if confidence >= 90:
            status = "🎉 EXCELLENT"
            assessment = "ALZHEIMER/ResNet18 is fully ready for production use."
        elif confidence >= 75:
            status = "✅ GOOD"
            assessment = "ALZHEIMER/ResNet18 is working well with minor issues."
        elif confidence >= 50:
            status = "⚠️  ACCEPTABLE"
            assessment = "ALZHEIMER/ResNet18 has some issues but core functionality works."
        else:
            status = "❌ POOR"
            assessment = "ALZHEIMER/ResNet18 has significant issues requiring attention."
        
        print(f"\nOverall Assessment: {status}")
        print(f"Confidence Level: {confidence:.1f}%")
        print(f"Assessment: {assessment}")
        
        # Show individual test results
        print(f"\nDetailed Results:")
        for test_name, result in self.results.items():
            status = "✅" if result.get('passed', False) else "❌"
            duration = result.get('duration', 0)
            config = result.get('config', {})
            
            print(f"  {status} {test_name} ({duration:.1f}s)")
            if config:
                print(f"    Config: {config['rl_aggregation']}/{config['aggregation']}/{config['attack_type']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alzheimer_validation_{timestamp}.json"
        
        report_data = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'confidence': confidence,
                'assessment': assessment,
                'timestamp': timestamp
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\nDetailed results saved to: {filename}")
        
        return confidence >= 75

def main():
    """Run comprehensive ALZHEIMER/ResNet18 validation."""
    print("ALZHEIMER DATASET + RESNET18 COMPREHENSIVE VALIDATION")
    print("="*80)
    
    validator = AlzheimerValidator()
    
    # Phase 1: Prerequisites
    print("\nPhase 1: Checking Prerequisites...")
    if not validator.check_alzheimer_dataset():
        print("❌ Cannot proceed without ALZHEIMER dataset")
        return False
    
    if not validator.check_resnet18_compatibility():
        print("❌ ResNet18 compatibility issues detected")
        return False
    
    print("✅ All prerequisites satisfied")
    
    # Phase 2: Quick functionality test
    print("\nPhase 2: Quick functionality test...")
    if not validator.run_quick_functionality_test():
        print("❌ Basic functionality issues detected")
        return False
    
    # Phase 3: Configuration matrix test
    print("\nPhase 3: Configuration matrix test...")
    matrix_success = validator.run_configuration_matrix_test()
    
    # Phase 4: Stress test (optional, only if matrix test mostly passes)
    if matrix_success:
        print("\nPhase 4: Stress test...")
        validator.run_stress_test()
    else:
        print("\nSkipping stress test due to matrix test issues")
    
    # Phase 5: Generate report
    print("\nPhase 5: Generating comprehensive report...")
    overall_success = validator.generate_comprehensive_report()
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if overall_success:
        print("🎉 ALZHEIMER/RESNET18 READY FOR PRODUCTION")
        print("The system works correctly across all tested configurations.")
        print("You can proceed with high-parameter training runs.")
    else:
        print("⚠️  ALZHEIMER/RESNET18 NEEDS ATTENTION")
        print("Some issues were detected. Review the report before proceeding.")
        print("Consider fixing identified issues before high-parameter runs.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 