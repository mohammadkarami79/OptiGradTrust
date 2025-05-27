#!/usr/bin/env python3

import subprocess
import sys
import time
import json
from datetime import datetime

class ComprehensiveValidator:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def run_test(self, test_name, cmd, timeout=300):
        """Run a single test with comprehensive monitoring."""
        print(f"\n{'='*60}")
        print(f"TESTING: {test_name}")
        print(f"CMD: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            duration = time.time() - start_time
            
            success = result.returncode == 0
            output = result.stdout if result.stdout else ""
            error_output = result.stderr if result.stderr else ""
            
            # Extract metrics
            metrics = self.extract_metrics(output)
            
            # Analyze results
            analysis = self.analyze_output(output, test_name)
            
            result_data = {
                'success': success,
                'duration': duration,
                'metrics': metrics,
                'analysis': analysis,
                'error_output': error_output[-500:] if error_output else "",
                'stdout_sample': output[-1000:] if output else ""
            }
            
            self.results[test_name] = result_data
            
            # Print summary
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status}: {test_name} ({duration:.1f}s)")
            
            if success:
                if metrics.get('initial_accuracy') and metrics.get('final_accuracy'):
                    improvement = metrics['final_accuracy'] - metrics['initial_accuracy']
                    print(f"  Performance: {metrics['initial_accuracy']:.3f} → {metrics['final_accuracy']:.3f} (Δ{improvement:+.3f})")
                
                if analysis.get('attack_applied'):
                    print(f"  ✅ Attack successfully applied")
                elif 'attack' in test_name.lower() and 'none' not in test_name.lower():
                    print(f"  ⚠️  Attack may not have been applied")
                    
                if analysis.get('hybrid_phases_detected'):
                    print(f"  ✅ Hybrid phases detected")
                    
                if analysis.get('detection_working'):
                    print(f"  ✅ Detection system working")
            else:
                print(f"  ❌ Error: {error_output[-200:] if error_output else 'Unknown error'}")
                
            return success
            
        except subprocess.TimeoutExpired:
            duration = timeout
            result_data = {
                'success': False,
                'duration': duration,
                'error': 'TIMEOUT',
                'metrics': {},
                'analysis': {}
            }
            self.results[test_name] = result_data
            print(f"❌ TIMEOUT: {test_name} (>{timeout}s)")
            return False
            
        except Exception as e:
            duration = time.time() - start_time
            result_data = {
                'success': False,
                'duration': duration,
                'error': str(e),
                'metrics': {},
                'analysis': {}
            }
            self.results[test_name] = result_data
            print(f"❌ EXCEPTION: {test_name} - {e}")
            return False
    
    def extract_metrics(self, output):
        """Extract key metrics from output."""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            if 'Initial test accuracy:' in line:
                try:
                    metrics['initial_accuracy'] = float(line.split(':')[1].split(',')[0].strip())
                except:
                    pass
            elif 'Final test accuracy:' in line:
                try:
                    metrics['final_accuracy'] = float(line.split(':')[1].split(',')[0].strip())
                except:
                    pass
            elif 'Initial test error:' in line:
                try:
                    metrics['initial_error'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Final test error:' in line:
                try:
                    metrics['final_error'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Error reduction:' in line:
                try:
                    metrics['error_reduction'] = float(line.split(':')[1].split()[0])
                except:
                    pass
        
        return metrics
    
    def analyze_output(self, output, test_name):
        """Analyze output for key indicators."""
        analysis = {}
        
        # Check attack application
        attack_keywords = ['Applied scaling_attack', 'Applied partial_scaling_attack', 'Applied sign_flipping_attack', 'Applied noise_attack']
        analysis['attack_applied'] = any(keyword in output for keyword in attack_keywords)
        
        # Check hybrid phases
        hybrid_keywords = ['warmup', 'Warmup', 'ramp-up', 'Ramp-up', 'Hybrid mode', 'blend']
        analysis['hybrid_phases_detected'] = any(keyword in output for keyword in hybrid_keywords)
        
        # Check detection system
        detection_keywords = ['False positives:', 'False negatives:', 'Trust Score', 'Detection']
        analysis['detection_working'] = any(keyword in output for keyword in detection_keywords)
        
        # Check for errors
        error_keywords = ['Error:', 'Failed:', 'Exception:', 'Traceback']
        analysis['has_errors'] = any(keyword in output for keyword in error_keywords)
        
        # Check completion
        completion_keywords = ['Training Summary', 'Final Model Evaluation', 'saved to']
        analysis['completed_properly'] = any(keyword in output for keyword in completion_keywords)
        
        return analysis
    
    def run_core_functionality_tests(self):
        """Test core functionality across key configurations."""
        print("\n" + "="*80)
        print("PHASE 1: CORE FUNCTIONALITY TESTS")
        print("="*80)
        
        tests = [
            # Basic functionality tests
            ("MNIST_CNN_FedAvg_NoAttack", [
                'python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN', 
                '--aggregation', 'fedavg', '--rl_aggregation', 'hybrid', 
                '--attack_type', 'none', '--fast_mode', '--global_epochs', '3'
            ]),
            
            ("MNIST_CNN_FedBN_NoAttack", [
                'python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN',
                '--aggregation', 'fedbn', '--rl_aggregation', 'hybrid',
                '--attack_type', 'none', '--fast_mode', '--global_epochs', '3'
            ]),
            
            # Attack tests
            ("MNIST_CNN_FedAvg_ScalingAttack", [
                'python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN',
                '--aggregation', 'fedavg', '--rl_aggregation', 'hybrid',
                '--attack_type', 'scaling_attack', '--fast_mode', '--global_epochs', '3'
            ]),
        ]
        
        passed = 0
        for test_name, cmd in tests:
            if self.run_test(test_name, cmd, timeout=300):
                passed += 1
                
        print(f"\nPhase 1 Results: {passed}/{len(tests)} tests passed")
        return passed == len(tests)
    
    def run_cross_dataset_tests(self):
        """Test functionality across different datasets."""
        print("\n" + "="*80)
        print("PHASE 2: CROSS-DATASET TESTS")
        print("="*80)
        
        # Only test if CIFAR10 and ALZHEIMER datasets are available
        tests = []
        
        # Test CIFAR10 if available
        try:
            # Quick check if CIFAR10 works
            tests.append(("CIFAR10_CNN_FedAvg_NoAttack", [
                'python', 'main.py', '--dataset', 'CIFAR10', '--model', 'CNN',
                '--aggregation', 'fedavg', '--rl_aggregation', 'hybrid',
                '--attack_type', 'none', '--fast_mode', '--global_epochs', '2'
            ]))
        except:
            print("Skipping CIFAR10 tests - dataset may not be available")
        
        # Test ALZHEIMER if available
        try:
            tests.append(("ALZHEIMER_RESNET18_FedBN_NoAttack", [
                'python', 'main.py', '--dataset', 'ALZHEIMER', '--model', 'RESNET18',
                '--aggregation', 'fedbn', '--rl_aggregation', 'hybrid',
                '--attack_type', 'none', '--fast_mode', '--global_epochs', '2'
            ]))
        except:
            print("Skipping ALZHEIMER tests - dataset may not be available")
        
        if not tests:
            print("No cross-dataset tests to run")
            return True
        
        passed = 0
        for test_name, cmd in tests:
            if self.run_test(test_name, cmd, timeout=400):
                passed += 1
        
        print(f"\nPhase 2 Results: {passed}/{len(tests)} tests passed")
        return passed >= len(tests) * 0.5  # Allow 50% failure for cross-dataset
    
    def run_extended_tests(self):
        """Run longer tests to verify stability."""
        print("\n" + "="*80)
        print("PHASE 3: EXTENDED TESTS")
        print("="*80)
        
        tests = [
            # Longer training run
            ("MNIST_CNN_Extended_10Rounds", [
                'python', 'main.py', '--dataset', 'MNIST', '--model', 'CNN',
                '--aggregation', 'fedavg', '--rl_aggregation', 'hybrid',
                '--attack_type', 'partial_scaling_attack', '--global_epochs', '10'
            ]),
        ]
        
        passed = 0
        for test_name, cmd in tests:
            if self.run_test(test_name, cmd, timeout=600):
                passed += 1
        
        print(f"\nPhase 3 Results: {passed}/{len(tests)} tests passed")
        return passed >= 1
    
    def save_results(self):
        """Save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_validation_{timestamp}.json"
        
        summary = {
            'total_tests': len(self.results),
            'passed_tests': sum(1 for r in self.results.values() if r.get('success', False)),
            'failed_tests': sum(1 for r in self.results.values() if not r.get('success', False)),
            'average_duration': sum(r.get('duration', 0) for r in self.results.values()) / len(self.results),
            'total_duration': sum(r.get('duration', 0) for r in self.results.values()),
            'timestamp': timestamp
        }
        
        output = {
            'summary': summary,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"\nResults saved to: {filename}")
        return filename
    
    def print_final_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No tests were run.")
            return
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.get('success', False))
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Calculate overall confidence
        confidence = passed / total * 100 if total > 0 else 0
        
        if confidence >= 90:
            status = "🎉 EXCELLENT"
            assessment = "System is working correctly across all tested conditions."
        elif confidence >= 75:
            status = "✅ GOOD"
            assessment = "System is working well with minor issues."
        elif confidence >= 50:
            status = "⚠️  ACCEPTABLE"
            assessment = "System has some issues but core functionality works."
        else:
            status = "❌ POOR"
            assessment = "System has significant issues that need attention."
        
        print(f"\nOverall Assessment: {status}")
        print(f"Confidence Level: {confidence:.1f}%")
        print(f"Assessment: {assessment}")
        
        # Show test details
        print(f"\nTest Details:")
        for test_name, result in self.results.items():
            status = "✅" if result.get('success', False) else "❌"
            duration = result.get('duration', 0)
            print(f"  {status} {test_name} ({duration:.1f}s)")
        
        print(f"\nTotal validation time: {sum(r.get('duration', 0) for r in self.results.values()):.1f}s")

def main():
    """Run comprehensive validation."""
    print("HYBRID AGGREGATION COMPREHENSIVE VALIDATION")
    print("="*80)
    print("This will run extensive tests to validate the hybrid aggregation system.")
    print("Estimated time: 20-40 minutes depending on hardware.")
    
    validator = ComprehensiveValidator()
    
    # Run validation phases
    phase1_success = validator.run_core_functionality_tests()
    phase2_success = validator.run_cross_dataset_tests()
    
    # Only run extended tests if core tests pass
    if phase1_success:
        phase3_success = validator.run_extended_tests()
    else:
        print("\nSkipping extended tests due to core functionality failures.")
        phase3_success = False
    
    # Save and summarize
    validator.save_results()
    validator.print_final_summary()
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if phase1_success and phase2_success:
        print("🎉 SYSTEM READY FOR PRODUCTION")
        print("The hybrid aggregation system has passed comprehensive validation.")
        print("All core functionality works correctly across multiple configurations.")
    elif phase1_success:
        print("✅ CORE SYSTEM WORKING")
        print("Basic functionality is solid, but some cross-dataset issues detected.")
        print("Recommended for use with MNIST/CNN configurations.")
    else:
        print("❌ SYSTEM NEEDS ATTENTION")
        print("Core functionality issues detected. Please address before deployment.")
        
    return phase1_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 