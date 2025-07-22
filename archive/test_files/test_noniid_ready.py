#!/usr/bin/env python3
"""
🧪 NON-IID READINESS TEST
========================

Quick validation test to confirm we're ready for Phase 2: Non-IID experiments.

Author: Research Team  
Date: 2025-01-27
Purpose: Validate all systems before starting Non-IID phase
"""

import os
import sys
import torch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_phase1_completion():
    """Test that Phase 1 (IID) is completed successfully"""
    print("🔍 Checking Phase 1 (IID) Completion...")
    
    # Check if corrected results exist
    corrected_results_path = 'results/final_paper_submission_ready/FINAL_CORRECTED_RESULTS.md'
    if os.path.exists(corrected_results_path):
        print("✅ Phase 1 corrected results found")
        
        # Read and validate key metrics
        with open(corrected_results_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for fixed issues
        accuracy_fixed = '85.20' in content  # CIFAR-10 accuracy fixed
        sign_flipping_fixed = '45.00' in content  # Sign flipping detection fixed
        label_flipping_fixed = '40.00' in content  # Label flipping detection fixed
        
        if accuracy_fixed and sign_flipping_fixed and label_flipping_fixed:
            print("✅ All Phase 1 issues confirmed fixed")
            return True
        else:
            print("❌ Some Phase 1 issues still remain")
            return False
    else:
        print("❌ Phase 1 corrected results not found")
        return False

def test_noniid_config():
    """Test Non-IID configuration loading"""
    print("\n🔍 Testing Non-IID Configuration...")
    
    try:
        # Import Non-IID config
        from federated_learning.config import config_noniid
        
        # Check key Non-IID parameters
        checks = {
            'Data Distribution': config_noniid.DATA_DISTRIBUTION == 'non_iid',
            'Heterogeneity Level': hasattr(config_noniid, 'HETEROGENEITY_LEVEL'),
            'Classes Per Client': hasattr(config_noniid, 'NON_IID_CLASSES_PER_CLIENT'),
            'Expected Results': hasattr(config_noniid, 'IID_BASELINE'),
            'Timeline': hasattr(config_noniid, 'PHASE_2_TIMELINE'),
        }
        
        all_passed = all(checks.values())
        
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}")
        
        if all_passed:
            print("✅ Non-IID configuration validated")
            
            # Print expected timeline
            timeline = config_noniid.get_non_iid_timeline()
            print(f"⏱️ Expected timeline: {timeline['TOTAL_ESTIMATED']}")
            
            return True
        else:
            print("❌ Non-IID configuration validation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import Non-IID config: {e}")
        return False

def test_system_resources():
    """Test system resources for Non-IID experiments"""
    print("\n🔍 Checking System Resources...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   {'✅' if cuda_available else '⚠️'} CUDA Available: {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   📱 GPU: {gpu_name}")
        print(f"   💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Memory check for Non-IID (needs more than IID)
        memory_sufficient = gpu_memory >= 5.0  # Need at least 5GB for Non-IID
        print(f"   {'✅' if memory_sufficient else '⚠️'} Memory Sufficient: {memory_sufficient}")
        
        return memory_sufficient
    else:
        print("   ⚠️ Will use CPU (slower but functional)")
        return True

def test_expected_performance():
    """Show expected Non-IID performance vs IID"""
    print("\n📊 Expected Non-IID Performance:")
    
    try:
        from federated_learning.config import config_noniid
        
        print("   🔄 IID vs Non-IID Comparison:")
        
        for dataset, iid_results in config_noniid.IID_BASELINE.items():
            expected_drops = config_noniid.EXPECTED_DROPS[dataset]
            
            iid_acc = iid_results['accuracy']
            iid_det = iid_results['detection']
            
            noniid_acc = iid_acc + expected_drops['accuracy']
            noniid_det = iid_det + expected_drops['detection']
            
            print(f"\n   📋 {dataset}:")
            print(f"      Accuracy: {iid_acc:.1f}% → {noniid_acc:.1f}% ({expected_drops['accuracy']:+.1f}%)")
            print(f"      Detection: {iid_det:.1f}% → {noniid_det:.1f}% ({expected_drops['detection']:+.1f}%)")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading expected performance: {e}")
        return False

def main():
    """Main readiness test"""
    print("="*60)
    print("🚀 NON-IID READINESS TEST")
    print("="*60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        ("Phase 1 Completion", test_phase1_completion),
        ("Non-IID Configuration", test_noniid_config), 
        ("System Resources", test_system_resources),
        ("Expected Performance", test_expected_performance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Final assessment
    print("\n" + "="*60)
    print("🏆 READINESS ASSESSMENT")
    print("="*60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready to proceed with Phase 2: Non-IID Experiments")
        print("\n📋 Next Steps:")
        print("   1. Run Non-IID experiments on MNIST (30 min)")
        print("   2. Run Non-IID experiments on Alzheimer (45 min)")
        print("   3. Run Non-IID experiments on CIFAR-10 (60 min)")
        print("   4. Generate comparative analysis (15 min)")
        print("   ⏱️ Total estimated time: 2.5 hours")
        return True
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("❌ Not ready for Phase 2. Please address issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 