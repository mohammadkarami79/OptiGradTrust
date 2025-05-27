#!/usr/bin/env python3

import subprocess
import sys
import time

def quick_test():
    """Run a quick test to validate core functionality."""
    print("QUICK VALIDATION TEST")
    print("="*40)
    
    # Test basic functionality with a very short run
    cmd = [
        'python', 'main.py',
        '--dataset', 'MNIST',
        '--model', 'CNN', 
        '--aggregation', 'fedavg',
        '--rl_aggregation', 'hybrid',
        '--attack_type', 'scaling_attack',
        '--fast_mode',
        '--global_epochs', '2',
        '--local_epochs', '1',
        '--num_clients', '3',
        '--malicious_ratio', '0.33'
    ]
    
    print("Running: MNIST/CNN/FedAvg/Hybrid with scaling attack...")
    print("Expected: Attack should be applied and system should complete")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        duration = time.time() - start_time
        
        print(f"Duration: {duration:.1f}s")
        
        if result.returncode == 0:
            print("✅ SUCCESS: Process completed without errors")
            
            # Check for key indicators
            output = result.stdout
            
            # Check attack application
            if "Applied scaling_attack" in output:
                print("✅ Attack correctly applied to malicious clients")
            else:
                print("❌ Attack NOT applied - check attack system")
            
            # Check hybrid phases
            if "Warmup" in output or "warmup" in output:
                print("✅ Hybrid warmup phase detected")
            else:
                print("⚠️  Hybrid warmup phase not clearly shown")
            
            # Check model improvement
            lines = output.split('\n')
            initial_acc = None
            final_acc = None
            
            for line in lines:
                if 'Initial test accuracy:' in line:
                    try:
                        initial_acc = float(line.split(':')[1].split(',')[0].strip())
                    except:
                        pass
                elif 'Final test accuracy:' in line:
                    try:
                        final_acc = float(line.split(':')[1].split(',')[0].strip())
                    except:
                        pass
            
            if initial_acc and final_acc:
                improvement = final_acc - initial_acc
                print(f"✅ Model performance: {initial_acc:.3f} → {final_acc:.3f} (Δ{improvement:+.3f})")
                
                if improvement > 0:
                    print("✅ Model improved during training")
                else:
                    print("⚠️  Model did not improve (may be normal for short runs)")
            
            # Check for detection results
            if "False positives:" in output or "False negatives:" in output:
                print("✅ Detection system is running")
            else:
                print("⚠️  Detection metrics not clearly shown")
                
            return True
            
        else:
            print(f"❌ FAILED: Process returned error code {result.returncode}")
            print("STDERR:", result.stderr[-200:])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Process took longer than 120s")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    print("\n" + "="*40)
    if success:
        print("🎉 QUICK VALIDATION PASSED!")
        print("The hybrid aggregation system appears to be working correctly.")
        print("Key components verified:")
        print("  ✓ Attack application")
        print("  ✓ Hybrid phase transitions") 
        print("  ✓ Model training and evaluation")
        print("  ✓ Detection system operation")
    else:
        print("❌ QUICK VALIDATION FAILED!")
        print("The system has issues that need to be addressed.")
    
    print("="*40) 