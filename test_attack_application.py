#!/usr/bin/env python3

import subprocess
import sys
import time

def test_attack_application():
    """Test that attacks are properly applied to malicious clients."""
    print("TESTING ATTACK APPLICATION")
    print("="*50)
    
    # Test with different attack types
    attack_types = ['scaling_attack', 'partial_scaling_attack', 'sign_flipping_attack']
    
    for attack in attack_types:
        print(f"\nTesting {attack}...")
        
        cmd = [
            'python', 'main.py',
            '--dataset', 'MNIST',
            '--model', 'CNN',
            '--aggregation', 'fedavg', 
            '--rl_aggregation', 'hybrid',
            '--attack_type', attack,
            '--fast_mode',
            '--global_epochs', '2',
            '--num_clients', '3',
            '--malicious_ratio', '0.33'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            
            if result.returncode == 0:
                output = result.stdout
                
                # Check for attack application
                if f"Applied {attack}" in output:
                    print(f"✅ {attack} successfully applied")
                else:
                    print(f"❌ {attack} NOT applied")
                    
                # Check for attack effects
                if "norm increased by:" in output or "Modified gradient norm:" in output:
                    print(f"✅ {attack} had measurable effect on gradients")
                else:
                    print(f"⚠️  {attack} effect not clearly visible")
                    
                # Check for any error messages about attacks
                if "Failed to apply gradient attack" in output:
                    print(f"❌ {attack} failed to apply with error")
                elif "Error" in output and attack in output:
                    print(f"⚠️  Possible error with {attack}")
                else:
                    print(f"✅ {attack} applied without errors")
                    
            else:
                print(f"❌ Test failed for {attack}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout for {attack}")
        except Exception as e:
            print(f"❌ Exception for {attack}: {e}")

def test_no_attack_baseline():
    """Test baseline with no attacks."""
    print(f"\nTesting baseline (no attacks)...")
    
    cmd = [
        'python', 'main.py',
        '--dataset', 'MNIST',
        '--model', 'CNN',
        '--aggregation', 'fedavg',
        '--rl_aggregation', 'hybrid', 
        '--attack_type', 'none',
        '--fast_mode',
        '--global_epochs', '2',
        '--num_clients', '3',
        '--malicious_ratio', '0.0'  # No malicious clients
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        
        if result.returncode == 0:
            print("✅ Baseline (no attacks) completed successfully")
            
            # Should not see any attack-related messages
            output = result.stdout
            if "Applied" in output and "attack" in output:
                print("⚠️  Unexpected attack application in no-attack test")
            else:
                print("✅ No attacks applied as expected")
                
        else:
            print("❌ Baseline test failed")
            
    except Exception as e:
        print(f"❌ Baseline test exception: {e}")

if __name__ == "__main__":
    test_attack_application()
    test_no_attack_baseline()
    print("\n" + "="*50)
    print("Attack application testing complete!") 