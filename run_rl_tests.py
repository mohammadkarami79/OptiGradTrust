import os
import sys
import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Ensure the federated_learning module is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import key config values at module level
from federated_learning.config.config import (
    RL_AGGREGATION_METHOD, RL_ACTOR_HIDDEN_DIMS, RL_CRITIC_HIDDEN_DIMS,
    RL_LEARNING_RATE, RL_GAMMA, RL_ENTROPY_COEF, RL_WARMUP_ROUNDS,
    RL_RAMP_UP_ROUNDS, RL_INITIAL_TEMP, RL_MIN_TEMP, AGGREGATION_METHOD,
    device as config_device
)

def run_all_tests():
    """Run all RL and hybrid mode tests and report results"""
    print("\n===================================================")
    print("        RUNNING RL AND HYBRID AGGREGATION TESTS     ")
    print("===================================================\n")
    
    # Store start time
    start_time = time.time()
    
    # Initialize results dictionary
    results = {}
    errors = {}
    
    # Create test results directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    os.makedirs('test_results/plots', exist_ok=True)
    
    # Import test modules - do this here to avoid early import errors
    try:
        from test_rl_aggregation import test_rl_aggregation, test_hybrid_mode
        from test_hybrid_transition import test_hybrid_weight_blending
    except ImportError as e:
        print(f"Critical error importing test modules: {e}")
        return False
    
    # Test 1: Basic RL Aggregation
    print("\n----- Test 1: Basic RL Aggregation -----")
    try:
        passed = test_rl_aggregation()
        results['RL Aggregation'] = passed
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
    except Exception as e:
        print(f"Error in RL Aggregation test: {str(e)}")
        traceback.print_exc()
        results['RL Aggregation'] = False
        errors['RL Aggregation'] = str(e)
    
    # Test 2: Hybrid Mode Phases
    print("\n----- Test 2: Hybrid Mode Phases -----")
    try:
        passed = test_hybrid_mode()
        results['Hybrid Mode Phases'] = passed
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
    except Exception as e:
        print(f"Error in Hybrid Mode test: {str(e)}")
        traceback.print_exc()
        results['Hybrid Mode Phases'] = False
        errors['Hybrid Mode Phases'] = str(e)
    
    # Test 3: Hybrid Weight Blending
    print("\n----- Test 3: Hybrid Weight Blending -----")
    try:
        passed = test_hybrid_weight_blending()
        results['Hybrid Weight Blending'] = passed
        print(f"Result: {'PASSED' if passed else 'FAILED'}")
    except Exception as e:
        print(f"Error in Hybrid Weight Blending test: {str(e)}")
        traceback.print_exc()
        results['Hybrid Weight Blending'] = False
        errors['Hybrid Weight Blending'] = str(e)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n===================================================")
    print("                TEST RESULTS SUMMARY                ")
    print("===================================================")
    for test_name, passed in results.items():
        status = 'PASSED' if passed else 'FAILED'
        error_info = f" - Error: {errors.get(test_name, 'Unknown error')}" if not passed else ""
        print(f"{test_name}: {status}{error_info}")
    
    total_passed = sum(1 for passed in results.values() if passed)
    print(f"\nPassed {total_passed} out of {len(results)} tests")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    test_names = list(results.keys())
    test_results = [1 if results[name] else 0 for name in test_names]
    
    plt.bar(test_names, test_results, color=['green' if r else 'red' for r in test_results])
    plt.title('RL and Hybrid Aggregation Test Results')
    plt.ylim(0, 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, passed in enumerate(test_results):
        color = 'green' if passed else 'red'
        plt.text(i, passed + 0.05, 'PASSED' if passed else 'FAILED', 
                ha='center', va='bottom', color=color, fontweight='bold')
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f'test_results/rl_test_summary_{timestamp}.png')
    print(f"Test summary saved to test_results/rl_test_summary_{timestamp}.png")
    
    # Save detailed error information
    if errors:
        error_file = f'test_results/error_details_{timestamp}.txt'
        with open(error_file, 'w') as f:
            f.write("RL and Hybrid Aggregation Test Error Details\n")
            f.write("===========================================\n\n")
            for test_name, error in errors.items():
                f.write(f"{test_name}:\n{error}\n\n")
        print(f"Error details saved to {error_file}")
    
    # Return overall success/failure
    return total_passed == len(results)

def check_rl_config():
    """Check if the RL configuration is properly set up"""
    print("\n===================================================")
    print("            CHECKING RL CONFIGURATION               ")
    print("===================================================\n")
    
    try:
        # Check against required configurations
        required_configs = [
            'RL_AGGREGATION_METHOD',
            'RL_ACTOR_HIDDEN_DIMS',
            'RL_CRITIC_HIDDEN_DIMS',
            'RL_LEARNING_RATE',
            'RL_GAMMA',
            'RL_ENTROPY_COEF',
            'RL_WARMUP_ROUNDS',
            'RL_RAMP_UP_ROUNDS',
            'RL_INITIAL_TEMP',
            'RL_MIN_TEMP'
        ]
        
        # Access imported config values from the global scope
        config_values = {
            'RL_AGGREGATION_METHOD': RL_AGGREGATION_METHOD,
            'RL_ACTOR_HIDDEN_DIMS': RL_ACTOR_HIDDEN_DIMS,
            'RL_CRITIC_HIDDEN_DIMS': RL_CRITIC_HIDDEN_DIMS,
            'RL_LEARNING_RATE': RL_LEARNING_RATE,
            'RL_GAMMA': RL_GAMMA,
            'RL_ENTROPY_COEF': RL_ENTROPY_COEF,
            'RL_WARMUP_ROUNDS': RL_WARMUP_ROUNDS,
            'RL_RAMP_UP_ROUNDS': RL_RAMP_UP_ROUNDS,
            'RL_INITIAL_TEMP': RL_INITIAL_TEMP,
            'RL_MIN_TEMP': RL_MIN_TEMP
        }
        
        config_status = {}
        all_configs_present = True
        
        # Check each required config
        for config_name in required_configs:
            if config_name in config_values:
                config_status[config_name] = True
                value = config_values[config_name]
                print(f"{config_name}: {value}")
            else:
                config_status[config_name] = False
                all_configs_present = False
                print(f"{config_name}: MISSING")
        
        # Print summary
        print("\n----- Configuration Summary -----")
        if all_configs_present:
            print("✅ All required RL configurations are present")
        else:
            print("❌ Some required RL configurations are missing")
        
        # Check specific configuration values
        method = RL_AGGREGATION_METHOD
        if method not in ['dual_attention', 'rl_actor_critic', 'hybrid']:
            print(f"⚠️ RL_AGGREGATION_METHOD is set to '{method}', which is not a recognized value")
            print("   Valid values are: 'dual_attention', 'rl_actor_critic', 'hybrid'")
        else:
            print(f"✅ RL_AGGREGATION_METHOD is set to a valid value: '{method}'")
        
        # Verify hybrid mode settings
        if method == 'hybrid':
            print("\n----- Hybrid Mode Configuration -----")
            
            # Check warmup rounds
            warmup = RL_WARMUP_ROUNDS
            if warmup < 1:
                print(f"⚠️ RL_WARMUP_ROUNDS is set to {warmup}, which may be too low")
            else:
                print(f"✅ RL_WARMUP_ROUNDS is set to {warmup}")
            
            # Check ramp-up rounds
            rampup = RL_RAMP_UP_ROUNDS
            if rampup < 1:
                print(f"⚠️ RL_RAMP_UP_ROUNDS is set to {rampup}, which may be too low")
            else:
                print(f"✅ RL_RAMP_UP_ROUNDS is set to {rampup}")
        
        # Check device consistency
        if torch.cuda.is_available():
            print("\n----- Device Configuration -----")
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
            
            # Check if our config is using GPU
            if str(config_device).startswith('cuda'):
                print(f"✅ Config device is set to: {config_device}")
            else:
                print(f"⚠️ Config device might not be using GPU: {config_device}")
        
        return all_configs_present
    
    except Exception as e:
        print(f"Error checking configuration: {e}")
        traceback.print_exc()
        return False

def verify_device_consistency():
    """Verify that device usage is consistent across the codebase"""
    print("\n===================================================")
    print("            CHECKING DEVICE CONSISTENCY             ")
    print("===================================================\n")
    
    try:
        # Create a server instance to check its device
        from federated_learning.training.server import Server
        server = Server()
        
        # Check ActorCritic model device
        from federated_learning.models.rl_actor_critic import ActorCritic
        actor_critic = ActorCritic()
        
        # Report device assignments
        print(f"Config device: {config_device}")
        print(f"Server device: {server.device}")
        print(f"Actor-Critic device: {next(actor_critic.parameters()).device}")
        
        # Check consistency
        devices_match = (str(config_device) == str(server.device) and 
                         str(config_device) == str(next(actor_critic.parameters()).device))
        
        if devices_match:
            print("\n✅ Device usage is consistent across the codebase")
        else:
            print("\n❌ Device usage is inconsistent! This may cause errors.")
            print("   Consider using only one device or ensuring proper device transfers.")
        
        return devices_match
    
    except Exception as e:
        print(f"Error verifying device consistency: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        # Check configuration first
        config_ok = check_rl_config()
        
        # Verify device consistency
        device_ok = verify_device_consistency()
        
        if not config_ok or not device_ok:
            print("\n⚠️ Configuration or device issues detected. Tests may fail.")
        
        # Run all tests
        success = run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error running tests: {e}")
        traceback.print_exc()
        sys.exit(1) 