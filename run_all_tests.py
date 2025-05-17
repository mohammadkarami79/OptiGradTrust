import sys
import os
import time
import importlib
from datetime import datetime

def run_test(test_name):
    """Run a single test and return its result."""
    print(f"\n{'='*80}")
    print(f"Running test: {test_name}")
    print(f"{'='*80}")
    
    try:
        # Import the test module dynamically
        test_module = importlib.import_module(test_name.replace('.py', ''))
        
        # Find a test function to run
        test_function = None
        for func_name in dir(test_module):
            if func_name.startswith('test_'):
                test_function = getattr(test_module, func_name)
                break
        
        if test_function is None:
            print(f"No test function found in {test_name}")
            return False
        
        # Run the test
        start_time = time.time()
        result = test_function()
        end_time = time.time()
        
        # Report result
        elapsed = end_time - start_time
        status = "PASSED" if result else "FAILED"
        print(f"\n{test_name}: {status} (elapsed: {elapsed:.2f}s)")
        
        return result
    except Exception as e:
        print(f"\nError running {test_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests in the project."""
    # Get all test files
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    
    # Sort by priority (dimension test first, then others)
    priority_order = {
        'test_dimension_fix.py': 0,
        'test_federated_learning.py': 1
    }
    
    test_files.sort(key=lambda x: priority_order.get(x, 999))
    
    # Print header
    print(f"\n{'='*80}")
    print(f" Federated Learning Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Found {len(test_files)} tests to run\n")
    
    # Run tests
    results = {}
    for test_file in test_files:
        results[test_file] = run_test(test_file)
    
    # Print summary
    print(f"\n{'='*80}")
    print(" Test Summary")
    print(f"{'='*80}")
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for test_file, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_file}: {status}")
    
    print(f"\nTotal: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    # Return success if all tests pass
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 