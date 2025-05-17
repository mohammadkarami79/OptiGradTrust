import torch
import torch.nn as nn
import numpy as np
import time
import gc
import os
import matplotlib.pyplot as plt
from datetime import datetime

from federated_learning.config.config import *
from federated_learning.models.cnn import CNNMnist
from federated_learning.models.resnet import ResNet18Alzheimer
from federated_learning.training.server import Server
from federated_learning.training.client import Client
from federated_learning.utils.model_utils import update_model_with_gradient

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

class MemoryProfiler:
    """Simple memory profiler for tracking memory usage"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_stats = []
        self.timestamp = []
        self.markers = []
        self.marker_positions = []
        
    def reset(self):
        """Reset memory stats"""
        self.memory_stats = []
        self.timestamp = []
        self.markers = []
        self.marker_positions = []
        
    def track(self, marker=None):
        """Track current memory usage"""
        # Force garbage collection first
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
            torch.cuda.empty_cache()
        else:
            # For CPU, we can't easily measure memory usage, so just use a placeholder
            mem_allocated = 0
            
        self.memory_stats.append(mem_allocated)
        self.timestamp.append(time.time())
        
        if marker:
            self.markers.append(marker)
            self.marker_positions.append(len(self.memory_stats) - 1)
            
        return mem_allocated
    
    def plot(self, title="Memory Usage", save_path=None):
        """Plot memory usage over time"""
        if not self.memory_stats:
            print("No memory statistics collected")
            return
            
        # Adjust timestamps to be relative to start
        start_time = self.timestamp[0]
        rel_timestamps = [(t - start_time) for t in self.timestamp]
        
        plt.figure(figsize=(12, 6))
        plt.plot(rel_timestamps, self.memory_stats, 'b-')
        
        # Add markers
        for marker, pos in zip(self.markers, self.marker_positions):
            plt.plot(rel_timestamps[pos], self.memory_stats[pos], 'ro')
            plt.annotate(marker, 
                        (rel_timestamps[pos], self.memory_stats[pos]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()

def test_gradient_chunking():
    """Test memory usage with and without gradient chunking"""
    print("\n=== Testing Gradient Chunking Memory Optimization ===")
    
    # Create output directory
    output_dir = "test_results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize memory profiler
    profiler = MemoryProfiler()
    
    # Test with different models and chunk sizes
    models = {
        "CNNMnist": CNNMnist(),
        "ResNet18": ResNet18Alzheimer()
    }
    
    chunk_sizes = [None, 1000, 10000]  # None means no chunking
    
    # Create large dummy gradient
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTesting with {model_name} model")
        model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model has {model_param_count:,} parameters")
        
        # Create a random gradient with parameter size
        gradient = torch.randn(model_param_count)
        print(f"  Gradient size: {gradient.shape[0]:,} elements")
        
        model_results = {}
        
        for chunk_size in chunk_sizes:
            print(f"\n  Testing with chunk_size = {chunk_size if chunk_size else 'None'} (no chunking)")
            
            # Enable/disable chunking
            GRADIENT_CHUNK_SIZE = chunk_size
            
            # Reset profiler
            profiler.reset()
            
            # First, track baseline memory
            profiler.track("Start")
            
            # Copy model to ensure we start fresh
            model_copy = type(model)()  # Create a new instance
            with torch.no_grad():
                for target_param, source_param in zip(model_copy.parameters(), model.parameters()):
                    target_param.copy_(source_param)
            
            # Move model to target device if needed
            if torch.cuda.is_available():
                model_copy = model_copy.cuda()
                gradient = gradient.cuda()
                
            profiler.track("Model loaded")
            
            # Time the operation
            start_time = time.time()
            
            # Update model with gradient using current chunk size
            if chunk_size:
                # Manually chunk update
                original_params = {}
                for name, param in model_copy.named_parameters():
                    if param.requires_grad:
                        original_params[name] = param.clone()
                
                # Apply updates in chunks
                start_idx = 0
                update_count = 0
                
                profiler.track("Before chunking")
                
                while start_idx < gradient.shape[0]:
                    # Get chunk indices
                    end_idx = min(start_idx + chunk_size, gradient.shape[0])
                    current_chunk = gradient[start_idx:end_idx]
                    
                    # Apply chunk updates to parameters
                    chunk_start_idx = start_idx
                    for name, param in model_copy.named_parameters():
                        if not param.requires_grad:
                            continue
                            
                        param_size = param.numel()
                        if chunk_start_idx < param_size:
                            # This parameter is part of current chunk
                            param_end_idx = min(chunk_start_idx + current_chunk.shape[0], param_size)
                            param_chunk_size = param_end_idx - chunk_start_idx
                            
                            if param_chunk_size > 0:
                                grad_chunk = current_chunk[:param_chunk_size]
                                param.data.view(-1)[chunk_start_idx:param_end_idx] -= 0.01 * grad_chunk
                                current_chunk = current_chunk[param_chunk_size:]
                                update_count += param_chunk_size
                                
                        chunk_start_idx = max(0, chunk_start_idx - param_size)
                        
                    start_idx = end_idx
                    profiler.track(f"Chunk {start_idx//chunk_size}")
                    
                print(f"    Applied {update_count} parameter updates in chunks")
            else:
                # Standard update without chunking
                try:
                    updated_model, total_change, avg_change = update_model_with_gradient(
                        model_copy, 
                        gradient, 
                        learning_rate=0.01,
                        proximal_mu=0.0,
                        preserve_bn=False
                    )
                    profiler.track("Full update")
                    print(f"    Applied full gradient update with total change: {total_change:.4f}")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"    ❌ Out of memory error: {str(e)}")
                        # Mark peak memory and continue
                        profiler.track("OOM Error")
                    else:
                        raise e
            
            # Calculate memory usage
            max_memory = max(profiler.memory_stats)
            
            # Calculate time
            end_time = time.time()
            duration = end_time - start_time
            
            model_results[chunk_size] = {
                'max_memory': max_memory,
                'duration': duration
            }
            
            # Plot memory usage
            title = f"{model_name} Memory Usage (Chunk Size: {chunk_size if chunk_size else 'None'})"
            save_path = os.path.join(output_dir, f"memory_{model_name}_{chunk_size if chunk_size else 'none'}.png")
            profiler.plot(title=title, save_path=save_path)
            
            print(f"    Peak memory usage: {max_memory:.2f} MB")
            print(f"    Duration: {duration:.4f} seconds")
            
            # Clean up
            del model_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        # Store results for this model
        results[model_name] = model_results
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for model_name, model_results in results.items():
        chunk_sizes_with_names = [str(size) if size else "None" for size in chunk_sizes]
        memory_usage = [model_results[size]['max_memory'] for size in chunk_sizes]
        
        plt.bar(
            [f"{model_name}-{size}" for size in chunk_sizes_with_names],
            memory_usage,
            label=model_name
        )
    
    plt.ylabel('Peak Memory Usage (MB)')
    plt.xlabel('Model - Chunk Size')
    plt.title('Memory Usage Comparison: With vs Without Chunking')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "memory_comparison.png")
    plt.savefig(save_path)
    plt.close()
    
    # Print summary
    print("\nSummary of memory optimization results:")
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        baseline = model_results[None]['max_memory']
        for chunk_size, data in model_results.items():
            if chunk_size:  # Skip the baseline (None)
                memory_reduction = (baseline - data['max_memory']) / baseline * 100
                print(f"  Chunk size {chunk_size}: {data['max_memory']:.2f} MB " +
                      f"({memory_reduction:.1f}% reduction, {data['duration']:.4f}s)")
    
    return results

def test_model_state_management():
    """Test memory usage with explicit model state management"""
    print("\n=== Testing Model State Management ===")
    
    # Create output directory
    output_dir = "test_results/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize memory profiler
    profiler = MemoryProfiler()
    
    # Initialize a server which will manage models
    server = Server()
    
    # Create a large model
    model = ResNet18Alzheimer()
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size:,} parameters")
    
    # Test with different numbers of clients
    num_clients_options = [5, 10, 20]
    
    results = {}
    
    # Test 1: Naive approach - keep all models in memory
    print("\nTesting naive approach (all models in memory):")
    profiler.reset()
    profiler.track("Start")
    
    # Create models for all clients
    all_client_models = []
    for i in range(max(num_clients_options)):
        client_model = type(model)()  # Create new instance
        if torch.cuda.is_available():
            client_model = client_model.cuda()
        all_client_models.append(client_model)
        
        # Track memory after each client model created
        if (i+1) in num_clients_options:
            profiler.track(f"{i+1} models")
            
            # Store the result
            results[f"naive_{i+1}"] = {
                'memory': profiler.memory_stats[-1],
                'models': i+1
            }
    
    # Clean up
    del all_client_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Test 2: Optimized approach - state dict management
    print("\nTesting optimized approach (state dict management):")
    profiler.reset()
    profiler.track("Start")
    
    # Server has one model reference
    global_model = type(model)()
    if torch.cuda.is_available():
        global_model = global_model.cuda()
        
    profiler.track("Global model")
    
    # Create state dicts for clients instead of models
    all_client_states = []
    for i in range(max(num_clients_options)):
        # Get the state dict
        client_state = global_model.state_dict()
        
        # Copy to CPU if needed
        if torch.cuda.is_available():
            client_state = {k: v.cpu() for k, v in client_state.items()}
            
        all_client_states.append(client_state)
        
        # Track memory after each state dict created
        if (i+1) in num_clients_options:
            profiler.track(f"{i+1} states")
            
            # Store the result
            results[f"optimized_{i+1}"] = {
                'memory': profiler.memory_stats[-1],
                'models': i+1
            }
    
    # Now test loading state dicts back to the model
    profiler.track("Before load")
    
    # Load each state back (one at a time, as would happen in practice)
    for i, state in enumerate(all_client_states):
        # Move state back to device if needed
        if torch.cuda.is_available():
            state = {k: v.cuda() for k, v in state.items()}
            
        # Load state into model
        global_model.load_state_dict(state)
        
        # Do some dummy computation to simulate usage
        dummy_input = torch.randn(1, 3, 128, 128)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        _ = global_model(dummy_input)
        
        # Track memory
        if (i+1) % 5 == 0:
            profiler.track(f"Load {i+1}")
    
    # Plot memory usage
    title = "Memory Usage: State Dict vs Full Model"
    save_path = os.path.join(output_dir, "memory_state_management.png")
    profiler.plot(title=title, save_path=save_path)
    
    # Create comparison bar chart
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    labels = []
    naive_memory = []
    optimized_memory = []
    
    for num_clients in num_clients_options:
        labels.append(f"{num_clients} clients")
        naive_memory.append(results[f"naive_{num_clients}"]['memory'])
        optimized_memory.append(results[f"optimized_{num_clients}"]['memory'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, naive_memory, width, label='Naive (Full Models)')
    plt.bar(x + width/2, optimized_memory, width, label='Optimized (State Dicts)')
    
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Number of Clients')
    plt.title('Memory Usage Comparison: Full Models vs State Dicts')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "memory_state_comparison.png")
    plt.savefig(save_path)
    plt.close()
    
    # Print summary
    print("\nSummary of state management results:")
    for num_clients in num_clients_options:
        naive = results[f"naive_{num_clients}"]['memory']
        optimized = results[f"optimized_{num_clients}"]['memory']
        reduction = (naive - optimized) / naive * 100
        
        print(f"  {num_clients} clients:")
        print(f"    Naive: {naive:.2f} MB")
        print(f"    Optimized: {optimized:.2f} MB")
        print(f"    Reduction: {reduction:.1f}%")
    
    return results

if __name__ == "__main__":
    print(f"=== MEMORY OPTIMIZATION TESTS ===")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test gradient chunking
    chunking_results = test_gradient_chunking()
    
    # Test model state management
    state_results = test_model_state_management() 