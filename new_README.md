# Federated Learning with Advanced Aggregation Methods and Defense Mechanisms

## Project Overview and Goals

This project implements a robust federated learning system with multiple aggregation methods and advanced defense mechanisms against malicious clients. Our primary goals are to:

1. **Develop efficient aggregation methods** that handle non-IID data distributions, particularly with BatchNorm layers
2. **Design robust defense mechanisms** against various types of attacks in federated learning
3. **Compare and evaluate** different aggregation techniques under various attack scenarios
4. **Optimize memory efficiency** for handling large model architectures

## Key Innovations

Our implementation provides several key innovations:

### 1. Enhanced BatchNorm Handling in Federated Learning

- **FedBN**: We've implemented an improved version of FedBN that correctly handles BatchNorm layers:
  - Batch normalization parameters remain local to clients, preserving client-specific data distribution statistics
  - During aggregation, BN parameters are masked to prevent them from being included in the global model
  - Running statistics (mean/variance) are properly preserved between rounds
  - Memory usage is optimized by reducing gradient matrix dimensions

- **FedBN+FedProx**: A novel combination that leverages both approaches:
  - BatchNorm parameters remain local to clients
  - Non-BatchNorm parameters use proximal term regularization for stability
  - Proximal term is selectively applied only to shared parameters

### 2. Comprehensive Aggregation Methods

Our system implements and compares multiple aggregation strategies:

- **FedAvg**: Standard Federated Averaging
- **FedProx**: Adds proximal term to client optimization
- **FedBN**: Keeps batch normalization parameters local to clients
- **FedBN+FedProx**: Our combined approach
- **FedNova**: Normalized averaging based on local steps
- **FedDWA**: Dynamic weighted aggregation based on client performance
- **FedADMM**: ADMM-based federated optimization

### 3. Advanced Malicious Client Detection

- **Dual Attention Mechanism**: A novel approach using self-attention and cross-attention to evaluate client trustworthiness
- **VAE-based Anomaly Detection**: Detects anomalous gradient patterns using a variational autoencoder
- **Gradient Feature Extraction**: Computes diverse statistical features from gradients for better attack detection
- **Shapley Value Integration**: Measures client contribution to model performance (optional)

### 4. Comprehensive Attack Simulations

The system supports various attack types:
- Scaling attacks (full and partial)
- Label flipping attacks
- Sign flipping attacks
- Noise injection attacks
- Min-max and min-sum attacks
- Targeted model poisoning attacks
- Backdoor attacks
- Adaptive attacks

## Current Implementation Details

### FedBN and FedBN+FedProx Implementation

Our improved FedBN implementation correctly handles BatchNorm layers by:

1. **Enhanced BatchNorm Identification**:
   ```python
   # Identify all BatchNorm layers by module type
   for name, module in model.named_modules():
       if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
           bn_layers.add(name)
   
   # Additional check for common BatchNorm naming patterns
   for name, _ in model.named_parameters():
       if '.bn.' in name or 'downsample.1' in name or name.endswith('.bn.weight') or name.endswith('.bn.bias'):
           parts = name.split('.')
           bn_name = '.'.join(parts[:-1])
           bn_layers.add(bn_name)
   ```

2. **Gradient Masking During Aggregation**:
   ```python
   # Create a mask for non-BN parameters
   gradient_mask = torch.ones_like(client_gradients[0])
   for name, (start_idx, end_idx) in param_indices.items():
       if param_is_bn[name]:
           gradient_mask[start_idx:end_idx] = 0.0
   
   # Apply mask to each client gradient
   masked_gradients = [grad * gradient_mask for grad in client_gradients]
   
   # Compute mean of masked gradients
   mean_gradient = torch.mean(torch.stack(masked_gradients), dim=0)
   ```

3. **Client-Side Handling**:
   ```python
   # For FedBN, store BatchNorm gradients separately
   if is_fedbn and is_bn_param:
       bn_grad_list.append(param_diff.view(-1))
       # Insert zeros as placeholders in the main gradient
       grad_list.append(torch.zeros_like(param_diff.view(-1)))
   else:
       grad_list.append(param_diff.view(-1))
   ```

4. **Preserving BatchNorm Statistics**:
   ```python
   # For FedBN, also preserve BatchNorm running mean/variance in the buffers
   if preserve_bn:
       for name, buffer in model.named_buffers():
           # Check if this buffer belongs to a BatchNorm layer
           is_bn_buffer = False
           buffer_path = name.rsplit('.', 1)[0]  # Get module path without buffer name
           
           if buffer_path in bn_layers or 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
               is_bn_buffer = True
           
           if is_bn_buffer:
               # Copy original buffer values to preserve BN statistics
               updated_buffer.copy_(buffer.data)
   ```

### Dual Attention Mechanism

Our dual attention mechanism evaluates client trustworthiness by:

1. **Feature Extraction**: Computing a rich set of features from client gradients
2. **Self-Attention**: Learning relationships between feature components
3. **Cross-Attention**: Comparing client features to trusted reference data
4. **Trust Score Generation**: Producing a trust score for each client
5. **Adaptive Weighting**: Using trust scores to weight client contributions

### Memory Optimization Techniques

The implementation includes several memory optimization techniques:

1. **Gradient Chunking**: Processing large gradients in manageable chunks
2. **Dimension Reduction**: Optional PCA-based dimension reduction for large models
3. **Device Management**: Strategic placement of models between CPU and GPU memory
4. **Aggressive Cleanup**: Optional memory cleanup during processing

## Experimental Results

Our experiments on MNIST, CIFAR-10, and Alzheimer's dataset have shown:

1. **FedBN effectiveness**: Significantly improved performance on non-IID data
2. **FedBN+FedProx benefits**: Better convergence stability than either method alone
3. **Attack resilience**: Successful detection and mitigation of various attack types
4. **Memory efficiency**: Successful handling of large models through optimization techniques

## Configuration and Usage

The system is highly configurable through the `federated_learning/config/config.py` file:

```python
# Aggregation method selection
AGGREGATION_METHOD = 'fedbn'  # Options: 'fedavg', 'fedprox', 'fedadmm', 'fedbn', 'feddwa', 'fednova', 'fedbn_fedprox'

# FedProx parameters
FEDPROX_MU = 0.1  # μ coefficient for proximal term

# FedBN settings
VERBOSE = True  # Set to True for detailed BN layer tracking
```

To run the system with the desired aggregation method:

```bash
cd federated_learning
python main.py
```

## Expected Outcomes and Future Directions

We expect our system to:

1. **Demonstrate superior performance** of FedBN and FedBN+FedProx on non-IID data
2. **Accurately detect** malicious clients under various attack scenarios
3. **Show memory efficiency** benefits from our optimized implementation
4. **Provide insights** into the relative strengths of different aggregation methods

Future work will focus on:

1. **Scale testing** with larger models and datasets
2. **Personalization techniques** to further improve client-specific performance
3. **Integration with more privacy-preserving** technologies
4. **Further optimization** of memory usage for resource-constrained devices

## Development and Extension

The modular architecture makes it easy to:
- Add new aggregation methods
- Implement additional attack types
- Integrate new defense mechanisms
- Support different model architectures
- Experiment with various datasets 