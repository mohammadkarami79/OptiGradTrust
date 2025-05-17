import torch
import torchvision.models as models

# Create a ResNet50 model
resnet50 = models.resnet50()

# Get all BatchNorm modules
bn_modules = [(name, module) for name, module in resnet50.named_modules() 
              if isinstance(module, torch.nn.BatchNorm2d)]

print(f"ResNet50 has {len(bn_modules)} BatchNorm2d modules")
print("\nSample BatchNorm module names:")
for name, _ in bn_modules[:5]:
    print(f"  {name}")

# Get all parameters related to BatchNorm
bn_params = []
for module_name, module in bn_modules:
    for param_name, param in module.named_parameters():
        full_name = f"{module_name}.{param_name}"
        bn_params.append((full_name, param.shape))

print(f"\nResNet50 has {len(bn_params)} BatchNorm trainable parameters")
print("\nSample BatchNorm parameter names:")
for name, shape in bn_params[:5]:
    print(f"  {name}: {shape}")

# Check if they contain 'bn', 'batch', etc.
contains_bn = [name for name, _ in bn_params if 'bn' in name.lower()]
contains_batch = [name for name, _ in bn_params if 'batch' in name.lower()]
contains_running = [name for name, _ in bn_params if 'running' in name.lower()]

print(f"\nParameters containing 'bn': {len(contains_bn)}")
print(f"Parameters containing 'batch': {len(contains_batch)}")
print(f"Parameters containing 'running': {len(contains_running)}")

# Check buffer parameters (running mean/var)
bn_buffers = []
for module_name, module in bn_modules:
    for buffer_name, buffer in module.named_buffers():
        full_name = f"{module_name}.{buffer_name}"
        bn_buffers.append((full_name, buffer.shape))

print(f"\nResNet50 has {len(bn_buffers)} BatchNorm buffers (running stats)")
print("\nSample BatchNorm buffer names:")
for name, shape in bn_buffers[:5]:
    print(f"  {name}: {shape}")

# Check if buffer names match the filter patterns
contains_running_buffer = [name for name, _ in bn_buffers if 'running' in name.lower()]
print(f"\nBuffers containing 'running': {len(contains_running_buffer)}")

# Get all model parameters and check which ones are BatchNorm related
all_params = list(resnet50.named_parameters())
bn_param_names = [name for name, _ in bn_params]

# Check how FedBN would filter them
fedbn_filtered = []
for name, _ in all_params:
    if ('bn' in name.lower()) or ('running_' in name) or ('num_batches_tracked' in name):
        fedbn_filtered.append(name)

print(f"\nParameters that would be filtered by FedBN: {len(fedbn_filtered)}")
print("\nSample filtered parameters:")
for name in fedbn_filtered[:5]:
    print(f"  {name}")

# Check if any BatchNorm parameters would be missed
missed_params = [name for name in bn_param_names if name not in fedbn_filtered]
print(f"\nBatchNorm parameters that would be missed by FedBN filter: {len(missed_params)}")
if missed_params:
    print("Sample missed parameters:")
    for name in missed_params[:5]:
        print(f"  {name}") 