import re

file_path = 'federated_learning/training/training_utils.py'

# Read the content of the file
with open(file_path, 'r') as file:
    content = file.read()

# Replace the import statement
new_content = re.sub(
    r'from torch.utils.data import DataLoader, TensorDataset',
    'from torch.utils.data import DataLoader, TensorDataset, random_split',
    content
)

# Write the updated content back to the file
with open(file_path, 'w') as file:
    file.write(new_content)

print("Import statement updated successfully.") 