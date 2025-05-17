def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    base_indent = ' ' * 20  # Base indentation level
    
    for line in lines:
        if 'print("-" * 100)' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("=' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("Detection Metrics:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("Accuracy:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("Precision:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("Recall:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("F1 Score:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("True Positives:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("True Negatives:' in line:
            fixed_lines.append(base_indent + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 