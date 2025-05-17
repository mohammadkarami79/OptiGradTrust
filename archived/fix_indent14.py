def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    base_indent = ' ' * 12  # Base indentation level
    
    for i, line in enumerate(lines):
        # Fix indentation for the feature processing section
        if i > 875 and i < 885:
            if 'print("=" * 80)' in line:
                fixed_lines.append(base_indent + line.lstrip())
            elif any(x in line for x in ['feature_arrays = np.array', 'feature_means', 'feature_stds']):
                fixed_lines.append(base_indent + ' ' * 4 + line.lstrip())
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 