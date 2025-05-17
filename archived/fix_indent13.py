def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    base_indent = ' ' * 8  # Base indentation level
    
    for i, line in enumerate(lines):
        # Fix indentation for lines around 880
        if i > 875 and i < 885:
            if any(x in line for x in ['print("=" * 80)', 'feature_arrays = np.array(feature_arrays)', 'feature_arrays = feature_arrays.reshape']):
                fixed_lines.append(base_indent + line.lstrip())
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 