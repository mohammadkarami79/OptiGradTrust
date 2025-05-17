def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    current_indent = 0
    base_indent = ' ' * 4  # Standard Python indentation
    
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            fixed_lines.append('\n')
            continue
        
        # Adjust indentation based on context
        if i > 870 and i < 890:  # Around line 877-880
            if 'print("=" * 80)' in line:
                current_indent = 3
            elif 'feature_arrays =' in line or 'feature_means =' in line or 'feature_stds =' in line:
                current_indent = 3
        
        # Add the line with correct indentation
        fixed_lines.append(base_indent * current_indent + stripped_line + '\n')
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 