def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    base_indent = ' ' * 8  # Base indentation level
    
    for i, line in enumerate(lines):
        # Fix the first indentation error (line 877)
        if 'print("=" * 80)' in line and i > 870 and i < 880:
            fixed_lines.append(base_indent + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 