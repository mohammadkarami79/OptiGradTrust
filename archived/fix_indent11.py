def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    base_indent = ' ' * 12  # Base indentation level
    
    for i, line in enumerate(lines):
        # Fix the comprehensive report indentation
        if 'print("COMPREHENSIVE PERFORMANCE REPORT")' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'print("=" * 100)' in line and i > 1135 and i < 1145:
            fixed_lines.append(base_indent + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 