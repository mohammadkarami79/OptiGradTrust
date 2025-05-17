def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_try_block = False
    
    for line in lines:
        if 'try:' in line:
            in_try_block = True
            fixed_lines.append(line)
        elif in_try_block and 'RE_val = ' in line and line.strip().startswith('RE_val'):
            # Fix the indentation of the RE_val line
            fixed_lines.append(' ' * 20 + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 