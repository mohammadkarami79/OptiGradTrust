def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_try_block = False
    
    for line in lines:
        if 'try:' in line:
            in_try_block = True
            fixed_lines.append(line)
        elif in_try_block and 'RE_val = min(RE_val / 10.0, 1.0)' in line:
            # Fix the indentation of the RE_val line
            fixed_lines.append(' ' * 16 + line.lstrip())
        elif in_try_block and 'except Exception as e:' in line:
            in_try_block = False
            fixed_lines.append(line)
        elif 'RE_val = 1.0' in line and line.strip().startswith('RE_val'):
            # Fix the indentation of the fallback RE_val line
            fixed_lines.append(' ' * 12 + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 