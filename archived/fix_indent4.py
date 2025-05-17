def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_else_block = False
    
    for i, line in enumerate(lines):
        if 'else:' in line and '# If no client_gradients provided' in lines[i+1]:
            in_else_block = True
            fixed_lines.append(line)
        elif in_else_block and 'mean_neighbor_sim = 0.0' in line:
            # Fix the indentation of the mean_neighbor_sim line
            fixed_lines.append(' ' * 12 + line.lstrip())
            in_else_block = False
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 