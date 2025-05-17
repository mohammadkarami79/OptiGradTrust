def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    base_indent = ' ' * 20  # Base indentation level
    
    for i, line in enumerate(lines):
        # Fix the first indentation error (line 877)
        if 'print("=" * 80)' in line and i > 870 and i < 880:
            fixed_lines.append(base_indent + line.lstrip())
        
        # Fix the second indentation error (line 1031)
        elif 'print("-" * 100)' in line and i > 1025 and i < 1035:
            fixed_lines.append(base_indent + line.lstrip())
        
        # Fix the third indentation error (lines 1034-1037)
        elif 'accuracy = ' in line and i > 1030 and i < 1040:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'precision = ' in line and i > 1030 and i < 1040:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'recall = ' in line and i > 1030 and i < 1040:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'f1 = ' in line and i > 1030 and i < 1040:
            fixed_lines.append(base_indent + line.lstrip())
        
        # Fix the fourth indentation error (lines 1090-1094)
        elif 'print("\\nUpdating dual attention model...")' in line and i > 1085 and i < 1095:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'self.dual_attention.train()' in line and i > 1085 and i < 1095:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'train_dual_attention(' in line and i > 1085 and i < 1095:
            fixed_lines.append(base_indent + line.lstrip())
        
        # Fix the fifth indentation error (line 1140)
        elif 'print("=" * 100)' in line and i > 1135 and i < 1145:
            fixed_lines.append(base_indent + line.lstrip())
        
        # Fix the for loop indentation (around line 1029)
        elif 'for i, score in enumerate(trust_scores):' in line:
            fixed_lines.append(base_indent + line.lstrip())
        elif 'client_id = gradient_dicts[i][\'client_id\']' in line and i > 1025 and i < 1035:
            fixed_lines.append(base_indent + ' ' * 4 + line.lstrip())
        elif 'client = self.clients[client_id]' in line and i > 1025 and i < 1035:
            fixed_lines.append(base_indent + ' ' * 4 + line.lstrip())
        elif 'true_label = "MALICIOUS"' in line and i > 1025 and i < 1035:
            fixed_lines.append(base_indent + ' ' * 4 + line.lstrip())
        elif 'predicted = "DETECTED"' in line and i > 1025 and i < 1035:
            fixed_lines.append(base_indent + ' ' * 4 + line.lstrip())
        elif 'status = "CORRECT"' in line and i > 1025 and i < 1035:
            fixed_lines.append(base_indent + ' ' * 4 + line.lstrip())
        
        else:
            fixed_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py') 