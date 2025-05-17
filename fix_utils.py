import re

def fix_utils_file():
    print("Fixing indentation issues in the training_utils.py file...")
    
    # Path to the file
    file_path = "federated_learning/training/training_utils.py"
    
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Fix the broken try-except block by looking for specific patterns
    try_block_fixed = False
    early_stop_fixed = False
    memory_usage_fixed = False
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix get_process_memory_usage function
        if "def get_process_memory_usage()" in line and not memory_usage_fixed:
            # Add function definition line
            fixed_lines.append(line)
            i += 1
            
            # Add docstring line
            if i < len(lines) and '"""' in lines[i]:
                fixed_lines.append(lines[i])
                i += 1
                
                # Check and fix indentation of the next lines
                while i < len(lines) and "return memory_gb" not in lines[i]:
                    if "process = " in lines[i] or "memory_info = " in lines[i]:
                        # Ensure proper indentation (4 spaces)
                        fixed_lines.append("    " + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
                
                # Add return line
                if i < len(lines) and "return memory_gb" in lines[i]:
                    fixed_lines.append(lines[i])
                    i += 1
                    
                memory_usage_fixed = True
                continue
                
        # Fix the try-except block
        if "try:" in line and "global_context" in lines[i+1] and not try_block_fixed:
            # Keep all try block lines until we reach 'optimizer.step()'
            try_block_lines = [line]  # Start with the try: line
            j = i + 1
            
            while j < len(lines) and "optimizer.step()" not in lines[j]:
                try_block_lines.append(lines[j])
                j += 1
            
            # Now find the optimizer.step() and epoch_loss line
            if j < len(lines) and "optimizer.step()" in lines[j]:
                # Add optimizer.step() inside the try block
                try_block_lines.append("                # Step optimizer and update loss\n")
                try_block_lines.append("                " + lines[j].lstrip())
                j += 1
                
                # Find the epoch_loss line and add it properly indented
                if j < len(lines) and "epoch_loss +=" in lines[j]:
                    try_block_lines.append("                " + lines[j].lstrip())
                    j += 1
                
                # Add the except line
                if j < len(lines) and "except" in lines[j]:
                    try_block_lines.append(lines[j])
                    j += 1
                    
                    # Add all code inside the except block
                    while j < len(lines) and "continue" not in lines[j]:
                        try_block_lines.append(lines[j])
                        j += 1
                    
                    if j < len(lines) and "continue" in lines[j]:
                        try_block_lines.append(lines[j])
                        j += 1
                
                # Add all fixed lines
                fixed_lines.extend(try_block_lines)
                i = j
                try_block_fixed = True
                continue
        
        # Fix early stopping indentation
        if "if no_improve_epochs >=" in line and "early_stopping" in line and not early_stop_fixed:
            indent_level = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            i += 1
            
            # Fix the next two lines (print and break)
            for _ in range(2):
                if i < len(lines):
                    current_indent = len(lines[i]) - len(lines[i].lstrip())
                    if current_indent != indent_level + 4:
                        fixed_lines.append(" " * (indent_level + 4) + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
            
            early_stop_fixed = True
            continue
            
        # Add any other lines as is
        fixed_lines.append(line)
        i += 1
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(fixed_lines)
    
    print("training_utils.py fixed successfully!")

if __name__ == "__main__":
    fix_utils_file() 