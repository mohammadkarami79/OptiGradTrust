import re

def fix_attention_file():
    print("Fixing indentation issues in the DualAttention module...")
    
    # Path to the attention file
    file_path = "federated_learning/models/attention.py"
    
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Fix _init_weights method indentation issues
    for i in range(len(lines)):
        if "_init_weights" in lines[i]:
            # Process the next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if "for m in self.modules()" in lines[j]:
                    leading_spaces = lines[j].index("for")
                    # Fix the next line (if isinstance)
                    if j+1 < len(lines) and "if isinstance" in lines[j+1]:
                        if lines[j+1].lstrip().startswith("if") and lines[j+1].index("if") <= leading_spaces:
                            lines[j+1] = " " * (leading_spaces + 4) + lines[j+1].lstrip()
                            
                            # Also fix subsequent indentation
                            if j+2 < len(lines) and "nn.init" in lines[j+2]:
                                lines[j+2] = " " * (leading_spaces + 8) + lines[j+2].lstrip()
                            
                            # Fix the if m.bias line and its content
                            if j+3 < len(lines) and "if m.bias" in lines[j+3]:
                                if lines[j+3].lstrip().startswith("if") and lines[j+3].index("if") <= leading_spaces + 4:
                                    lines[j+3] = " " * (leading_spaces + 8) + lines[j+3].lstrip()
                                    if j+4 < len(lines) and "nn.init" in lines[j+4]:
                                        lines[j+4] = " " * (leading_spaces + 12) + lines[j+4].lstrip()
                
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("DualAttention module fixed successfully!")

if __name__ == "__main__":
    fix_attention_file() 