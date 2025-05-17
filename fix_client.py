import re

def fix_client_file():
    print("Fixing indentation issues in the Client module...")
    
    # Path to the client file
    file_path = "federated_learning/training/client.py"
    
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Fix indentation issues in the else block
    for i in range(len(lines)):
        if "else:" in lines[i] and lines[i].strip() == "else:":
            # Find the indentation level of the else statement
            indent_level = lines[i].index("else:")
            
            # Check if the next line needs fixing
            if i+1 < len(lines) and "self.train_loader" in lines[i+1] and lines[i+1].lstrip().startswith("self.train_loader"):
                if lines[i+1].index("self.train_loader") <= indent_level:
                    # Fix indentation for the dataloader line and its contents
                    lines[i+1] = " " * (indent_level + 4) + lines[i+1].lstrip()  # Add 4 spaces for proper indentation
                    
                    # Fix indentation for parameters
                    for j in range(i+2, i+7):  # Check next few lines that might be part of the dataloader definition
                        if j < len(lines) and any(param in lines[j] for param in ["dataset", "batch_size", "shuffle", "num_workers", "pin_memory"]):
                            lines[j] = " " * (indent_level + 8) + lines[j].lstrip()  # Add 8 spaces (4 for else block + 4 for parameter alignment)
                        elif j < len(lines) and ")" in lines[j] and lines[j].strip() == ")":
                            lines[j] = " " * (indent_level + 4) + lines[j].lstrip()  # Add 4 spaces to align with dataloader call
                            break
                            
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("Client module fixed successfully!")

if __name__ == "__main__":
    fix_client_file() 