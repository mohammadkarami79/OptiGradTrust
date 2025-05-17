import re

def fix_main_file():
    print("Fixing syntax error in main.py...")
    
    # Path to the file
    file_path = "federated_learning/main.py"
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find and fix the syntax error with the train_dual_attention call
        pattern = r'self\.dual_attention = train_dual_attention\(.*?\)\s*\),\s*,?\s*\),.*?lr=0\.001\s*\)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # Get the matched text
            matched_text = match.group(0)
            print("Found problematic code section:")
            print(matched_text)
            
            # Create the fixed version
            fixed_text = """self.dual_attention = train_dual_attention(
                        gradient_features=X_train,
                        labels=y_train,
                        epochs=50,
                        batch_size=min(16, len(X_train)),  # Smaller batch size for stability
                        lr=0.001
                    )"""
            
            # Replace in the content
            fixed_content = content.replace(matched_text, fixed_text)
            
            # Write the fixed content back to the file
            with open(file_path, 'w') as f:
                f.write(fixed_content)
                
            print("Successfully fixed the syntax error in main.py")
            return True
        else:
            # If the pattern wasn't found, try a more direct approach
            lines = content.split('\n')
            fixed_lines = []
            skip_until_line = None
            
            for i, line in enumerate(lines):
                # Check if we should skip this line
                if skip_until_line is not None and i <= skip_until_line:
                    continue
                
                # Check for the beginning of the problematic section
                if "self.dual_attention = train_dual_attention(" in line and skip_until_line is None:
                    # Add the fixed version of the code
                    fixed_lines.append("                    self.dual_attention = train_dual_attention(")
                    fixed_lines.append("                        gradient_features=X_train,")
                    fixed_lines.append("                        labels=y_train,")
                    fixed_lines.append("                        epochs=50,")
                    fixed_lines.append("                        batch_size=min(16, len(X_train)),  # Smaller batch size for stability")
                    fixed_lines.append("                        lr=0.001")
                    fixed_lines.append("                    )")
                    
                    # Find the end of the problematic section
                    j = i + 1
                    while j < len(lines) and "print(" not in lines[j]:
                        j += 1
                    
                    # Skip all lines until the end of the problematic section
                    skip_until_line = j - 1
                else:
                    fixed_lines.append(line)
            
            # Join the fixed lines and write back to the file
            fixed_content = '\n'.join(fixed_lines)
            with open(file_path, 'w') as f:
                f.write(fixed_content)
                
            print("Fixed the syntax error using line-by-line approach")
            return True
    except Exception as e:
        print(f"Error fixing main.py: {str(e)}")
        return False

if __name__ == "__main__":
    fix_main_file() 