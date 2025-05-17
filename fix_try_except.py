def fix_try_except_block():
    print("Fixing try-except block in training_utils.py...")
    
    filename = "federated_learning/training/training_utils.py"
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # This is the broken try-except block pattern to search for
    broken_pattern = """            try:
                # Use global context as mean of features in this batch
                global_context = torch.mean(features, dim=0, keepdim=True)
                scores, _ = model(features, global_context)
                
                # Ensure scores are properly clamped to avoid numerical issues
                scores = torch.clamp(scores, 1e-7, 1-1e-7)
                
                # Loss
                loss = criterion(scores, batch_labels)
                
                # Accuracy
                predicted = (scores > 0.5).float()
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            
                epoch_loss += loss.item() * features.size(0)
                
            except RuntimeError"""
    
    # This is the corrected version
    fixed_pattern = """            try:
                # Use global context as mean of features in this batch
                global_context = torch.mean(features, dim=0, keepdim=True)
                scores, _ = model(features, global_context)
                
                # Ensure scores are properly clamped to avoid numerical issues
                scores = torch.clamp(scores, 1e-7, 1-1e-7)
                
                # Loss
                loss = criterion(scores, batch_labels)
                
                # Accuracy
                predicted = (scores > 0.5).float()
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Step optimizer and update loss
                optimizer.step()
                epoch_loss += loss.item() * features.size(0)
                
            except RuntimeError"""
    
    # Replace the broken pattern with the fixed pattern
    fixed_content = content.replace(broken_pattern, fixed_pattern)
    
    # If no replacement was made, try a more specific approach
    if content == fixed_content:
        print("Pattern not found. Trying a different approach...")
        
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Find the try block
            if 'try:' in line and 'global_context' in lines[i+1]:
                fixed_lines.append(line)  # Add try line
                
                # Loop until we find optimizer.step or an except block
                j = i + 1
                inside_try_block = True
                
                while j < len(lines) and inside_try_block:
                    current_line = lines[j]
                    
                    # Found optimizer.step outside try block
                    if 'optimizer.step()' in current_line and not current_line.strip().startswith('#'):
                        # We found the optimizer.step outside the try block
                        # Keep adding lines until we reach it
                        while i < j-1:
                            i += 1
                            fixed_lines.append(lines[i])
                        
                        # Add optimizer.step with proper indentation
                        fixed_lines.append("                # Step optimizer and update loss")
                        fixed_lines.append("                optimizer.step()")
                        j += 1
                        
                        # Find the epoch_loss line and add it with proper indentation
                        if j < len(lines) and 'epoch_loss +=' in lines[j]:
                            fixed_lines.append("                " + lines[j].lstrip())
                            j += 1
                        
                        # Now add the except line
                        if j < len(lines) and 'except' in lines[j]:
                            fixed_lines.append(lines[j])
                            inside_try_block = False
                    elif 'except' in current_line:
                        # Found except before optimizer.step - this means try block is already correct
                        fixed_lines.append(current_line)
                        inside_try_block = False
                    else:
                        # Normal line within try block
                        fixed_lines.append(current_line)
                        j += 1
                
                # Skip the lines we've already processed
                i = j
            else:
                fixed_lines.append(line)
                i += 1
        
        # Join all lines and create the fixed content
        fixed_content = '\n'.join(fixed_lines)
    
    # Write the fixed content back to the file
    with open(filename, 'w') as file:
        file.write(fixed_content)
    
    print("Fixed try-except block in training_utils.py!")

if __name__ == "__main__":
    fix_try_except_block() 