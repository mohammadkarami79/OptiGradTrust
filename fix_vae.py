import re

def fix_vae_file():
    print("Fixing indentation issues in the VAE module...")
    
    # Path to the VAE file
    file_path = "federated_learning/models/vae.py"
    
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Fix specific lines with indentation issues
    for i in range(len(lines)):
        # Fix the else block in VAE class
        if i > 0 and "else:" in lines[i] and "self.final_layer" in lines[i+1] and lines[i].strip() == "else:":
            leading_spaces = lines[i].index("else:")
            lines[i+1] = " " * (leading_spaces + 4) + lines[i+1].lstrip()
            lines[i+2] = " " * (leading_spaces + 8) + lines[i+2].lstrip()
            lines[i+3] = " " * (leading_spaces + 8) + lines[i+3].lstrip()
            lines[i+4] = " " * (leading_spaces + 4) + lines[i+4].lstrip()
        
        # Fix _init_weights method
        if "_init_weights" in lines[i] and i+2 < len(lines) and "if isinstance" in lines[i+2]:
            leading_spaces = lines[i+1].index("for")
            if lines[i+2].index("if") < leading_spaces + 4:
                lines[i+2] = " " * (leading_spaces + 4) + lines[i+2].lstrip()
        
        # Fix encode method
        if "def encode" in lines[i]:
            for j in range(i, min(i+10, len(lines))):
                if "if self.use_projection" in lines[j]:
                    # Find the line with h = self.encoder(x)
                    for k in range(j, min(j+5, len(lines))):
                        if "h = self.encoder" in lines[k]:
                            leading_spaces = lines[j].index("if")
                            if lines[k].index("h") < leading_spaces:
                                lines[k] = " " * leading_spaces + lines[k].lstrip()
                            break
                    break
        
        # Fix other indentation in GradientVAE encode method
        if "def encode(self, x):" in lines[i] and "Process in chunks" in lines[i+1]:
            # Next line should have consistent indentation
            leading_spaces = " " * 8  # Standard indentation for method body
            for j in range(i+1, min(i+10, len(lines))):
                if "if self.use_projection" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
                    # Next line should be further indented
                    if j+1 < len(lines) and "x = self.projection" in lines[j+1]:
                        lines[j+1] = " " * 12 + lines[j+1].lstrip()
                if "h = self.encoder" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
                    break
        
        # Fix reparameterize method
        if "def reparameterize" in lines[i]:
            leading_spaces = " " * 8  # Standard indentation
            for j in range(i+1, min(i+5, len(lines))):
                if "std = torch.exp" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
                if "eps = torch.randn" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
                if "return mu + eps" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
        
        # Fix calculate_reconstruction_error method
        if "def calculate_reconstruction_error" in lines[i] and "x" in lines[i]:
            leading_spaces = " " * 8  # Standard indentation
            for j in range(i+1, min(i+5, len(lines))):
                if "with torch.no_grad" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
                if "recon_x" in lines[j] and "self.forward" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                if "error = F.mse_loss" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
        
        # Fix get_reconstruction_error method
        if "def get_reconstruction_error" in lines[i]:
            leading_spaces = " " * 8  # Standard indentation
            for j in range(i+1, min(i+10, len(lines))):
                if "with torch.no_grad" in lines[j]:
                    lines[j] = leading_spaces + lines[j].lstrip()
                if "self.eval()" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                if "if self.use_projection" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                    if j+1 < len(lines) and "x_projected" in lines[j+1]:
                        lines[j+1] = " " * 16 + lines[j+1].lstrip()  # Further nested
                if "else:" in lines[j] and j+1 < len(lines) and "x_projected" in lines[j+1]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                    lines[j+1] = " " * 16 + lines[j+1].lstrip()  # Further nested
                if "recon_x" in lines[j] and "self.forward" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                if "recon_error" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                if "normalized_error" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
                if "return normalized_error" in lines[j]:
                    lines[j] = " " * 12 + lines[j].lstrip()  # Nested inside with block
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    
    print("VAE module fixed successfully!")

if __name__ == "__main__":
    fix_vae_file() 