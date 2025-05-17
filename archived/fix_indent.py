#!/usr/bin/env python

import re

def fix_indentation(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the indentation for the CNN model initialization
    content = re.sub(
        r'if MODEL == \'CNN\':\n\s*self\.global_model',
        'if MODEL == \'CNN\':\n            self.global_model',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    fix_indentation('federated_learning/training/server.py')

print("Indentation fixed in server.py")  
