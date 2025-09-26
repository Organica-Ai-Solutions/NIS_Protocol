#!/usr/bin/env python3
"""
Fix indentation issues in unified_coordinator.py
This script will fix all method definitions to have proper indentation
"""

import re

def fix_indentation():
    with open('src/meta/unified_coordinator.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    current_class_indent = 0
    in_class = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track class indentation
        if line.strip().startswith('class '):
            current_class_indent = len(line) - len(line.lstrip())
            in_class = True
            fixed_lines.append(line)
            continue
        elif line.strip().startswith('def ') and in_class:
            # Check if this is a method (inside class)
            current_line_indent = len(line) - len(line.lstrip())
            
            if current_line_indent <= current_class_indent:
                # This is a method - should be indented at class level + 4
                fixed_line = '    ' + line.lstrip()
                fixed_lines.append(fixed_line)
                print(f"Fixed line {i+1}: {line.strip()} -> {fixed_line.strip()}")
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back the fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    with open('src/meta/unified_coordinator.py', 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed indentation issues in {len([l for l in lines if 'def ' in l and l.strip() != 'def '])} methods")

if __name__ == "__main__":
    fix_indentation()
