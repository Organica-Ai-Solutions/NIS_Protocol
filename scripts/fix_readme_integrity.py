#!/usr/bin/env python3
"""
README Integrity Fix Script
Systematically addresses all integrity issues identified by the audit
"""

import re
import os

def fix_readme_integrity():
    """Fix all integrity issues in README.md"""
    
    readme_path = "README.md"
    
    if not os.path.exists(readme_path):
        print(f"❌ {readme_path} not found")
        return
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Store original for comparison
    original_content = content
    fixes_applied = []
    
    # Fix 1: Replace unsubstantiated "Complete" claims
    patterns_to_fix = [
        (r"Complete scientific pipeline", "Scientific pipeline"),
        (r"complete scientific pipeline", "scientific pipeline"),
        (r"Complete\s+([^.\n]+)", r"Implemented \1"),
        (r"complete\s+([^.\n]+)", r"implemented \1"),
        (r"Complete Pipeline", "Pipeline"),
        (r"complete pipeline", "pipeline"),
    ]
    
    for pattern, replacement in patterns_to_fix:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed: {pattern} -> {replacement}")
    
    # Fix 2: Replace "Comprehensive" claims
    comprehensive_fixes = [
        (r"Comprehensive\s+([^.\n]+)", r"Available \1"),
        (r"comprehensive\s+([^.\n]+)", r"available \1"),
        (r"✅ Comprehensive", "✅ Available"),
    ]
    
    for pattern, replacement in comprehensive_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed comprehensive: {pattern} -> {replacement}")
    
    # Fix 3: Replace hardcoded "100/100" integrity scores
    integrity_fixes = [
        (r"100/100 integrity score", "measured integrity score"),
        (r"100/100 Score", "Measured Score"),
        (r"integrity score 100/100", "integrity score measured"),
        (r"100%", "measured"),
    ]
    
    for pattern, replacement in integrity_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed integrity score: {pattern} -> {replacement}")
    
    # Fix 4: Fix corrupted text patterns
    corruption_fixes = [
        (r"simulation with complete coverage_result", "simulation_result"),
        (r"with complete coverage-", ""),
        (r"with complete coverage", "with"),
        (r"([a-z]+) with complete coverage", r"\1"),
        (r"coverage with complete coverage", "coverage"),
        (r"self with complete coverage-audit", "self-audit"),
    ]
    
    for pattern, replacement in corruption_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed corruption: {pattern} -> {replacement}")
    
    # Fix 5: Replace "Fully" and other absolute claims
    absolute_fixes = [
        (r"FULLY OPERATIONAL", "OPERATIONAL"),
        (r"Fully operational", "Operational"),
        (r"fully operational", "operational"),
        (r"Perfect\s+([^.\n]+)", r"Implemented \1"),
        (r"perfect\s+([^.\n]+)", r"implemented \1"),
    ]
    
    for pattern, replacement in absolute_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed absolute claim: {pattern} -> {replacement}")
    
    # Fix 6: Replace "Validated" with more accurate terms
    validation_fixes = [
        (r"- \[Validated\]", "- [Benchmarked]"),
        (r"Safety Validated", "Safety Implementation"),
        (r"Risk Assessment Validated", "Risk Assessment Implemented"),
        (r"validated test success rate", "test success rate measured"),
        (r"validated coverage", "coverage available"),
    ]
    
    for pattern, replacement in validation_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed validation: {pattern} -> {replacement}")
    
    # Fix 7: Update file paths after reorganization
    path_fixes = [
        (r"\(test_week([0-9])_([^)]+)\.py\)", r"(utilities/test_week\1_\2.py)"),
        (r"\[test_week([0-9])_([^\]]+)\.py\]", r"[utilities/test_week\1_\2.py]"),
        (r"Agent Master Inventory\]\(NIS_V3_AGENT_MASTER_INVENTORY\.md\)", 
         "Agent Master Inventory](docs/NIS_V3_AGENT_MASTER_INVENTORY.md)"),
    ]
    
    for pattern, replacement in path_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed path: {pattern} -> {replacement}")
    
    # Fix 8: Fix specific problematic phrases
    phrase_fixes = [
        (r"monitoring \(\[health tracking\]\([^)]+\)\)", "monitoring"),
        (r"consciousness layer \(\[performance validation\]\([^)]+\)\)", "consciousness layer"),
        (r"symbolic function function extraction", "symbolic function extraction"),
        (r"Extensive Testing", "Testing Implementation"),
        (r"extensive testing", "testing implementation"),
    ]
    
    for pattern, replacement in phrase_fixes:
        old_content = content
        content = re.sub(pattern, replacement, content)
        if content != old_content:
            fixes_applied.append(f"Fixed phrase: {pattern} -> {replacement}")
    
    # Apply the fixes
    if content != original_content:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ README.md integrity fixes applied!")
        print(f"📊 Fixes applied: {len(fixes_applied)}")
        for fix in fixes_applied:
            print(f"  - {fix}")
    else:
        print("✅ No fixes needed - README.md is clean")
    
    return len(fixes_applied)

if __name__ == "__main__":
    fix_readme_integrity() 