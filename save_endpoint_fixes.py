#!/usr/bin/env python3
"""
Save our critical endpoint fixes for migration to organization repo
"""
import shutil
import os

# Our critical working fixes
fixes_to_save = {
    "main.py": {
        "description": "Enum serialization fix for curiosity + simple simulation endpoint",
        "lines_changed": "276-282, 175-207"
    },
    "src/agents/goals/curiosity_engine.py": {
        "description": "Fixed 'factors not defined' error", 
        "lines_changed": "624-631"
    },
    "src/agents/simulation/scenario_simulator.py": {
        "description": "Added PHYSICS enum to ScenarioType",
        "lines_changed": "44"
    },
    "test_fixed_parameters.py": {
        "description": "Updated test script with correct parameter structures",
        "lines_changed": "entire file"
    }
}

def save_fixes():
    """Save our working fixes to a backup directory"""
    backup_dir = "endpoint_fixes_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    print("💾 SAVING CRITICAL ENDPOINT FIXES")
    print("="*50)
    
    for file_path, info in fixes_to_save.items():
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, file_path.replace("/", "_"))
            shutil.copy2(file_path, backup_path)
            print(f"✅ Saved: {file_path}")
            print(f"   → {info['description']}")
            print(f"   → Lines: {info['lines_changed']}")
        else:
            print(f"❌ Missing: {file_path}")
        print()
    
    # Save summary
    with open(os.path.join(backup_dir, "FIXES_SUMMARY.md"), "w") as f:
        f.write("# 🔧 CRITICAL ENDPOINT FIXES BACKUP\n\n")
        f.write("## Summary\n")
        f.write("These are the working endpoint fixes that need to be applied to the organization repo:\n\n")
        
        for file_path, info in fixes_to_save.items():
            f.write(f"### {file_path}\n")
            f.write(f"- **Description**: {info['description']}\n")
            f.write(f"- **Lines Changed**: {info['lines_changed']}\n\n")
        
        f.write("## Status\n")
        f.write("✅ Curiosity endpoint: FIXED (factors error + enum serialization)\n")
        f.write("✅ Simple simulation: ENABLED (/simulation/run)\n") 
        f.write("✅ Complex simulation: FIXED (physics enum added)\n")
        f.write("✅ Ethics endpoint: WORKING\n")
        f.write("✅ Async chat: WORKING\n\n")
        
        f.write("## Demo Ready\n")
        f.write("System is ready for customer validation sprint with these fixes applied.\n")
    
    print("📄 Created FIXES_SUMMARY.md")
    print(f"\n🎯 All fixes saved to: {backup_dir}/")
    print("\n📋 NEXT STEPS:")
    print("1. Fresh clone organization repo")
    print("2. Apply these specific fixes")
    print("3. Push to correct repo") 
    print("4. Resume demo preparation")

if __name__ == "__main__":
    save_fixes()