#!/usr/bin/env python3
"""
NIS Protocol v3.0 Project Organization Script
Clean up root folder and organize files for v3.1 development
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """Organize the project structure"""
    print("🧹 Organizing NIS Protocol v3.0 Project Structure...")
    
    # Create dev directories
    dev_dirs = [
        "dev/experimental",
        "dev/testing", 
        "dev/v31-development",
        "dev/documentation"
    ]
    
    for dir_path in dev_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Files to move to dev/experimental
    experimental_files = [
        "step1_main.py", "step2_main.py", "step3_main.py",
        "ultra_simple_main.py", "simple_main.py",
        "Dockerfile.simple", "Dockerfile.step1", "Dockerfile.step2", 
        "Dockerfile.step3", "Dockerfile.ultra"
    ]
    
    # Files to move to dev/testing
    testing_files = [
        "test_api.py", "test_v3_api.py", "test_full_implementation.py",
        "test_endpoints.py", "test_results_tracker.py",
        "quick_nis_benchmark.py", "run_nis_benchmarks.py",
        "quick_system_check.py", "validate_env_config.py", 
        "validate_environment.py", "quick_benchmark_results.json"
    ]
    
    # Files to move to dev/v31-development
    v31_files = [
        "v31_main.py"
    ]
    
    # Files to move to dev/documentation
    doc_files = [
        "DOCUMENTATION_UPDATE_SUMMARY.md",
        "DOCUMENTATION_VALIDATION_REPORT.md",
        "DOCKER_README.md",
        "POSTMAN_IMPORT_GUIDE.md",
        "MEETING_READY_SUMMARY.md",
        "AWS_MIGRATION_ACCELERATOR_GUIDE.md",
        "NIS_OFFLINE_MODEL_TRAINING_PLAN.md",
        "NIS_PROTOCOL_V3_COMPREHENSIVE_ANALYSIS.md",
        "audit-report.json"
    ]
    
    # Move files
    file_moves = [
        (experimental_files, "dev/experimental"),
        (testing_files, "dev/testing"),
        (v31_files, "dev/v31-development"),
        (doc_files, "dev/documentation")
    ]
    
    for files, destination in file_moves:
        for file_name in files:
            if os.path.exists(file_name):
                try:
                    shutil.move(file_name, os.path.join(destination, file_name))
                    print(f"✅ Moved {file_name} → {destination}/")
                except Exception as e:
                    print(f"⚠️ Could not move {file_name}: {e}")
            else:
                print(f"⚠️ File not found: {file_name}")
    
    print("\n📋 Essential files remaining in root:")
    print("=" * 50)
    
    # List essential files that should remain in root
    essential_files = [
        "README.md",
        "LICENSE", "LICENSE_BSL", "LICENSING_FAQ.md",
        "start.sh", "stop.sh", "reset.sh",
        "docker-compose.yml", "Dockerfile", "nginx.conf",
        "requirements.txt", "requirements_enhanced_infrastructure.txt",
        "main.py", "setup.py",
        "NIS_Protocol_v3_COMPLETE_Postman_Collection.json",
        ".env", ".env.example", ".gitignore", ".dockerignore"
    ]
    
    remaining_files = [f for f in os.listdir(".") if os.path.isfile(f)]
    
    for file_name in remaining_files:
        if file_name in essential_files:
            print(f"✅ {file_name} (essential)")
        else:
            print(f"⚠️ {file_name} (consider organizing)")
    
    print(f"\n🎯 Project organization complete!")
    print(f"📁 Development files moved to dev/ subdirectories")
    print(f"🚀 Ready for v3.0 commit and v3.1 branch creation!")

if __name__ == "__main__":
    organize_project() 