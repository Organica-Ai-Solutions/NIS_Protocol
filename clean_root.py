import os
import shutil

def clean_root_directory():
    """
    Cleans the root directory by moving files into their appropriate subdirectories.
    """
    # Create directories if they don't exist
    os.makedirs("docs/drone", exist_ok=True)
    os.makedirs("dev", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)

    # Move drone files
    drone_files = ["DRONE_PROJECT_OVERVIEW.md", "NIS_DRONE_HARDWARE_PLAN.md", "NIS_DRONE_PROJECT_PLAN.md"]
    for f in drone_files:
        if os.path.exists(f):
            shutil.move(f, "docs/drone/")

    # Move documentation files
    doc_files = ["LICENSING_FAQ.md", "COMMERCIAL_LICENSE.md", "LICENSE_BSL", "NIS_Protocol_V3_Whitepaper.md"]
    for f in doc_files:
        if os.path.exists(f):
            shutil.move(f, "docs/")

    # Move development files
    dev_files = [
        "main_v31_complete.py", "main_v31_part2.py", "main_v31_part3.py", "main_v31.py",
        "quick_v31_test.py", "test_v31_complete.py", "v31_test_results.json",
        "NIS_Protocol_v3_COMPLETE_Postman_Collection.json", "NIS_Protocol_v3_Postman_Collection.json",
        "rebuild_v31.sh", "v31_main.py"
    ]
    for f in dev_files:
        if os.path.exists(f):
            shutil.move(f, "dev/")

    # Move helper scripts
    script_files = [
        "organize_project.py", "setup_repo.sh", "implement_licensing.sh", 
        "update_pypi_token.sh", "start.sh", "stop.sh", "reset.sh"
    ]
    for f in script_files:
        if os.path.exists(f):
            shutil.move(f, "scripts/")

    print("Root directory cleaned successfully.")

if __name__ == "__main__":
    clean_root_directory() 