#!/usr/bin/env python3
"""
🔧 NIS Protocol v3.2 - Warning Fix Utility
Automatically installs missing dependencies to resolve backend warnings
"""

import subprocess
import sys
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a Python package using pip"""
    try:
        logger.info(f"📦 Installing {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully installed {package}")
            return True
        else:
            logger.error(f"❌ Failed to install {package}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error installing {package}: {e}")
        return False

def main():
    """Fix all NIS Protocol warnings by installing missing dependencies"""
    logger.info("🚀 NIS Protocol v3.2 - Fixing Backend Warnings")
    logger.info("=" * 60)
    
    # Core packages to fix warnings
    packages = [
        # Visualization dependencies (fixes seaborn warning)
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0", 
        "plotly>=5.17.0",
        "networkx>=3.0",
        
        # Transformers compatibility (fixes Keras 3 issue)
        "tf-keras>=2.16.0",
        
        # Tech stack dependencies (fixes tech stack warnings)
        "kafka-python>=2.0.2",
        "redis>=4.5.0",
        
        # Scientific computing
        "scipy>=1.10.0",
        "psutil>=5.9.0",
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    logger.info("=" * 60)
    logger.info(f"📊 Installation Summary: {success_count}/{total_count} packages installed")
    
    if success_count == total_count:
        logger.info("🎉 All dependencies installed successfully!")
        logger.info("🔄 Restart your NIS Protocol container to apply changes")
    else:
        logger.warning(f"⚠️ {total_count - success_count} packages failed to install")
        logger.info("💡 Some warnings may persist - check container logs")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)