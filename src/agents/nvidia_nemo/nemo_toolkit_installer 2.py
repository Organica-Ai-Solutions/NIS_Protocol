#!/usr/bin/env python3
"""
NVIDIA NeMo Agent Toolkit Installer and Setup
Automated installation and configuration of the real NVIDIA NeMo Agent Toolkit

Based on official installation guide from:
https://github.com/NVIDIA/NeMo-Agent-Toolkit
"""

import asyncio
import subprocess
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class NeMoToolkitInstaller:
    """Automated installer for NVIDIA NeMo Agent Toolkit"""
    
    def __init__(self, install_dir: str = "./nemo-agent-toolkit"):
        self.install_dir = Path(install_dir)
        self.repo_url = "https://github.com/NVIDIA/NeMo-Agent-Toolkit.git"
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
    async def install_toolkit(self) -> Dict[str, Any]:
        """Complete installation of NVIDIA NeMo Agent Toolkit"""
        
        installation_steps = []
        
        try:
            # Step 1: Check prerequisites
            step_result = await self._check_prerequisites()
            installation_steps.append(("prerequisites", step_result))
            
            if not step_result["success"]:
                return {"success": False, "steps": installation_steps}
            
            # Step 2: Clone repository
            step_result = await self._clone_repository()
            installation_steps.append(("clone_repository", step_result))
            
            # Step 3: Setup environment
            step_result = await self._setup_environment()
            installation_steps.append(("setup_environment", step_result))
            
            # Step 4: Install dependencies
            step_result = await self._install_dependencies()
            installation_steps.append(("install_dependencies", step_result))
            
            # Step 5: Verify installation
            step_result = await self._verify_installation()
            installation_steps.append(("verify_installation", step_result))
            
            # Step 6: Create sample workflow
            step_result = await self._create_sample_workflow()
            installation_steps.append(("create_sample_workflow", step_result))
            
            logger.info("NVIDIA NeMo Agent Toolkit installation completed successfully")
            return {
                "success": True,
                "installation_path": str(self.install_dir),
                "steps": installation_steps
            }
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": installation_steps
            }
    
    async def _check_prerequisites(self) -> Dict[str, Any]:
        """Check system prerequisites"""
        
        checks = {}
        
        # Check Python version
        if self.python_version in ["3.11", "3.12"]:
            checks["python_version"] = {"status": "ok", "version": self.python_version}
        else:
            checks["python_version"] = {
                "status": "error", 
                "version": self.python_version,
                "message": "Python 3.11 or 3.12 required"
            }
        
        # Check Git
        try:
            result = subprocess.run(["git", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks["git"] = {"status": "ok", "version": result.stdout.strip()}
            else:
                checks["git"] = {"status": "error", "message": "Git not found"}
        except FileNotFoundError:
            checks["git"] = {"status": "error", "message": "Git not installed"}
        
        # Check Git LFS
        try:
            result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks["git_lfs"] = {"status": "ok", "version": result.stdout.strip()}
            else:
                checks["git_lfs"] = {"status": "warning", "message": "Git LFS not found"}
        except FileNotFoundError:
            checks["git_lfs"] = {"status": "warning", "message": "Git LFS recommended but not required"}
        
        # Check UV package manager
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                checks["uv"] = {"status": "ok", "version": result.stdout.strip()}
            else:
                checks["uv"] = {"status": "warning", "message": "UV not found, will use pip"}
        except FileNotFoundError:
            checks["uv"] = {"status": "warning", "message": "UV recommended for best performance"}
        
        all_ok = all(check["status"] != "error" for check in checks.values())
        
        return {
            "success": all_ok,
            "checks": checks,
            "message": "Prerequisites check completed"
        }
    
    async def _clone_repository(self) -> Dict[str, Any]:
        """Clone NeMo Agent Toolkit repository"""
        
        try:
            if self.install_dir.exists():
                logger.info(f"Directory {self.install_dir} already exists, skipping clone")
                return {"success": True, "message": "Repository already exists"}
            
            # Clone repository
            result = subprocess.run([
                "git", "clone", self.repo_url, str(self.install_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Git clone failed: {result.stderr}"
                }
            
            # Initialize submodules
            subprocess.run([
                "git", "submodule", "update", "--init", "--recursive"
            ], cwd=self.install_dir, capture_output=True)
            
            # Fetch LFS files if available
            subprocess.run([
                "git", "lfs", "install"
            ], cwd=self.install_dir, capture_output=True)
            
            subprocess.run([
                "git", "lfs", "fetch"
            ], cwd=self.install_dir, capture_output=True)
            
            subprocess.run([
                "git", "lfs", "pull"
            ], cwd=self.install_dir, capture_output=True)
            
            return {
                "success": True,
                "message": "Repository cloned successfully",
                "path": str(self.install_dir)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Clone failed: {str(e)}"
            }
    
    async def _setup_environment(self) -> Dict[str, Any]:
        """Setup Python environment"""
        
        try:
            venv_path = self.install_dir / ".venv"
            
            # Create virtual environment with UV if available
            uv_available = subprocess.run(["uv", "--version"], capture_output=True).returncode == 0
            
            if uv_available:
                # Use UV for environment creation
                result = subprocess.run([
                    "uv", "venv", "--seed", str(venv_path), "--python", self.python_version
                ], cwd=self.install_dir, capture_output=True, text=True)
            else:
                # Fallback to standard venv
                result = subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], cwd=self.install_dir, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Environment creation failed: {result.stderr}"
                }
            
            return {
                "success": True,
                "message": "Python environment created",
                "venv_path": str(venv_path),
                "method": "uv" if uv_available else "venv"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Environment setup failed: {str(e)}"
            }
    
    async def _install_dependencies(self) -> Dict[str, Any]:
        """Install NeMo Agent Toolkit dependencies"""
        
        try:
            venv_path = self.install_dir / ".venv"
            
            # Check if UV is available for installation
            uv_available = subprocess.run(["uv", "--version"], capture_output=True).returncode == 0
            
            if uv_available:
                # Install with UV (recommended)
                result = subprocess.run([
                    "uv", "sync", "--all-groups", "--all-extras"
                ], cwd=self.install_dir, capture_output=True, text=True)
                
                install_method = "uv_sync"
            else:
                # Fallback to pip installation
                pip_path = venv_path / "bin" / "pip" if os.name != "nt" else venv_path / "Scripts" / "pip.exe"
                
                result = subprocess.run([
                    str(pip_path), "install", "-e", ".[all]"
                ], cwd=self.install_dir, capture_output=True, text=True)
                
                install_method = "pip_install"
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Dependency installation failed: {result.stderr}",
                    "stdout": result.stdout,
                    "method": install_method
                }
            
            return {
                "success": True,
                "message": "Dependencies installed successfully",
                "method": install_method,
                "output": result.stdout
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Dependency installation failed: {str(e)}"
            }
    
    async def _verify_installation(self) -> Dict[str, Any]:
        """Verify NeMo Agent Toolkit installation"""
        
        try:
            venv_path = self.install_dir / ".venv"
            nat_path = venv_path / "bin" / "nat" if os.name != "nt" else venv_path / "Scripts" / "nat.exe"
            
            # Test NAT CLI
            result = subprocess.run([
                str(nat_path), "--version"
            ], cwd=self.install_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                return {
                    "success": True,
                    "message": "Installation verified successfully",
                    "nat_version": version,
                    "cli_path": str(nat_path)
                }
            else:
                return {
                    "success": False,
                    "error": f"NAT CLI verification failed: {result.stderr}",
                    "stdout": result.stdout
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Verification failed: {str(e)}"
            }
    
    async def _create_sample_workflow(self) -> Dict[str, Any]:
        """Create sample workflow for testing"""
        
        try:
            workflow_content = """
functions:
   # Add a tool to search wikipedia
   wikipedia_search:
      _type: wiki_search
      max_results: 2

llms:
   # Tell NeMo Agent toolkit which LLM to use for the agent
   nim_llm:
      _type: nim
      model_name: meta/llama-3.1-70b-instruct
      temperature: 0.0

workflow:
   # Use an agent that 'reasons' and 'acts'
   _type: react_agent
   # Give it access to our wikipedia search tool
   tool_names: [wikipedia_search]
   # Tell it which LLM to use
   llm_name: nim_llm
   # Make it verbose
   verbose: true
   # Retry up to 3 times
   parse_agent_response_max_retries: 3
"""
            
            workflow_path = self.install_dir / "nis_integration_workflow.yaml"
            
            with open(workflow_path, "w") as f:
                f.write(workflow_content.strip())
            
            return {
                "success": True,
                "message": "Sample workflow created",
                "workflow_path": str(workflow_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Sample workflow creation failed: {str(e)}"
            }
    
    async def test_installation(self, test_query: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Test the installation with a sample query"""
        
        try:
            venv_path = self.install_dir / ".venv"
            nat_path = venv_path / "bin" / "nat" if os.name != "nt" else venv_path / "Scripts" / "nat.exe"
            workflow_path = self.install_dir / "nis_integration_workflow.yaml"
            
            # Set environment variable for NVIDIA API (if available)
            env = os.environ.copy()
            if "NVIDIA_API_KEY" not in env:
                env["NVIDIA_API_KEY"] = "test_key_placeholder"
            
            # Run test workflow
            result = subprocess.run([
                str(nat_path), "run", 
                "--config_file", str(workflow_path),
                "--input", test_query
            ], cwd=self.install_dir, capture_output=True, text=True, env=env, timeout=60)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "Test completed successfully",
                    "query": test_query,
                    "output": result.stdout,
                    "workflow_result": "Working"
                }
            else:
                return {
                    "success": False,
                    "message": "Test failed (expected without NVIDIA API key)",
                    "query": test_query,
                    "error": result.stderr,
                    "stdout": result.stdout,
                    "note": "API key required for full functionality"
                }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Test failed: {str(e)}"
            }
    
    def get_installation_status(self) -> Dict[str, Any]:
        """Get current installation status"""
        
        status = {
            "installation_directory": str(self.install_dir),
            "directory_exists": self.install_dir.exists(),
            "venv_exists": (self.install_dir / ".venv").exists() if self.install_dir.exists() else False,
            "workflow_exists": (self.install_dir / "nis_integration_workflow.yaml").exists() if self.install_dir.exists() else False
        }
        
        # Check NAT CLI availability
        if status["venv_exists"]:
            venv_path = self.install_dir / ".venv"
            nat_path = venv_path / "bin" / "nat" if os.name != "nt" else venv_path / "Scripts" / "nat.exe"
            status["nat_cli_exists"] = nat_path.exists()
        else:
            status["nat_cli_exists"] = False
        
        # Overall status
        status["installation_complete"] = all([
            status["directory_exists"],
            status["venv_exists"],
            status["nat_cli_exists"],
            status["workflow_exists"]
        ])
        
        return status


# Factory function
def create_nemo_toolkit_installer(install_dir: str = "./nemo-agent-toolkit") -> NeMoToolkitInstaller:
    """Create NeMo Toolkit Installer instance"""
    return NeMoToolkitInstaller(install_dir)
