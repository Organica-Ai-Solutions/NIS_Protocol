#!/usr/bin/env python3
"""
NIS Protocol v3 - Environment Validation Script

Comprehensive pre-flight validation to ensure all dependencies and configurations
are properly set up before system deployment.
"""

import sys
import os
import importlib
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """Comprehensive environment validation for NIS Protocol v3."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        
    def validate_python_version(self) -> bool:
        """Validate Python version requirements."""
        logger.info("🐍 Checking Python version...")
        
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            self.passed_checks.append(f"✅ Python {sys.version.split()[0]} (required: {required_version[0]}.{required_version[1]}+)")
            return True
        else:
            self.errors.append(f"❌ Python {current_version[0]}.{current_version[1]} found, but {required_version[0]}.{required_version[1]}+ required")
            return False
    
    def validate_core_dependencies(self) -> bool:
        """Validate core Python dependencies."""
        logger.info("📦 Checking core dependencies...")
        
        critical_modules = [
            # Web Framework
            ('fastapi', 'FastAPI web framework'),
            ('uvicorn', 'ASGI server'),
            
            # LLM Providers
            ('openai', 'OpenAI SDK'),
            ('anthropic', 'Anthropic SDK'),
            ('tiktoken', 'Token counting'),
            
            # Dashboard
            ('flask', 'Flask web framework'),
            ('flask_socketio', 'Flask-SocketIO for real-time updates'),
            
            # Infrastructure
            ('redis', 'Redis client'),
            ('kafka', 'Kafka client (kafka-python)'),
            
            # Agent Framework
            ('langchain', 'LangChain agent framework'),
            ('langgraph', 'LangGraph workflow framework'),
            
            # Scientific Computing
            ('numpy', 'NumPy for numerical computing'),
            ('scipy', 'SciPy for scientific computing'),
            ('torch', 'PyTorch for deep learning'),
            
            # Configuration
            ('pydantic', 'Data validation'),
            ('python_dotenv', 'Environment variable loading'),
        ]
        
        all_passed = True
        for module_name, description in critical_modules:
            try:
                importlib.import_module(module_name)
                self.passed_checks.append(f"✅ {description} ({module_name})")
            except ImportError:
                self.errors.append(f"❌ Missing critical dependency: {module_name} ({description})")
                all_passed = False
        
        return all_passed
    
    def validate_nis_imports(self) -> bool:
        """Validate NIS Protocol internal imports."""
        logger.info("🧠 Checking NIS Protocol components...")
        
        # Add current directory to Python path for imports
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        nis_components = [
            ('src.cognitive_agents.cognitive_system', 'CognitiveSystem'),
            ('src.agents.consciousness.enhanced_conscious_agent', 'EnhancedConsciousAgent'),
            ('src.infrastructure.integration_coordinator', 'InfrastructureCoordinator'),
            ('src.monitoring.real_time_dashboard', 'RealTimeDashboard'),
            ('src.utils.env_config', 'EnvironmentConfig'),
            ('src.llm.cognitive_orchestra', 'CognitiveOrchestra'),
            ('src.llm.llm_manager', 'LLMManager'),
        ]
        
        all_passed = True
        for module_path, class_name in nis_components:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    self.passed_checks.append(f"✅ {class_name} from {module_path}")
                else:
                    self.errors.append(f"❌ Class {class_name} not found in {module_path}")
                    all_passed = False
            except ImportError as e:
                self.errors.append(f"❌ Cannot import {module_path}: {e}")
                all_passed = False
        
        return all_passed
    
    def validate_api_keys(self) -> bool:
        """Validate LLM provider API keys."""
        logger.info("🔑 Checking API key configuration...")
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            self.warnings.append("⚠️ python-dotenv not available, relying on system environment variables")
        
        api_keys = {
            'OPENAI_API_KEY': 'OpenAI API key',
            'ANTHROPIC_API_KEY': 'Anthropic API key',
            'DEEPSEEK_API_KEY': 'DeepSeek API key (optional)',
            'GOOGLE_API_KEY': 'Google API key (optional)',
        }
        
        required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        optional_keys = ['DEEPSEEK_API_KEY', 'GOOGLE_API_KEY']
        
        has_required = False
        for key, description in api_keys.items():
            value = os.getenv(key)
            
            if value and value not in ['your_openai_api_key_here', 'your_anthropic_api_key_here', 
                                      'your_deepseek_api_key_here', 'your_google_api_key_here']:
                self.passed_checks.append(f"✅ {description} configured")
                if key in required_keys:
                    has_required = True
            elif key in required_keys:
                self.warnings.append(f"⚠️ {description} not configured or is placeholder")
            else:
                self.warnings.append(f"⚠️ {description} not configured (optional)")
        
        if not has_required:
            self.errors.append("❌ At least one LLM provider API key (OpenAI or Anthropic) is required")
            return False
        
        return True
    
    def validate_docker_environment(self) -> bool:
        """Validate Docker environment if running in container."""
        logger.info("🐳 Checking Docker environment...")
        
        # Check if running in Docker
        if os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER'):
            self.passed_checks.append("✅ Running in Docker container")
            
            # Check Docker-specific environment variables
            docker_vars = [
                'DATABASE_URL',
                'KAFKA_BOOTSTRAP_SERVERS', 
                'REDIS_HOST',
                'PYTHONPATH'
            ]
            
            for var in docker_vars:
                if os.getenv(var):
                    self.passed_checks.append(f"✅ Docker environment variable: {var}")
                else:
                    self.warnings.append(f"⚠️ Docker environment variable not set: {var}")
        else:
            self.passed_checks.append("✅ Running in local environment")
        
        return True
    
    def validate_file_system(self) -> bool:
        """Validate required files and directories."""
        logger.info("📁 Checking file system...")
        
        required_files = [
            'main.py',
            'docker-compose.yml',
            'Dockerfile',
            'requirements.txt',
            'requirements_enhanced_infrastructure.txt',
        ]
        
        required_dirs = [
            'src/',
            'src/cognitive_agents/',
            'src/agents/consciousness/',
            'src/infrastructure/',
            'src/llm/',
            'src/monitoring/',
        ]
        
        all_passed = True
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.passed_checks.append(f"✅ Required file: {file_path}")
            else:
                self.errors.append(f"❌ Missing required file: {file_path}")
                all_passed = False
        
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                self.passed_checks.append(f"✅ Required directory: {dir_path}")
            else:
                self.errors.append(f"❌ Missing required directory: {dir_path}")
                all_passed = False
        
        return all_passed
    
    def run_validation(self) -> bool:
        """Run comprehensive validation."""
        logger.info("🚀 Starting NIS Protocol v3 Environment Validation")
        logger.info("=" * 60)
        
        validation_steps = [
            self.validate_python_version,
            self.validate_file_system,
            self.validate_core_dependencies,
            self.validate_nis_imports,
            self.validate_api_keys,
            self.validate_docker_environment,
        ]
        
        all_passed = True
        for step in validation_steps:
            try:
                if not step():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"❌ Validation step failed: {e}")
                all_passed = False
        
        return all_passed
    
    def print_results(self):
        """Print validation results."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 VALIDATION RESULTS")
        logger.info("=" * 60)
        
        # Print passed checks
        if self.passed_checks:
            logger.info(f"\n✅ PASSED CHECKS ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                logger.info(f"  {check}")
        
        # Print warnings
        if self.warnings:
            logger.info(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.info(f"  {warning}")
        
        # Print errors
        if self.errors:
            logger.error(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  {error}")
        
        logger.info("\n" + "=" * 60)
        
        if self.errors:
            logger.error("🚫 VALIDATION FAILED - Please fix the errors above before deployment")
            logger.info("\n💡 Quick fixes:")
            logger.info("  • Install missing dependencies: pip install -r requirements_enhanced_infrastructure.txt")
            logger.info("  • Configure API keys in .env file")
            logger.info("  • Ensure all required files are present")
            return False
        else:
            logger.info("🎉 VALIDATION PASSED - System ready for deployment!")
            if self.warnings:
                logger.info("⚠️  Some optional components have warnings but system can start")
            return True

def main():
    """Main validation entry point."""
    validator = EnvironmentValidator()
    
    try:
        success = validator.run_validation()
        validator.print_results()
        
        if success:
            logger.info("\n🚀 You can now start the system with: ./start.sh")
            sys.exit(0)
        else:
            logger.error("\n🛑 Please fix the issues above before starting the system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Unexpected error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 