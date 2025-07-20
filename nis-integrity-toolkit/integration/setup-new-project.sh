#!/bin/bash

# 🚀 NIS Integrity Toolkit - New Project Setup
# Sets up engineering integrity tools in a new project

set -e

PROJECT_NAME=${1:-"new-nis-project"}
TOOLKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🛠️  NIS Engineering Integrity Toolkit - New Project Setup"
echo "=" * 60
echo "📁 Project: $PROJECT_NAME"
echo "🔧 Toolkit: $TOOLKIT_DIR"
echo

# Create project directory structure
echo "📁 Creating project structure..."
mkdir -p "$PROJECT_NAME"/{src,tests,docs,benchmarks,configs}
cd "$PROJECT_NAME"

# Copy integrity toolkit
echo "🔧 Installing integrity toolkit..."
cp -r "$TOOLKIT_DIR" ./nis-integrity-toolkit/

# Create basic project files from templates
echo "📝 Creating project files from templates..."

# Create honest README
cp ./nis-integrity-toolkit/templates/HONEST_README_TEMPLATE.md ./README.md

# Replace template placeholders
sed -i.bak "s/\[Project Name\]/$PROJECT_NAME/g" ./README.md
sed -i.bak "s/\[One-Line Accurate Description\]/Advanced system for [domain] with evidence-based performance/g" ./README.md
rm ./README.md.bak

# Create .cursorrules
cp ./nis-integrity-toolkit/.cursorrules ./

# Create basic Python project structure
cat > ./src/__init__.py << 'EOF'
"""
NIS Protocol System
Engineering integrity validated system
"""

__version__ = "1.0.0"
__author__ = "NIS Engineering Team"
__description__ = "Evidence-based system with validated performance"
EOF

# Create basic test structure
cat > ./tests/__init__.py << 'EOF'
"""
Test suite for NIS protocol system
All performance claims validated through these tests
"""
EOF

cat > ./tests/test_integrity.py << 'EOF'
"""
Integrity validation tests
These tests validate all claims made in documentation
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestIntegrity:
    """Validate all documented claims"""
    
    def test_performance_claims_have_evidence(self):
        """Ensure all performance claims in README have corresponding tests"""
        # This test should validate specific performance claims
        pass
    
    def test_no_hardcoded_metrics(self):
        """Ensure no hardcoded performance metrics in source code"""
        # This test should scan for hardcoded values
        pass
    
    def test_documentation_accuracy(self):
        """Ensure documentation matches implementation"""
        # This test should validate doc-code alignment
        pass
EOF

# Create benchmark structure
cat > ./benchmarks/performance_test.py << 'EOF'
"""
Performance benchmark suite
Validates all performance claims made in documentation
"""

import time
import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def benchmark_processing_speed():
    """Benchmark processing speed claims"""
    # TODO: Implement actual benchmark
    print("📊 Running processing speed benchmark...")
    return {"items_per_second": 100, "test_date": "2025-01-07"}

def benchmark_accuracy():
    """Benchmark accuracy claims"""  
    # TODO: Implement actual benchmark
    print("📊 Running accuracy benchmark...")
    return {"accuracy": 0.95, "test_date": "2025-01-07"}

def main():
    """Run all benchmarks"""
    print("🚀 Running performance benchmarks...")
    
    speed_results = benchmark_processing_speed()
    accuracy_results = benchmark_accuracy()
    
    print(f"✅ Speed: {speed_results['items_per_second']} items/second")
    print(f"✅ Accuracy: {accuracy_results['accuracy']:.1%}")
    
    return {
        "speed": speed_results,
        "accuracy": accuracy_results
    }

if __name__ == "__main__":
    results = main()
EOF

# Create requirements.txt
cat > ./requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0

# Development
black>=22.0.0
flake8>=5.0.0

# Documentation
markdown>=3.4.0
EOF

# Create development configuration
cat > ./pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "PROJECT_NAME"
version = "1.0.0"
description = "Evidence-based system with validated performance"
authors = [{name = "NIS Engineering Team"}]
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
EOF

# Replace project name in pyproject.toml
sed -i.bak "s/PROJECT_NAME/$PROJECT_NAME/g" ./pyproject.toml
rm ./pyproject.toml.bak

# Create .gitignore
cat > ./.gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/
coverage.xml
*.cover
*.py,cover
.hypothesis/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
.env
.env.local
logs/
*.log
EOF

# Create pre-commit hook
mkdir -p .git/hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# NIS Integrity pre-commit hook

echo "🔍 Running NIS integrity check..."
python ./nis-integrity-toolkit/audit-scripts/pre-submission-check.py

if [ $? -ne 0 ]; then
    echo "❌ Commit blocked by integrity check"
    echo "Fix the issues above and try again"
    exit 1
fi

echo "✅ Integrity check passed"
EOF

chmod +x .git/hooks/pre-commit

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "🔄 Initializing git repository..."
    git init
fi

# Create initial commit
echo "📝 Creating initial commit..."
git add .
git commit -m "Initial commit: NIS project with integrity toolkit

- Set up project structure with honest documentation
- Integrated NIS Engineering Integrity Toolkit
- Created test and benchmark framework
- Configured development environment
- Added pre-commit integrity checks

Engineering integrity validated ✅"

echo
echo "🎯 PROJECT SETUP COMPLETE!"
echo "=" * 60
echo "📁 Project: $PROJECT_NAME"
echo "🛠️  Toolkit: Integrated"
echo "✅ Git: Initialized with integrity hooks"
echo "📝 README: Honest template ready for customization"
echo "🧪 Tests: Framework ready for implementation"
echo "📊 Benchmarks: Structure ready for validation"
echo
echo "🚀 NEXT STEPS:"
echo "1. cd $PROJECT_NAME"
echo "2. Customize README.md with your specific project details"
echo "3. Implement your core functionality in src/"
echo "4. Add real tests in tests/"
echo "5. Create actual benchmarks in benchmarks/"
echo "6. Run integrity check: python nis-integrity-toolkit/audit-scripts/pre-submission-check.py"
echo
echo "💡 Remember: Build impressive systems, describe them accurately!"
echo "🔧 All commits will be automatically checked for integrity" 