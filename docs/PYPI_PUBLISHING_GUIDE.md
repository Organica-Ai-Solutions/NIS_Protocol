# 📦 NIS Protocol v3.0 - PyPI Publishing Guide

## 🚀 Quick Start

### **Option 1: Using Our Publishing Script (Recommended)**

```bash
# Publish to PyPI test
python scripts/publish_to_pypi.py --test

# Publish to production PyPI  
python scripts/publish_to_pypi.py --prod

# Build only (no upload)
python scripts/publish_to_pypi.py --build-only

# Clean build artifacts
python scripts/publish_to_pypi.py --clean
```

### **Option 2: Manual Publishing**

```bash
# Clean and build
rm -rf dist/ build/ *.egg-info/
python -m build

# Check package
twine check dist/*

# Upload to PyPI test
twine upload --repository testpypi dist/*

# Upload to production PyPI
twine upload dist/*
```

## 🔑 PyPI Setup Instructions

### **1. Create PyPI Test Account**
1. Go to https://test.pypi.org/account/register/
2. Create an account and verify email
3. Go to https://test.pypi.org/manage/account/token/
4. Create API token with name "NIS-Protocol-v3"
5. Copy the token (starts with `pypi-`)

### **2. Create Production PyPI Account**
1. Go to https://pypi.org/account/register/
2. Create account and verify email  
3. Go to https://pypi.org/manage/account/token/
4. Create API token
5. Copy the token

### **3. Configure Authentication**

**Method A: Edit ~/.pypirc**
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token-here
```

**Method B: Environment Variables**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-token-here
```

**Method C: Pass token directly**
```bash
twine upload --repository testpypi dist/* -u __token__ -p pypi-your-token-here
```

## 📋 Pre-Publishing Checklist

### **Version Management**
- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md` with new features
- [ ] Tag the release in git
- [ ] Update documentation

### **Quality Checks**
- [ ] All tests pass
- [ ] Code coverage > 80%
- [ ] No critical linting errors
- [ ] Documentation is up to date
- [ ] Dependencies are pinned

### **Build Verification**
- [ ] Package builds without errors
- [ ] `twine check` passes
- [ ] Install test works
- [ ] Basic import test passes

## 🧪 Testing Your Package

### **Test Installation from PyPI Test**
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install from PyPI test
pip install --index-url https://test.pypi.org/simple/ \
           --extra-index-url https://pypi.org/simple/ \
           nis-protocol

# Test basic functionality
python -c "import core.agent; print('Success!')"

# Cleanup
deactivate
rm -rf test_env
```

### **Test Installation from Production PyPI**
```bash
pip install nis-protocol
python -c "import core.agent; print('Success!')"
```

## 🔗 Package URLs

### **PyPI Test**
- **Package**: https://test.pypi.org/project/nis-protocol/
- **Installation**: 
  ```bash
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nis-protocol
  ```

### **Production PyPI**
- **Package**: https://pypi.org/project/nis-protocol/
- **Installation**:
  ```bash
  pip install nis-protocol
  ```

## 🎯 Version Strategy

### **Version Numbering**
- **Major.Minor.Patch** (e.g., 3.0.0)
- **Major**: Breaking changes
- **Minor**: New features, backward compatible  
- **Patch**: Bug fixes

### **Current Version: 3.0.0**
- **Major upgrade** from 2.x
- **New features**: LangGraph/LangSmith integration
- **Enhanced**: Multi-agent coordination
- **Added**: Physics-informed reasoning

## 🛠️ Troubleshooting

### **Common Issues**

**1. Authentication Error**
```
HTTP Error 403: Invalid or non-existent authentication information
```
**Solution**: Check your API token in ~/.pypirc

**2. Package Already Exists**
```
HTTP Error 400: File already exists
```
**Solution**: Increment version number in setup.py

**3. Build Fails**
```
No module named 'setuptools'
```
**Solution**: 
```bash
pip install --upgrade setuptools wheel build
```

**4. Missing Dependencies**
```
ModuleNotFoundError during build
```
**Solution**: Add missing dependencies to setup.py

### **Getting Help**
- **PyPI Documentation**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **Issue Tracker**: https://github.com/Organica-Ai-Solutions/NIS_Protocol/issues

## 📊 Release Process

### **1. Pre-Release**
```bash
# Update version
vim setup.py

# Run tests
python -m pytest

# Build and test
python scripts/publish_to_pypi.py --test --verify
```

### **2. Release**
```bash
# Tag release
git tag v3.0.0
git push origin v3.0.0

# Publish to production
python scripts/publish_to_pypi.py --prod --verify
```

### **3. Post-Release**
```bash
# Update documentation
# Announce on social media
# Update project README
# Plan next release
```

## 🎉 Success!

Once published, your package will be available:

**From PyPI Test:**
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nis-protocol
```

**From Production PyPI:**
```bash
pip install nis-protocol
```

**Usage:**
```python
from core.agent import NISAgent
from agents.consciousness.enhanced_conscious_agent import EnhancedConsciousAgent
from integrations.langchain_integration import NISLangChainIntegration

# Initialize the NIS Protocol v3.0 system
agent = EnhancedConsciousAgent()
integration = NISLangChainIntegration()

# Start building the future of AI!
```

---

**NIS Protocol v3.0** - Advanced Multi-Agent System with LangGraph/LangSmith Integration 🚀 