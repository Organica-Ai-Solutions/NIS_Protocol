# Dependency Security Analysis

**Date**: December 27, 2025  
**Status**: 🔴 10 vulnerabilities detected by GitHub (5 high, 5 moderate)

---

## 🔍 **Current Issues**

### **Critical Conflicts**

1. **aiobotocore/botocore version mismatch**
   - aiobotocore 2.7.0 requires botocore <1.31.65
   - Currently have botocore 1.40.25
   - **Fix**: Upgrade aiobotocore to 3.0.0+

2. **opencv-python/numpy incompatibility**
   - opencv-python 4.12.0.88 requires numpy >=2.0
   - Currently have numpy 1.26.4
   - **Risk**: May break PyTorch/TensorFlow compatibility
   - **Fix**: Test numpy 2.x compatibility or pin opencv-python

3. **Multiple pydantic version conflicts**
   - fastapi-mail, alpaca-py require pydantic <2.0
   - Currently have pydantic 2.11.7
   - **Fix**: Remove incompatible packages or find alternatives

4. **cryptography version conflict**
   - presidio-anonymizer requires cryptography <44.1
   - Currently have cryptography 45.0.6
   - **Fix**: Downgrade cryptography or update presidio-anonymizer

5. **aiohttp version conflict**
   - alpaca-trade-api requires aiohttp==3.8.2
   - Currently have aiohttp 3.12.15
   - **Risk**: Security vulnerability in 3.8.2
   - **Fix**: Remove alpaca-trade-api or update it

---

## 📊 **Outdated Packages (Security-Relevant)**

| Package | Current | Latest | Priority |
|---------|---------|--------|----------|
| accelerate | 1.5.2 | 1.12.0 | Medium |
| aiobotocore | 2.7.0 | 3.0.0 | High |
| aiohttp | 3.12.15 | 3.13.2 | High |
| aiosmtplib | 2.0.2 | 5.0.0 | Low |
| urllib3 | 2.5.0 | 2.2.3 | High (security) |

---

## 🎯 **Recommended Actions**

### **Phase 1: Remove Unused/Conflicting Packages**

These packages are causing conflicts and may not be actively used:

```bash
# Remove if not needed
pip uninstall -y alpaca-trade-api alpaca-py
pip uninstall -y fastapi-mail
pip uninstall -y presidio-anonymizer
pip uninstall -y conda-repo-cli
pip uninstall -y s3fs
```

### **Phase 2: Update Core Security Packages**

```bash
# Update critical security packages
pip install --upgrade aiobotocore==3.0.0
pip install --upgrade botocore>=1.40.0
pip install --upgrade aiohttp>=3.13.0
pip install --upgrade accelerate>=1.12.0
```

### **Phase 3: Test Numpy 2.x Compatibility**

```bash
# Test in isolated environment first
pip install numpy>=2.0.0
# Run tests to verify PyTorch/scipy compatibility
```

### **Phase 4: Update requirements.txt**

Update pinned versions to latest secure releases.

---

## 🔒 **GitHub Dependabot Alerts**

**Likely vulnerabilities** (based on outdated packages):

1. **urllib3** - Multiple CVEs in older versions
2. **aiohttp** - DoS vulnerabilities in <3.13.0
3. **cryptography** - Potential security issues in older versions
4. **requests** - Auth bypass in older versions
5. **pillow** - Image processing vulnerabilities

**Action**: Review GitHub Security tab for exact CVEs and affected versions.

---

## ✅ **Safe to Update Immediately**

These have no known conflicts:

- accelerate: 1.5.2 → 1.12.0
- aiodns: 3.5.0 → 3.6.1
- aioice: 0.10.1 → 0.10.2
- aiortc: 1.13.0 → 1.14.0
- aioitertools: 0.7.1 → 0.13.0

---

## ⚠️ **Requires Testing**

These may break compatibility:

- numpy: 1.26.4 → 2.x (test PyTorch/scipy first)
- aiobotocore: 2.7.0 → 3.0.0 (requires botocore update)
- aiosmtplib: 2.0.2 → 5.0.0 (major version jump)

---

## 📝 **Implementation Plan**

1. **Backup current environment**
   ```bash
   pip freeze > requirements_backup.txt
   ```

2. **Remove conflicting packages**
   - alpaca-trade-api (requires old aiohttp)
   - fastapi-mail (requires pydantic <2.0)
   - presidio-anonymizer (requires old cryptography)

3. **Update security-critical packages**
   - aiobotocore, botocore
   - aiohttp
   - accelerate

4. **Test application**
   - Run health check
   - Test critical endpoints
   - Verify no import errors

5. **Update requirements.txt**
   - Pin to tested versions
   - Document changes

6. **Commit and push**
   - Create branch `fix/dependency-security`
   - Test in CI/CD
   - Merge to main

---

## 🚀 **Next Command**

```bash
# Start with removing unused packages
pip uninstall -y alpaca-trade-api alpaca-py fastapi-mail presidio-anonymizer conda-repo-cli s3fs
```
