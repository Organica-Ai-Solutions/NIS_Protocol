# 📁 NIS Protocol File Organization Rules

## 🎯 **Core Principle**

**Keep the root directory CLEAN and PROFESSIONAL**. Only essential project files belong in the root. All utilities, tests, summaries, and temporary files must be organized in appropriate subdirectories.

## 📋 **Root Directory - ALLOWED Files**

### ✅ **ESSENTIAL FILES ONLY**
```
Root Directory (/) - ALLOWED:
├── main.py                           # Main application entry point
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # Docker orchestration
├── Dockerfile                        # Container build instructions
├── README.md                         # Main project documentation
├── LICENSE                           # Project license
├── .gitignore                        # Git ignore patterns
├── .dockerignore                     # Docker ignore patterns
├── .gitattributes                    # Git attributes configuration
├── .cursorrules                      # Cursor IDE configuration
├── .nojekyll                         # GitHub Pages configuration
├── CNAME                             # GitHub Pages domain
├── start.sh                          # System startup script
├── stop.sh                           # System shutdown script  
├── reset.sh                          # System reset script
└── NIS_Protocol_v3_COMPLETE_Postman_Collection.json  # API testing
```

### ❌ **PROHIBITED IN ROOT**
- Test files (`test_*.py`, `test_*.sh`)
- Utility scripts (`*_utility.py`, `*_helper.py`)
- Report files (`*_report.json`, `*_summary.md`)
- Temporary files (`*.tmp`, `*~`, `commit_message.txt`)
- Development tools (`fix_*.py`, `generate_*.py`)
- Log files (`*.log`, `server.log`)
- Backup files (`*.backup`, `*_backup.*`)

## 🗂️ **Proper File Locations**

### **🧪 Testing Files**
```
dev/testing/
├── root_cleanup/           # Tests moved from root during cleanup
├── integration/            # Integration test suites
├── benchmarks/            # Performance benchmarking
└── validation/            # System validation tests

Location Rules:
✅ test_*.py → dev/testing/root_cleanup/
✅ *_test.py → dev/testing/
✅ test_*.sh → dev/testing/root_cleanup/
✅ benchmark_*.py → dev/testing/benchmarks/
```

### **🔧 Development Utilities**
```
dev/utilities/
├── nvidia_*.py            # NVIDIA-related utilities
├── fix_*.py              # System fixing utilities
├── generate_*.py         # Report generation utilities
└── save_*.py             # Backup and save utilities

Location Rules:
✅ nvidia_cli_connector.py → dev/utilities/
✅ fix_terminal.py → dev/utilities/
✅ generate_integrity_report.py → dev/utilities/
✅ save_endpoint_fixes.py → dev/utilities/
```

### **📜 System Scripts**
```
scripts/
├── installation/          # Installation and setup scripts
├── deployment/           # Deployment automation scripts
├── maintenance/          # System maintenance scripts
└── utilities/            # General utility scripts

Location Rules:
✅ install_*.py → scripts/installation/
✅ deploy_*.sh → scripts/deployment/
✅ kill_process.sh → scripts/maintenance/
✅ check_port.sh → scripts/utilities/
```

### **📊 Reports & Summaries**
```
system/utilities/reports/
├── audit-report.json     # System audit reports
├── performance_*.json    # Performance analysis reports
├── integration_*.json    # Integration test reports
└── summary_*.md          # Summary documents

Location Rules:
✅ *_report.json → system/utilities/reports/
✅ audit-*.json → system/utilities/reports/
✅ *_summary.md → system/utilities/reports/
✅ finetuning_report.json → system/utilities/reports/
```

### **📝 Documentation**
```
system/docs/              # Main documentation
dev/documentation/        # Development documentation
private/documentation/    # Private/confidential documentation

Location Rules:
✅ API_*.md → system/docs/
✅ ARCHITECTURE.md → system/docs/
✅ NVIDIA_*.md → dev/documentation/
✅ NEMOTRON_*.md → dev/documentation/
✅ AWS_*.md → private/aws_migration/
```

### **🗄️ Data & Logs**
```
logs/
├── archived/             # Archived log files
├── application/          # Application logs
└── system/              # System logs

src/data/                # Data processing scripts
cache/                   # Temporary cache files
data/                    # Data storage

Location Rules:
✅ *.log → logs/application/
✅ server.log → logs/archived/
✅ era5_*.py → src/data/
```

### **⚙️ Configuration**
```
config/                  # System configuration
dev/                    # Development configuration
private/                # Private configuration

Location Rules:
✅ environment-template.txt → dev/
✅ *_config.json → config/
✅ requirements-*.txt → appropriate subdirectory
```

## 🚨 **Automated Enforcement Rules**

### **🔍 Pre-Commit Checks**
```bash
# Automatic checks before any commit
1. Root directory scan for prohibited files
2. Verify all test files are in dev/testing/
3. Ensure all utilities are properly categorized
4. Check for temporary files in root
5. Validate documentation organization
```

### **📋 File Creation Rules**
```yaml
# When creating new files, use these patterns:

Test Files:
  Pattern: test_*.py, *_test.py
  Location: dev/testing/
  Action: Never create in root

Utility Scripts:
  Pattern: *_utility.py, fix_*.py, generate_*.py
  Location: dev/utilities/ or system/utilities/
  Action: Never create in root

Reports:
  Pattern: *_report.json, *_summary.md
  Location: system/utilities/reports/
  Action: Never create in root

Documentation:
  Pattern: *.md (except README.md)
  Location: system/docs/ or dev/documentation/
  Action: Never create in root

Temporary Files:
  Pattern: *.tmp, *~, temp_*
  Location: /tmp or cache/
  Action: Auto-delete after use
```

### **🔄 Cleanup Automation**
```bash
# Automated cleanup scripts (run weekly)
1. Move misplaced files to correct locations
2. Delete temporary files older than 7 days
3. Archive old logs to logs/archived/
4. Update file organization documentation
5. Generate file organization compliance report
```

## 📐 **File Naming Conventions**

### **✅ Good File Names**
```
# Testing
test_agent_integration.py
benchmark_laplace_performance.py
validate_physics_compliance.py

# Utilities  
nvidia_model_validator.py
fix_dependency_conflicts.py
generate_audit_report.py

# Reports
physics_validation_report.json
system_performance_summary.md
integration_test_results.json

# Documentation
AGENT_DEVELOPMENT_GUIDE.md
API_REFERENCE_V3.md
TROUBLESHOOTING_NVIDIA.md
```

### **❌ Bad File Names**
```
# Too vague
test.py
utility.py
report.json

# Wrong location indicators
root_test.py          # Don't put in root!
temp_fix.py          # Use proper temp directory
quick_script.py      # Categorize properly

# Inconsistent naming
TestFile.py          # Use snake_case
test-file.py         # Use underscores
testFile.py          # Use snake_case
```

## 🛠️ **Implementation Commands**

### **📁 Directory Creation**
```bash
# Create proper directory structure
mkdir -p dev/utilities
mkdir -p dev/testing/root_cleanup
mkdir -p dev/testing/benchmarks
mkdir -p dev/testing/integration
mkdir -p system/utilities/reports
mkdir -p scripts/installation
mkdir -p scripts/deployment
mkdir -p scripts/maintenance
mkdir -p logs/archived
mkdir -p logs/application
```

### **🔄 File Migration**
```bash
# Move files to proper locations
mv test_*.py dev/testing/root_cleanup/
mv *_utility.py dev/utilities/
mv *_report.json system/utilities/reports/
mv *.log logs/archived/
mv install_*.py scripts/installation/
```

### **🧹 Cleanup Commands**
```bash
# Remove prohibited files from root
rm -f *.tmp *~ temp_* commit_message.txt

# Clean up empty directories
find . -type d -empty -delete

# Verify root directory compliance
ls -la | grep -E "\.(py|json|log)$" | grep -v -E "(main\.py|requirements\.txt)"
```

## 📊 **Compliance Monitoring**

### **✅ Daily Checks**
1. **Root Directory Scan**: Ensure only allowed files remain
2. **File Location Validation**: Check files are in correct directories
3. **Naming Convention Check**: Verify file names follow standards
4. **Documentation Updates**: Keep organization rules current

### **📈 Weekly Reports**
1. **File Organization Compliance Score**
2. **Misplaced Files Report**
3. **Cleanup Actions Taken**
4. **Directory Growth Analysis**

### **🎯 Compliance Targets**
- **Root Directory Compliance**: 100% (zero prohibited files)
- **Proper File Categorization**: 95%+ files in correct locations
- **Naming Convention Adherence**: 90%+ files follow standards
- **Documentation Currency**: Updated within 7 days of changes

## 🚀 **Benefits of This Organization**

### **👩‍💻 For Developers**
- **Clean Workspace**: Easy to navigate and understand
- **Fast File Location**: Predictable file organization
- **Reduced Conflicts**: Clear ownership and responsibility
- **Better Git History**: Logical file grouping in commits

### **🏢 For Enterprise**
- **Professional Appearance**: Clean, organized project structure
- **Audit Compliance**: Clear documentation and file tracking
- **Maintenance Efficiency**: Easy to maintain and update
- **Onboarding Speed**: New developers can navigate quickly

### **🔧 For System Operations**
- **Automated Management**: Scripts can reliably find files
- **Backup Efficiency**: Organized backup and recovery
- **Monitoring Clarity**: Clear separation of system components
- **Deployment Reliability**: Predictable file locations

## 📞 **Enforcement & Support**

### **🔍 How to Check Compliance**
```bash
# Run compliance check
python dev/utilities/check_file_organization.py

# Generate compliance report
python system/utilities/generate_organization_report.py

# Fix common violations
python dev/utilities/fix_file_organization.py
```

### **🆘 If You Need to Create Files**
1. **Check this guide first** - Find the correct location
2. **Use proper naming** - Follow naming conventions
3. **Never put in root** - Unless explicitly allowed
4. **Update documentation** - If creating new categories
5. **Test organization** - Run compliance checks

### **📝 Rule Updates**
This document should be updated whenever:
- New file categories are introduced
- Directory structure changes
- Naming conventions evolve
- Compliance requirements change

---

**Remember**: A clean, organized codebase is a maintainable codebase. Every file has its place! 🎯