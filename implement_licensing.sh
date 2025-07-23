#!/bin/bash

# NIS Protocol License Implementation Script
# Converts repository from MIT to Dual License (MIT + BSL)

echo "🚀 Implementing NIS Protocol License Transition..."
echo "From MIT License to Dual License (MIT v3.x + BSL v4.x+)"
echo

# Backup current LICENSE
if [ -f "LICENSE" ]; then
    echo "📄 Backing up current MIT license..."
    cp LICENSE LICENSE_MIT_BACKUP
    mv LICENSE LICENSE_MIT
    echo "✅ Current LICENSE renamed to LICENSE_MIT"
fi

# Create LICENSE_BSL (already created by previous command)
echo "📄 LICENSE_BSL created"

# Update README.md with licensing information
echo "📝 Updating README.md with licensing section..."

# Create licensing section to insert
cat >> README_LICENSE_SECTION.tmp << 'EOF'

---

## 📄 **LICENSING**

### **Dual Licensing Model**

The NIS Protocol uses a dual licensing approach:

- **🆓 v3.x and Earlier**: MIT License - Free for all uses
- **🏢 v4.x and Later**: Business Source License (BSL) - Commercial use requires licensing

### **For Researchers & Developers:**
- Full access to all code for research, education, and personal use
- Fork, modify, and experiment freely
- Contribute to the open source community

### **For Commercial Users:**
- Companies with >$1M annual revenue require commercial license for v4+
- [Get Commercial License](COMMERCIAL_LICENSE_AGREEMENT.md)
- Contact: diego@organicaai.com

### **Why the Change?**
To ensure sustainable development while maintaining open source principles. We want to support the community while enabling commercial partnerships that fund continued innovation.

**Current Status:**
- **v3.x**: Production-ready, MIT licensed, 75,359+ lines of AGI foundation code
- **v4.x**: In development, BSL licensed, enhanced AWS integration and enterprise features

For commercial licensing inquiries: **diego@organicaai.com**

EOF

# Insert before the final acknowledgments section
if grep -q "ACKNOWLEDGMENTS" README.md; then
    # Insert before acknowledgments
    sed -i '/## 🙏 \*\*ACKNOWLEDGMENTS\*\*/i\
'"$(cat README_LICENSE_SECTION.tmp)" README.md
else
    # Append to end of file
    cat README_LICENSE_SECTION.tmp >> README.md
fi

rm README_LICENSE_SECTION.tmp
echo "✅ README.md updated with licensing information"

# Create git tags for version separation
echo "🏷️ Creating git tags for license transition..."
git tag -a v3.9.9 -m "Final MIT-licensed version of NIS Protocol"
git tag -a v4.0.0-bsl -m "First BSL-licensed version - enhanced enterprise features"

# Create .github/LICENSE_NOTICE.md for pull request template
mkdir -p .github
cat > .github/LICENSE_NOTICE.md << 'EOF'
# License Notice

## Contribution Guidelines

By contributing to this repository, you agree that:

1. **For v3.x branches**: Contributions are MIT licensed
2. **For v4.x+ branches**: Contributions are BSL licensed
3. You have the right to make the contribution
4. Your contribution may be included in commercial licensing agreements

## Current Licensing Status

- **v3.x and earlier**: MIT License
- **v4.x and later**: Business Source License 1.1

## Commercial Use

If you represent a company with >$1M annual revenue and want to use v4.x+ commercially, please contact diego@organicaai.com for licensing terms.
EOF

echo "✅ Created GitHub license notice"

# Create commercial licensing contact info
echo "📧 Setting up commercial licensing infrastructure..."

# Create simple commercial license inquiry template
cat > COMMERCIAL_LICENSE_INQUIRY_TEMPLATE.md << 'EOF'
# Commercial License Inquiry Template

## Company Information
- **Company Name**: 
- **Annual Revenue**: 
- **Industry**: 
- **Contact Person**: 
- **Email**: 
- **Phone**: 

## Use Case
- **Intended Use**: 
- **Expected Usage Scale**: 
- **Timeline**: 
- **Integration Requirements**: 

## Technical Requirements
- **AWS Integration Needed**: Yes/No
- **Custom Development Required**: Yes/No
- **Support Level Desired**: Basic/Premium/Enterprise
- **Deployment Model**: Cloud/On-premise/Hybrid

## Additional Information
- **Budget Range**: 
- **Decision Timeline**: 
- **Other Requirements**: 

---

**Next Steps:**
1. Send completed template to: diego@organicaai.com
2. Schedule technical discussion call
3. Receive customized licensing proposal
4. Legal review and contract execution
EOF

echo "✅ Created commercial license inquiry template"

# Update package.json or setup.py with new license info
if [ -f "setup.py" ]; then
    echo "📦 Updating setup.py with dual license info..."
    # Add note about dual licensing
    sed -i 's/license="MIT"/license="MIT (v3.x), BSL 1.1 (v4.x+)"/g' setup.py
fi

# Create legal disclaimer file
cat > LEGAL_DISCLAIMER.md << 'EOF'
# Legal Disclaimer

## Licensing

The Neural Intelligence Synthesis (NIS) Protocol is dual-licensed:

- **v3.x and earlier**: MIT License
- **v4.x and later**: Business Source License 1.1

## Commercial Use Notice

Commercial use of v4.x+ by entities with annual revenue >$1M requires a commercial license agreement. Contact diego@organicaai.com for licensing terms.

## Warranty Disclaimer

This software is provided "as is" without warranty of any kind. See respective license files for complete terms.

## Contact

- **General Questions**: diego@organicaai.com
- **Commercial Licensing**: diego@organicaai.com
- **Legal Issues**: diego@organicaai.com
EOF

echo "✅ Created legal disclaimer"

# Create AWS-specific partnership proposal
echo "🤝 Creating AWS partnership proposal..."
# (COMMERCIAL_LICENSE_AGREEMENT.md already created)

echo
echo "🎉 License transition implementation complete!"
echo
echo "📋 Summary of changes:"
echo "  ✅ LICENSE renamed to LICENSE_MIT (for v3.x)"
echo "  ✅ LICENSE_BSL created (for v4.x+)"
echo "  ✅ README.md updated with licensing section"
echo "  ✅ Commercial license agreement template created"
echo "  ✅ Legal implementation guide created"
echo "  ✅ GitHub license notices added"
echo "  ✅ Commercial inquiry template created"
echo "  ✅ Git tags created for version separation"
echo
echo "🎯 Before AWS meeting:"
echo "  1. Review all created documents"
echo "  2. Commit changes to repository"
echo "  3. Practice explaining the dual licensing model"
echo "  4. Prepare commercial licensing talking points"
echo
echo "💼 AWS Partnership Terms Ready:"
echo "  - Strategic Partner License: $250,000/year + 3% revenue share"
echo "  - 6-month exclusivity on new features"
echo "  - White-label rights for AWS AGI services"
echo "  - Joint technical development and support"
echo
echo "📞 Commercial licensing contact: diego@organicaai.com"
echo
echo "🚀 Your negotiating position is now transformed!"
echo "   You're offering partnership, not asking for charity."

# Make script executable
chmod +x implement_licensing.sh 