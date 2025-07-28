#!/bin/bash
echo "🔑 PyPI Token Update Helper"
echo "=========================="
echo ""
echo "First, get your token from: https://test.pypi.org/manage/account/token/"
echo ""
read -p "Paste your PyPI test token here (starts with pypi-): " TOKEN

if [[ $TOKEN == pypi-* ]]; then
    # Update the ~/.pypirc file
    sed -i.bak "s/your-testpypi-token-here/$TOKEN/g" ~/.pypirc
    echo "✅ Token updated successfully!"
    echo ""
    echo "You can now run:"
    echo "python scripts/publish_to_pypi.py --test"
else
    echo "❌ Invalid token format. Token should start with 'pypi-'"
fi
