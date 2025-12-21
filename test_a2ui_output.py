#!/usr/bin/env python3
"""
Test A2UI Formatter Output
Verify the formatter produces correct JSON structure for GenUI
"""

import sys
import json
sys.path.insert(0, '/Users/diegofuego/Desktop/NIS_Protocol')

from src.utils.a2ui_formatter import format_text_as_a2ui, create_simple_text_widget, create_error_widget

print("=" * 80)
print("A2UI FORMATTER OUTPUT TESTS")
print("=" * 80)

# Test 1: Simple text
print("\n1. SIMPLE TEXT:")
print("-" * 80)
simple = create_simple_text_widget("Hello, this is a simple message.")
print(json.dumps(simple, indent=2))

# Test 2: Code block
print("\n2. CODE BLOCK:")
print("-" * 80)
code_test = """Here's a Python function:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

You can run this code to test it."""

code_result = format_text_as_a2ui(code_test, wrap_in_card=True, include_actions=True)
print(json.dumps(code_result, indent=2))

# Test 3: Lists
print("\n3. LISTS:")
print("-" * 80)
list_test = """Here are the key features:

- Fast performance
- Easy to use
- Well documented
- Open source"""

list_result = format_text_as_a2ui(list_test, wrap_in_card=True, include_actions=False)
print(json.dumps(list_result, indent=2))

# Test 4: Headers
print("\n4. HEADERS:")
print("-" * 80)
header_test = """# Main Title

This is some content.

## Subsection

More content here."""

header_result = format_text_as_a2ui(header_test, wrap_in_card=True, include_actions=False)
print(json.dumps(header_result, indent=2))

# Test 5: Mixed content
print("\n5. MIXED CONTENT (Real-world example):")
print("-" * 80)
mixed_test = """# Code Example

Here's how to deploy a server:

```bash
docker-compose up -d
```

## Features:
- Automatic scaling
- Load balancing
- Health checks

You can deploy this now!"""

mixed_result = format_text_as_a2ui(mixed_test, wrap_in_card=True, include_actions=True)
print(json.dumps(mixed_result, indent=2))

# Test 6: Error widget
print("\n6. ERROR WIDGET:")
print("-" * 80)
error = create_error_widget("Something went wrong!")
print(json.dumps(error, indent=2))

print("\n" + "=" * 80)
print("SCHEMA VERIFICATION")
print("=" * 80)
print("\nExpected GenUI Schema:")
print("""
{
  "a2ui_message": {
    "role": "model",
    "widgets": [
      {
        "type": "WidgetType",
        "data": { ... }
      }
    ]
  }
}
""")

print("\nWidget Types Generated:")
print("- Card")
print("- Text")
print("- CodeBlock")
print("- Button")
print("- Row")
print("- Column")

print("\n" + "=" * 80)
print("COMPATIBILITY CHECK")
print("=" * 80)

# Check structure
test_output = format_text_as_a2ui("Test", wrap_in_card=True)
checks = {
    "Has 'a2ui_message' key": "a2ui_message" in test_output,
    "Has 'role' field": "role" in test_output.get("a2ui_message", {}),
    "Has 'widgets' array": "widgets" in test_output.get("a2ui_message", {}),
    "Widgets is list": isinstance(test_output.get("a2ui_message", {}).get("widgets", None), list),
    "Widget has 'type'": len(test_output.get("a2ui_message", {}).get("widgets", [])) > 0 and "type" in test_output["a2ui_message"]["widgets"][0],
    "Widget has 'data'": len(test_output.get("a2ui_message", {}).get("widgets", [])) > 0 and "data" in test_output["a2ui_message"]["widgets"][0],
}

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")

print("\n" + "=" * 80)
print("READY FOR DEPLOYMENT" if all(checks.values()) else "NEEDS FIXES")
print("=" * 80)
