"""
A2UI Response Formatter for GenUI Integration

This module transforms plain text LLM responses into A2UI (Agent-to-UI) formatted
widget structures that can be rendered as rich interactive UI in GenUI-enabled clients.

Copyright 2025 Organica AI Solutions
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class A2UIFormatter:
    """
    Transforms plain text responses into A2UI widget structures.
    
    Detects and formats:
    - Code blocks (```language)
    - Lists (ordered and unordered)
    - Headers (# Title)
    - Bold/Italic markdown
    - Action buttons (based on keywords)
    - Tables
    - Links
    """
    
    def __init__(self):
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.list_item_pattern = re.compile(r'^[\s]*[-*]\s+(.+)$', re.MULTILINE)
        self.ordered_list_pattern = re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE)
        self.bold_pattern = re.compile(r'\*\*(.+?)\*\*')
        self.italic_pattern = re.compile(r'\*(.+?)\*')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        
        # Action keywords that trigger button creation
        self.action_keywords = [
            'run code', 'execute', 'deploy', 'start', 'stop', 'restart',
            'download', 'upload', 'save', 'delete', 'create', 'update',
            'test', 'verify', 'check', 'analyze', 'generate'
        ]
    
    def format_response(self, text: str, include_actions: bool = True) -> Dict[str, Any]:
        """
        Main entry point: Convert plain text to A2UI message structure.
        
        Args:
            text: Plain text response from LLM
            include_actions: Whether to detect and add action buttons
            
        Returns:
            A2UI message dictionary with widgets
        """
        widgets = self._parse_text_to_widgets(text)
        
        # Add action buttons if detected
        if include_actions:
            actions = self._detect_actions(text)
            if actions:
                widgets.append(self._create_action_row(actions))
        
        return {
            "role": "model",
            "widgets": widgets
        }
    
    def _parse_text_to_widgets(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text into widget list.
        
        Strategy:
        1. Extract code blocks first (they take precedence)
        2. Split remaining text into sections
        3. Parse each section for markdown elements
        """
        widgets = []
        
        # Extract code blocks and split text
        parts = self._split_by_code_blocks(text)
        
        for part in parts:
            if part['type'] == 'code':
                widgets.append(self._create_code_block_widget(
                    part['language'],
                    part['content']
                ))
            elif part['type'] == 'text':
                # Parse markdown in text sections
                text_widgets = self._parse_markdown_section(part['content'])
                widgets.extend(text_widgets)
        
        return widgets
    
    def _split_by_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into code blocks and text sections.
        
        Returns:
            List of dicts with 'type' (code/text), 'content', and optional 'language'
        """
        parts = []
        last_end = 0
        
        for match in self.code_block_pattern.finditer(text):
            # Add text before code block
            if match.start() > last_end:
                text_content = text[last_end:match.start()].strip()
                if text_content:
                    parts.append({
                        'type': 'text',
                        'content': text_content
                    })
            
            # Add code block
            language = match.group(1) or 'text'
            code_content = match.group(2).strip()
            parts.append({
                'type': 'code',
                'language': language,
                'content': code_content
            })
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            text_content = text[last_end:].strip()
            if text_content:
                parts.append({
                    'type': 'text',
                    'content': text_content
                })
        
        # If no code blocks found, return all as text
        if not parts:
            parts.append({
                'type': 'text',
                'content': text.strip()
            })
        
        return parts
    
    def _parse_markdown_section(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown elements in a text section.
        
        Handles:
        - Headers
        - Lists (unordered and ordered)
        - Paragraphs with inline formatting
        """
        widgets = []
        
        # Split by headers
        header_positions = [(m.start(), m.end(), m.group(1), m.group(2)) 
                           for m in self.header_pattern.finditer(text)]
        
        if not header_positions:
            # No headers, parse as paragraphs and lists
            widgets.extend(self._parse_paragraphs_and_lists(text))
        else:
            # Process text with headers
            last_end = 0
            for start, end, level, title in header_positions:
                # Content before header
                if start > last_end:
                    before_text = text[last_end:start].strip()
                    if before_text:
                        widgets.extend(self._parse_paragraphs_and_lists(before_text))
                
                # Add header
                widgets.append(self._create_header_widget(level, title))
                last_end = end
            
            # Remaining content after last header
            if last_end < len(text):
                after_text = text[last_end:].strip()
                if after_text:
                    widgets.extend(self._parse_paragraphs_and_lists(after_text))
        
        return widgets
    
    def _parse_paragraphs_and_lists(self, text: str) -> List[Dict[str, Any]]:
        """Parse text into paragraphs and lists."""
        widgets = []
        
        # Check for lists
        unordered_items = self.list_item_pattern.findall(text)
        ordered_items = self.ordered_list_pattern.findall(text)
        
        if unordered_items:
            # Create unordered list widget
            widgets.append(self._create_list_widget(unordered_items, ordered=False))
        elif ordered_items:
            # Create ordered list widget
            widgets.append(self._create_list_widget(ordered_items, ordered=True))
        else:
            # Regular paragraph(s)
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    widgets.append(self._create_text_widget(para))
        
        return widgets
    
    def _create_code_block_widget(self, language: str, code: str) -> Dict[str, Any]:
        """Create a NISCodeBlock widget (custom NIS catalog widget)."""
        return {
            "type": "NISCodeBlock",
            "data": {
                "language": language,
                "code": code,
                "showLineNumbers": True,
                "copyable": True
            }
        }
    
    def _create_header_widget(self, level: str, text: str) -> Dict[str, Any]:
        """Create a header/title widget."""
        # Map markdown levels to font sizes
        size_map = {
            '#': 'large',
            '##': 'medium',
            '###': 'medium',
            '####': 'small',
            '#####': 'small',
            '######': 'small'
        }
        
        return {
            "type": "Text",
            "data": {
                "text": text,
                "style": {
                    "fontSize": size_map.get(level, 'medium'),
                    "fontWeight": "bold"
                }
            }
        }
    
    def _create_text_widget(self, text: str) -> Dict[str, Any]:
        """Create a text widget with inline formatting."""
        # Process inline markdown (bold, italic, links)
        formatted_text = self._process_inline_markdown(text)
        
        return {
            "type": "Text",
            "data": {
                "text": formatted_text
            }
        }
    
    def _create_list_widget(self, items: List[str], ordered: bool = False) -> Dict[str, Any]:
        """Create a list widget."""
        # For now, render as Column with Text widgets
        # GenUI can enhance this with proper list styling
        children = []
        for i, item in enumerate(items):
            prefix = f"{i+1}. " if ordered else "• "
            children.append({
                "type": "Text",
                "data": {
                    "text": f"{prefix}{item}"
                }
            })
        
        return {
            "type": "Column",
            "data": {
                "children": children,
                "spacing": 8
            }
        }
    
    def _process_inline_markdown(self, text: str) -> str:
        """
        Process inline markdown (bold, italic, links).
        
        Note: GenUI may support rich text spans in the future.
        For now, we keep markdown syntax for basic formatting.
        """
        # Keep markdown syntax for now - GenUI can render it
        return text
    
    def _detect_actions(self, text: str) -> List[Dict[str, str]]:
        """
        Detect actionable items in text and create button specs.
        
        Returns:
            List of action dicts with 'label' and 'action' keys
        """
        actions = []
        text_lower = text.lower()
        
        # Check for common action patterns
        if 'run' in text_lower and 'code' in text_lower:
            actions.append({
                'label': 'Run Code',
                'action': 'execute_code'
            })
        
        if 'deploy' in text_lower:
            actions.append({
                'label': 'Deploy',
                'action': 'deploy'
            })
        
        if 'test' in text_lower:
            actions.append({
                'label': 'Run Tests',
                'action': 'run_tests'
            })
        
        if 'download' in text_lower or 'save' in text_lower:
            actions.append({
                'label': 'Download',
                'action': 'download'
            })
        
        return actions
    
    def _create_action_row(self, actions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a row of action buttons."""
        buttons = []
        for action in actions:
            buttons.append({
                "type": "Button",
                "data": {
                    "text": action['label'],
                    "action": action['action'],
                    "style": "primary"
                }
            })
        
        return {
            "type": "Row",
            "data": {
                "children": buttons,
                "spacing": 12,
                "mainAxisAlignment": "start"
            }
        }
    
    def wrap_in_card(self, widgets: List[Dict[str, Any]], title: Optional[str] = None) -> Dict[str, Any]:
        """
        Wrap widgets in a Card for better visual grouping.
        
        Args:
            widgets: List of widgets to wrap
            title: Optional card title
            
        Returns:
            Card widget containing the widgets
        """
        card_children = []
        
        if title:
            card_children.append({
                "type": "Text",
                "data": {
                    "text": title,
                    "style": {
                        "fontSize": "large",
                        "fontWeight": "bold"
                    }
                }
            })
        
        card_children.extend(widgets)
        
        return {
            "type": "Card",
            "data": {
                "child": {
                    "type": "Column",
                    "data": {
                        "children": card_children,
                        "spacing": 12
                    }
                }
            }
        }


def format_text_as_a2ui(text: str, wrap_in_card: bool = True, include_actions: bool = True) -> Dict[str, Any]:
    """
    Convenience function to format text as A2UI message.
    
    Args:
        text: Plain text response
        wrap_in_card: Whether to wrap widgets in a card
        include_actions: Whether to detect and add action buttons
        
    Returns:
        Complete A2UI message structure wrapped for GenUI SDK
    """
    formatter = A2UIFormatter()
    message = formatter.format_response(text, include_actions=include_actions)
    
    if wrap_in_card and message['widgets']:
        message['widgets'] = [formatter.wrap_in_card(message['widgets'])]
    
    # GenUI SDK expects messages wrapped in beginRendering/surfaceUpdate
    # For simplicity, we'll return the raw message and let the frontend handle wrapping
    return {
        "a2ui_message": {
            "widgets": message['widgets']
        }
    }


def create_simple_text_widget(text: str) -> Dict[str, Any]:
    """
    Create a minimal A2UI message with just text in a card.
    
    Use this for quick/simple responses that don't need parsing.
    """
    return {
        "a2ui_message": {
            "role": "model",
            "widgets": [
                {
                    "type": "Card",
                    "data": {
                        "child": {
                            "type": "Text",
                            "data": {
                                "text": text
                            }
                        }
                    }
                }
            ]
        }
    }


def create_error_widget(error_message: str) -> Dict[str, Any]:
    """Create an A2UI error message widget."""
    return {
        "a2ui_message": {
            "role": "model",
            "widgets": [
                {
                    "type": "Card",
                    "data": {
                        "color": "error",
                        "child": {
                            "type": "Column",
                            "data": {
                                "children": [
                                    {
                                        "type": "Text",
                                        "data": {
                                            "text": "⚠️ Error",
                                            "style": {
                                                "fontWeight": "bold",
                                                "color": "error"
                                            }
                                        }
                                    },
                                    {
                                        "type": "Text",
                                        "data": {
                                            "text": error_message
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ]
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Test with sample response
    sample_response = """
# Code Example

Here's a Python function to calculate fibonacci:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

You can run this code to test it. The function works recursively.

## Features:
- Simple implementation
- Works for small values
- Easy to understand

Let me know if you want to deploy this!
"""
    
    formatter = A2UIFormatter()
    result = formatter.format_response(sample_response)
    
    import json
    print(json.dumps(result, indent=2))
