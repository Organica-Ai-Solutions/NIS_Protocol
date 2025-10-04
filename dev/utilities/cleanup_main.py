#!/usr/bin/env python3
"""
Main.py Cleanup Script
Removes test/demo/broken endpoints from main.py
"""

import re

# Endpoints to remove (line numbers from grep)
ENDPOINTS_TO_REMOVE = [
    ("communication/stream/demo", 1745),
    ("llm/consensus/demo", 2425),
    ("test-debug", 4376),
    ("test/formatter", 4550),
    ("test/minimal-chat", 5472),
    ("api/mcp/demo", 7071),
    ("nvidia/nemo/toolkit/test", 7374),
    ("nvidia/nemo/cosmos/demo", 7408),
    ("voice-chat", 7541),  # BROKEN - calls non-existent functions
    ("test-audio", 7907),
    ("test-audio-chunked", 7937),
]

def find_endpoint_ranges(filepath):
    """Find line ranges for each endpoint to remove"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    ranges_to_remove = []
    
    for name, start_line in ENDPOINTS_TO_REMOVE:
        # Find the end of this endpoint (next @app decorator or end of file)
        end_line = len(lines)
        for i in range(start_line, len(lines)):
            # Check if we hit the next endpoint
            if i > start_line and lines[i].strip().startswith('@app.'):
                end_line = i
                break
        
        print(f"  - /{name}: lines {start_line}-{end_line} ({end_line - start_line + 1} lines)")
        ranges_to_remove.append((start_line, end_line, name))
    
    return ranges_to_remove

def remove_endpoints(filepath, output_path):
    """Remove endpoints and create cleaned file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    ranges = find_endpoint_ranges(filepath)
    
    # Sort ranges in reverse order so we can remove from bottom to top
    ranges.sort(reverse=True)
    
    total_removed = 0
    for start, end, name in ranges:
        # Convert to 0-indexed
        start_idx = start - 1
        end_idx = end - 1
        
        # Remove the lines
        removed_count = end_idx - start_idx
        del lines[start_idx:end_idx]
        total_removed += removed_count
        print(f"✅ Removed /{name}: {removed_count} lines")
    
    # Write cleaned file
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    print(f"\n🎉 Total removed: {total_removed} lines")
    print(f"📝 Original: {len(lines) + total_removed} lines")
    print(f"📝 New: {len(lines)} lines")
    print(f"📊 Reduction: {total_removed / (len(lines) + total_removed) * 100:.1f}%")

if __name__ == "__main__":
    print("🧹 Main.py Cleanup Script")
    print("=" * 50)
    print("\n📋 Endpoints to remove:")
    
    ranges = find_endpoint_ranges("main.py")
    
    print("\n⚠️  Ready to clean main.py")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        remove_endpoints("main.py", "main.py")
        print("\n✅ main.py cleaned successfully!")
    else:
        print("\n❌ Cleanup cancelled")

