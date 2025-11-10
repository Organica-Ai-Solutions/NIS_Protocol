#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

dart --version >/dev/null 2>&1 || {
  echo "Error: Dart SDK is not available in PATH." >&2
  exit 1
}

echo "Starting Dart MCP server (force roots fallback)..."
echo "Workspace root: $PROJECT_ROOT"

dart mcp-server --force-roots-fallback
