#!/usr/bin/env python3
"""
File Operations Tool for NIS Protocol
Enables autonomous agents to read and write files safely

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class FileOperationsTool:
    """
    Safe file operations for autonomous agents.
    
    Provides:
    - file_read: Read file contents
    - file_write: Write/create files
    - file_list: List directory contents
    - file_exists: Check if file exists
    
    Security:
    - Sandboxed to workspace directory
    - No access to system files
    - Size limits enforced
    """
    
    def __init__(self, workspace_dir: str = "/tmp/nis_workspace"):
        """Initialize file operations tool."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Security limits
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_files_per_dir = 1000
        
        logger.info(f"📁 File operations initialized: {self.workspace_dir}")
    
    def _validate_path(self, file_path: str) -> Path:
        """
        Validate and resolve file path within workspace.
        
        Args:
            file_path: Relative or absolute path
            
        Returns:
            Resolved absolute path
            
        Raises:
            ValueError: If path is outside workspace
        """
        # Convert to Path object
        path = Path(file_path)
        
        # If relative, make it relative to workspace
        if not path.is_absolute():
            path = self.workspace_dir / path
        
        # Resolve to absolute path
        path = path.resolve()
        
        # Security check: must be within workspace
        try:
            path.relative_to(self.workspace_dir)
        except ValueError:
            raise ValueError(f"Path outside workspace: {file_path}")
        
        return path
    
    def file_read(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Read file contents.
        
        Args:
            file_path: Path to file (relative to workspace)
            encoding: File encoding (default: utf-8)
            
        Returns:
            Dict with success status and content or error
        """
        try:
            path = self._validate_path(file_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            if not path.is_file():
                return {
                    "success": False,
                    "error": f"Not a file: {file_path}"
                }
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return {
                    "success": False,
                    "error": f"File too large: {file_size} bytes (max: {self.max_file_size})"
                }
            
            # Read file
            content = path.read_text(encoding=encoding)
            
            logger.info(f"✅ Read file: {file_path} ({file_size} bytes)")
            
            return {
                "success": True,
                "content": content,
                "file_path": str(path.relative_to(self.workspace_dir)),
                "size": file_size,
                "encoding": encoding
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def file_write(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to file.
        
        Args:
            file_path: Path to file (relative to workspace)
            content: Content to write
            encoding: File encoding (default: utf-8)
            create_dirs: Create parent directories if needed
            
        Returns:
            Dict with success status and details or error
        """
        try:
            path = self._validate_path(file_path)
            
            # Check content size
            content_size = len(content.encode(encoding))
            if content_size > self.max_file_size:
                return {
                    "success": False,
                    "error": f"Content too large: {content_size} bytes (max: {self.max_file_size})"
                }
            
            # Create parent directories if needed
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            path.write_text(content, encoding=encoding)
            
            logger.info(f"✅ Wrote file: {file_path} ({content_size} bytes)")
            
            return {
                "success": True,
                "file_path": str(path.relative_to(self.workspace_dir)),
                "size": content_size,
                "encoding": encoding,
                "created": not path.exists()
            }
            
        except Exception as e:
            logger.error(f"❌ Error writing file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def file_list(self, directory: str = ".", pattern: str = "*") -> Dict[str, Any]:
        """
        List files in directory.
        
        Args:
            directory: Directory path (relative to workspace)
            pattern: Glob pattern (default: *)
            
        Returns:
            Dict with success status and file list or error
        """
        try:
            path = self._validate_path(directory)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {directory}"
                }
            
            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Not a directory: {directory}"
                }
            
            # List files matching pattern
            files = []
            for item in path.glob(pattern):
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(self.workspace_dir)),
                    "type": "file" if item.is_file() else "directory",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            # Limit number of files
            if len(files) > self.max_files_per_dir:
                files = files[:self.max_files_per_dir]
                truncated = True
            else:
                truncated = False
            
            logger.info(f"✅ Listed directory: {directory} ({len(files)} items)")
            
            return {
                "success": True,
                "directory": str(path.relative_to(self.workspace_dir)),
                "files": files,
                "count": len(files),
                "truncated": truncated
            }
            
        except Exception as e:
            logger.error(f"❌ Error listing directory {directory}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def file_exists(self, file_path: str) -> Dict[str, Any]:
        """
        Check if file exists.
        
        Args:
            file_path: Path to file (relative to workspace)
            
        Returns:
            Dict with success status and existence info
        """
        try:
            path = self._validate_path(file_path)
            
            exists = path.exists()
            is_file = path.is_file() if exists else None
            is_dir = path.is_dir() if exists else None
            size = path.stat().st_size if exists and is_file else None
            
            return {
                "success": True,
                "exists": exists,
                "is_file": is_file,
                "is_directory": is_dir,
                "size": size,
                "file_path": str(path.relative_to(self.workspace_dir))
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
_file_ops_tool: Optional[FileOperationsTool] = None


def get_file_operations_tool(workspace_dir: str = "/tmp/nis_workspace") -> FileOperationsTool:
    """Get or create file operations tool instance."""
    global _file_ops_tool
    if _file_ops_tool is None:
        _file_ops_tool = FileOperationsTool(workspace_dir=workspace_dir)
    return _file_ops_tool
