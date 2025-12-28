#!/usr/bin/env python3
"""
Database Query Tool for NIS Protocol
Enables autonomous agents to query databases safely

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


class DatabaseQueryTool:
    """
    Safe database query execution for autonomous agents.
    
    Provides:
    - db_query: Execute SELECT queries
    - db_schema: Get table schema information
    - db_tables: List available tables
    
    Security:
    - Read-only queries (SELECT only)
    - No write operations (INSERT, UPDATE, DELETE)
    - Query timeout limits
    - Result size limits
    """
    
    def __init__(self, workspace_dir: str = "/tmp/nis_workspace"):
        """Initialize database query tool."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Security limits
        self.max_results = 1000
        self.query_timeout = 30  # seconds
        
        # Allowed SQL keywords (read-only)
        self.allowed_keywords = {"SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", 
                                "INNER", "OUTER", "ON", "GROUP", "BY", "HAVING", 
                                "ORDER", "LIMIT", "OFFSET", "AS", "AND", "OR", "NOT",
                                "IN", "BETWEEN", "LIKE", "IS", "NULL", "DISTINCT"}
        
        # Forbidden keywords (write operations)
        self.forbidden_keywords = {"INSERT", "UPDATE", "DELETE", "DROP", "CREATE", 
                                  "ALTER", "TRUNCATE", "REPLACE", "MERGE"}
        
        logger.info(f"🗄️ Database query tool initialized: {self.workspace_dir}")
    
    def _validate_path(self, db_path: str) -> Path:
        """Validate database path within workspace."""
        path = Path(db_path)
        if not path.is_absolute():
            path = self.workspace_dir / path
        path = path.resolve()
        
        try:
            path.relative_to(self.workspace_dir)
        except ValueError:
            raise ValueError(f"Database path outside workspace: {db_path}")
        
        return path
    
    def _validate_query(self, query: str) -> bool:
        """
        Validate query is read-only.
        
        Args:
            query: SQL query string
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        query_upper = query.upper()
        
        # Check for forbidden keywords
        for keyword in self.forbidden_keywords:
            if keyword in query_upper:
                raise ValueError(f"Forbidden keyword in query: {keyword}")
        
        # Must start with SELECT
        if not query_upper.strip().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        return True
    
    def db_query(
        self,
        db_path: str,
        query: str,
        params: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute read-only database query.
        
        Args:
            db_path: Path to SQLite database (relative to workspace)
            query: SQL SELECT query
            params: Query parameters (for parameterized queries)
            
        Returns:
            Dict with success status and results or error
        """
        try:
            # Validate query
            self._validate_query(query)
            
            # Validate and resolve path
            path = self._validate_path(db_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Database not found: {db_path}"
                }
            
            # Connect to database
            conn = sqlite3.connect(str(path), timeout=self.query_timeout)
            conn.row_factory = sqlite3.Row  # Return rows as dicts
            cursor = conn.cursor()
            
            # Execute query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results
            rows = cursor.fetchmany(self.max_results)
            
            # Convert to list of dicts
            results = [dict(row) for row in rows]
            
            # Check if more results available
            has_more = len(cursor.fetchmany(1)) > 0
            
            conn.close()
            
            logger.info(f"✅ Query executed: {len(results)} rows returned")
            
            return {
                "success": True,
                "results": results,
                "row_count": len(results),
                "has_more": has_more,
                "query": query
            }
            
        except ValueError as e:
            logger.error(f"❌ Query validation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        except sqlite3.Error as e:
            logger.error(f"❌ Database error: {e}")
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def db_schema(self, db_path: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            db_path: Path to SQLite database
            table_name: Optional specific table name
            
        Returns:
            Dict with schema information
        """
        try:
            path = self._validate_path(db_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Database not found: {db_path}"
                }
            
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            
            if table_name:
                # Get schema for specific table
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                schema = {
                    "table": table_name,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "not_null": bool(col[3]),
                            "default": col[4],
                            "primary_key": bool(col[5])
                        }
                        for col in columns
                    ]
                }
            else:
                # Get schema for all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                schema = {
                    "tables": []
                }
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    schema["tables"].append({
                        "name": table_name,
                        "columns": [
                            {
                                "name": col[1],
                                "type": col[2],
                                "not_null": bool(col[3]),
                                "default": col[4],
                                "primary_key": bool(col[5])
                            }
                            for col in columns
                        ]
                    })
            
            conn.close()
            
            logger.info(f"✅ Schema retrieved for: {db_path}")
            
            return {
                "success": True,
                "schema": schema
            }
            
        except Exception as e:
            logger.error(f"❌ Schema error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def db_tables(self, db_path: str) -> Dict[str, Any]:
        """
        List all tables in database.
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Dict with table list
        """
        try:
            path = self._validate_path(db_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Database not found: {db_path}"
                }
            
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            logger.info(f"✅ Tables listed: {len(tables)} tables")
            
            return {
                "success": True,
                "tables": tables,
                "count": len(tables)
            }
            
        except Exception as e:
            logger.error(f"❌ Tables error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
_db_query_tool: Optional[DatabaseQueryTool] = None


def get_database_query_tool(workspace_dir: str = "/tmp/nis_workspace") -> DatabaseQueryTool:
    """Get or create database query tool instance."""
    global _db_query_tool
    if _db_query_tool is None:
        _db_query_tool = DatabaseQueryTool(workspace_dir=workspace_dir)
    return _db_query_tool
