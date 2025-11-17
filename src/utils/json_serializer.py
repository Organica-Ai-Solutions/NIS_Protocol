#!/usr/bin/env python3
"""
JSON Serialization Utilities for NIS Protocol
Handles numpy arrays and other non-serializable types
"""

import numpy as np
from typing import Any, Dict, List, Union
from dataclasses import is_dataclass, asdict
import logging

logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """
    Convert any object to a JSON-serializable format.
    
    Handles:
    - numpy arrays -> lists
    - numpy scalars -> Python scalars
    - dataclasses -> dicts
    - nested structures (dicts, lists, tuples)
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version of the object
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    
    # Handle numpy complex numbers
    if isinstance(obj, np.complexfloating):
        return {"real": obj.real.item(), "imag": obj.imag.item()}
    
    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        return make_json_serializable(asdict(obj))
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    
    # Handle complex numbers
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    
    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return obj.hex()
    
    # Return as-is if already JSON-serializable
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Try to convert to string as last resort
    try:
        # Check if it's a simple object with __dict__
        if hasattr(obj, '__dict__'):
            return make_json_serializable(obj.__dict__)
        else:
            return str(obj)
    except Exception as e:
        logger.warning(f"Could not serialize object of type {type(obj)}: {e}")
        return f"<non-serializable: {type(obj).__name__}>"


def sanitize_response(response: Any) -> Any:
    """
    Sanitize a response object for JSON serialization.
    
    This is a convenience wrapper around make_json_serializable
    specifically for API responses.
    
    Args:
        response: Response object (dict, list, or other)
        
    Returns:
        JSON-serializable version of the response
    """
    return make_json_serializable(response)


def convert_numpy_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all numpy arrays and scalars in a dictionary to JSON-serializable types.
    
    This is a specialized version for dictionaries that's more efficient
    when you know you're dealing with dict responses.
    
    Args:
        data: Dictionary that may contain numpy types
        
    Returns:
        Dictionary with all numpy types converted
    """
    return make_json_serializable(data)


# Convenience function for agent results
def serialize_agent_result(result: Any) -> Dict[str, Any]:
    """
    Serialize an agent result for JSON response.
    
    Handles both dict results and dataclass results from agents.
    
    Args:
        result: Agent result (dict or dataclass)
        
    Returns:
        JSON-serializable dictionary
    """
    serialized = make_json_serializable(result)
    
    # Ensure it's a dictionary
    if not isinstance(serialized, dict):
        return {"result": serialized}
    
    return serialized
