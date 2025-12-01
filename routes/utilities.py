"""
NIS Protocol v4.0 - Utilities Routes

This module contains utility endpoints:
- Cost tracking
- Response cache
- Prompt templates
- Code execution

MIGRATION STATUS: Ready for testing
- These routes mirror the ones in main.py
- Can be tested independently before switching over

Usage:
    from routes.utilities import router as utilities_router
    app.include_router(utilities_router, tags=["Utilities"])
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("nis.routes.utilities")

# Create router
router = APIRouter(tags=["Utilities"])


# ====== Dependency Injection ======

def get_cost_tracker():
    tracker = getattr(router, '_cost_tracker', None)
    if tracker is None:
        try:
            from src.utils.cost_tracker import get_cost_tracker as get_tracker
            return get_tracker()
        except ImportError:
            return None
    return tracker

def get_response_cache():
    cache = getattr(router, '_response_cache', None)
    if cache is None:
        try:
            from src.utils.response_cache import get_response_cache as get_cache
            return get_cache()
        except ImportError:
            return None
    return cache

def get_template_manager():
    manager = getattr(router, '_template_manager', None)
    if manager is None:
        try:
            from src.utils.prompt_templates import get_template_manager as get_manager
            return get_manager()
        except ImportError:
            return None
    return manager

def get_code_executor():
    return getattr(router, '_code_executor', None)


# ====== Cost Tracking Endpoints ======

@router.get("/costs/session")
async def get_session_costs():
    """
    💰 Get current session cost summary
    
    Returns:
        - Total requests and tokens
        - Cost breakdown by provider
        - Average latency
    """
    tracker = get_cost_tracker()
    
    if not tracker:
        return {"success": False, "message": "Cost tracker not available"}
    
    return {
        "success": True,
        "data": tracker.get_session_summary()
    }


@router.get("/costs/daily")
async def get_daily_costs(date: Optional[str] = None):
    """
    📅 Get daily cost summary
    
    Args:
        date: Optional date in YYYY-MM-DD format (defaults to today)
    """
    tracker = get_cost_tracker()
    
    if not tracker:
        return {"success": False, "message": "Cost tracker not available"}
    
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        dt = datetime.now()
    
    return {
        "success": True,
        "data": tracker.get_daily_summary(dt)
    }


@router.get("/costs/monthly")
async def get_monthly_costs(year: Optional[int] = None, month: Optional[int] = None):
    """
    📊 Get monthly cost summary
    """
    tracker = get_cost_tracker()
    
    if not tracker:
        return {"success": False, "message": "Cost tracker not available"}
    
    return {
        "success": True,
        "data": tracker.get_monthly_summary(year, month)
    }


@router.get("/costs/estimate")
async def estimate_monthly_costs():
    """
    🔮 Estimate monthly costs based on current usage
    """
    tracker = get_cost_tracker()
    
    if not tracker:
        return {"success": False, "message": "Cost tracker not available"}
    
    return {
        "success": True,
        "data": tracker.estimate_monthly_cost()
    }


# ====== Cache Endpoints ======

@router.get("/cache/stats")
async def get_cache_stats():
    """
    📊 Get cache statistics
    
    Returns hit rate, tokens saved, cost saved
    """
    cache = get_response_cache()
    
    if not cache:
        return {"success": False, "message": "Response cache not available"}
    
    return {
        "success": True,
        "data": cache.get_stats()
    }


@router.post("/cache/clear")
async def clear_cache():
    """
    🗑️ Clear all cached responses
    """
    cache = get_response_cache()
    
    if not cache:
        return {"success": False, "message": "Response cache not available"}
    
    cache.clear()
    return {
        "success": True,
        "message": "Cache cleared"
    }


# ====== Template Endpoints ======

@router.get("/templates")
async def list_templates(category: Optional[str] = None):
    """
    📋 List available prompt templates
    
    Args:
        category: Optional filter by category (analysis, coding, writing, etc.)
    """
    manager = get_template_manager()
    
    if not manager:
        return {"success": False, "message": "Template manager not available", "templates": []}
    
    try:
        from src.utils.prompt_templates import TemplateCategory
        
        if category:
            try:
                cat = TemplateCategory(category)
                templates = [
                    {
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "variables": t.variables
                    }
                    for t in manager.list_by_category(cat)
                ]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        else:
            templates = manager.list_all()
        
        return {
            "success": True,
            "templates": templates,
            "categories": [c.value for c in TemplateCategory]
        }
    except ImportError:
        return {"success": False, "message": "Template system not available", "templates": []}


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """
    📄 Get a specific template with details
    """
    manager = get_template_manager()
    
    if not manager:
        raise HTTPException(status_code=503, detail="Template manager not available")
    
    template = manager.get(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")
    
    return {
        "success": True,
        "template": {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "category": template.category.value,
            "template": template.template,
            "variables": template.variables,
            "examples": template.examples,
            "estimated_tokens": template.estimated_tokens
        }
    }


@router.post("/templates/{template_id}/render")
async def render_template(template_id: str, request: Dict[str, Any]):
    """
    ✨ Render a template with variables
    
    Args:
        template_id: The template to use
        request body: Variable values (e.g., {"language": "python", "code": "..."})
    """
    manager = get_template_manager()
    
    if not manager:
        raise HTTPException(status_code=503, detail="Template manager not available")
    
    try:
        rendered = manager.render(template_id, **request)
        return {
            "success": True,
            "prompt": rendered
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/templates/search")
async def search_templates(request: Dict[str, Any]):
    """
    🔍 Search templates by keyword
    """
    manager = get_template_manager()
    
    if not manager:
        return {"success": False, "message": "Template manager not available", "results": []}
    
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    
    results = manager.search(query)
    
    return {
        "success": True,
        "results": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category.value
            }
            for t in results
        ]
    }


# ====== Code Execution Endpoints ======

@router.post("/execute")
async def execute_code_endpoint(request: Dict[str, Any]):
    """
    🖥️ Execute Python code safely and return results
    
    This is the critical piece that enables:
    - LLM generates code → Execute → See results → Iterate
    - Generate plots with matplotlib
    - Process data with pandas
    - Create visualizations
    - Autonomous task completion
    
    Returns stdout, plots (base64), dataframes, and any errors.
    """
    try:
        from src.execution.code_executor import execute_code as exec_code
        
        code = request.get("code") or request.get("code_content")
        if not code:
            raise HTTPException(status_code=400, detail="Code is required")
        
        timeout = request.get("timeout", 30)
        
        result = await exec_code(code, timeout_seconds=timeout)
        return {
            "success": result.success,
            "execution_id": result.execution_id,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result": str(result.result) if result.result else None,
            "plots": result.plots,
            "dataframes": result.dataframes,
            "execution_time_ms": result.execution_time_ms,
            "error": result.error
        }
    except ImportError:
        return {
            "success": False,
            "error": "Code executor not available"
        }
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/execute/plot")
async def execute_and_plot(request: Dict[str, Any]):
    """
    📊 Execute code that generates a matplotlib plot
    
    Convenience endpoint that wraps code in plot setup.
    Just provide the plotting code, we handle plt.figure() and capture.
    """
    try:
        from src.execution.code_executor import execute_code as exec_code
        
        code = request.get("code", "")
        title = request.get("title", "Generated Plot")
        
        # Wrap code with plot setup
        wrapped_code = f'''
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
{code}
plt.title("{title}")
plt.tight_layout()
'''
        
        result = await exec_code(wrapped_code)
        
        if result.plots:
            return {
                "success": True,
                "plot": result.plots[0],
                "execution_time_ms": result.execution_time_ms
            }
        else:
            return {
                "success": False,
                "error": result.error or "No plot generated",
                "stdout": result.stdout
            }
    except ImportError:
        return {
            "success": False,
            "error": "Code executor not available"
        }


@router.post("/execute/analyze")
async def execute_data_analysis(request: Dict[str, Any]):
    """
    📈 Execute data analysis code with pandas
    
    Provide data and analysis code, get back results and visualizations.
    """
    try:
        from src.execution.code_executor import execute_code as exec_code
        
        data = request.get("data", {})
        code = request.get("code", "")
        
        # Create DataFrame from data
        setup_code = f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load provided data
data = {json.dumps(data)}
df = pd.DataFrame(data)

# User analysis code
{code}
'''
        
        result = await exec_code(setup_code)
        
        return {
            "success": result.success,
            "stdout": result.stdout,
            "plots": result.plots,
            "dataframes": result.dataframes,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms
        }
    except ImportError:
        return {
            "success": False,
            "error": "Code executor not available"
        }


# ====== Dependency Injection Helper ======

def set_dependencies(
    cost_tracker=None,
    response_cache=None,
    template_manager=None,
    code_executor=None
):
    """Set dependencies for the utilities router"""
    router._cost_tracker = cost_tracker
    router._response_cache = response_cache
    router._template_manager = template_manager
    router._code_executor = code_executor
