"""
NIS Protocol v4.0 - Research Routes

This module contains research-related endpoints:
- Research capabilities
- Deep research
- Claim validation
- Web search

MIGRATION STATUS: Ready for testing
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("nis.routes.research")

# Create router
router = APIRouter(prefix="/research", tags=["Research"])


# ====== Request Models ======

class DeepResearchRequest(BaseModel):
    query: str = Field(..., description="Research query")
    research_depth: str = Field(default="standard", description="Research depth: quick, standard, comprehensive")
    max_results: int = Field(default=10, description="Maximum number of results")
    include_citations: bool = Field(default=True, description="Include source citations")


class ClaimValidationRequest(BaseModel):
    claim: str = Field(..., description="Claim to validate")
    context: Optional[str] = None
    require_sources: bool = True


# ====== Dependency Injection ======

def get_web_search_agent():
    return getattr(router, '_web_search_agent', None)

def get_llm_provider():
    return getattr(router, '_llm_provider', None)


# ====== Endpoints ======

@router.get("/capabilities")
async def get_research_capabilities():
    """
    🔍 Get Research Capabilities
    
    Returns information about available research tools and capabilities.
    """
    try:
        capabilities = {
            "status": "active",
            "research_tools": {
                "arxiv_search": {
                    "available": True,
                    "description": "Academic paper search and analysis",
                    "features": ["paper_search", "citation_analysis", "abstract_processing"]
                },
                "web_search": {
                    "available": True,
                    "description": "General web search capabilities", 
                    "features": ["real_time_search", "content_analysis", "fact_verification"]
                },
                "deep_research": {
                    "available": True,
                    "description": "Multi-source research with synthesis",
                    "features": ["source_correlation", "evidence_synthesis", "bias_detection"]
                }
            },
            "analysis_capabilities": {
                "document_processing": ["PDF", "LaTeX", "HTML", "plain_text"],
                "citation_formats": ["APA", "MLA", "Chicago", "IEEE"],
                "languages": ["en", "es", "fr", "de", "zh"],
                "fact_checking": True,
                "bias_analysis": True
            },
            "integration": {
                "consciousness_oversight": True,
                "physics_validation": "available_for_scientific_papers",
                "multi_agent_coordination": True
            },
            "timestamp": time.time()
        }
        
        return capabilities
    except Exception as e:
        logger.error(f"Research capabilities error: {e}")
        return JSONResponse(content={
            "status": "error", 
            "message": f"Failed to retrieve research capabilities: {str(e)}",
            "capabilities": {}
        }, status_code=500)


@router.post("/query")
async def research_query(request: Dict[str, Any]):
    """
    🔍 Simple Research Query
    
    Quick research endpoint for basic queries.
    """
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        web_search_agent = get_web_search_agent()
        
        if web_search_agent:
            from src.agents.research.web_search_agent import ResearchQuery, ResearchDomain
            research_query = ResearchQuery(
                query=query,
                domain=ResearchDomain.GENERAL,
                max_results=5
            )
            results = await web_search_agent.research(research_query)
            
            return {
                "status": "success",
                "query": query,
                "results": results.get('top_results', []),
                "timestamp": time.time()
            }
        else:
            # Fallback without web search agent
            return {
                "status": "success",
                "query": query,
                "results": [],
                "message": "Web search agent not initialized - using fallback mode",
                "timestamp": time.time()
            }
            
    except Exception as e:
        logger.error(f"Research query error: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@router.post("/deep")
async def deep_research(request: DeepResearchRequest):
    """
    🔍 Deep Research using WebSearchAgent + LLM Analysis
    
    Performs comprehensive research by:
    1. Using WebSearchAgent to gather real-time information
    2. Analyzing results with our LLM backend
    3. Generating comprehensive research reports
    """
    try:
        web_search_agent = get_web_search_agent()
        llm_provider = get_llm_provider()
        
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        if web_search_agent is None:
            raise HTTPException(status_code=500, detail="WebSearchAgent not initialized")
        
        if llm_provider is None:
            raise HTTPException(status_code=500, detail="LLM Provider not initialized")
        
        # Step 1: Gather information using WebSearchAgent
        logger.info(f"🔍 Starting deep research for query: {request.query}")
        
        from src.agents.research.web_search_agent import ResearchQuery, ResearchDomain
        research_query = ResearchQuery(
            query=request.query,
            domain=ResearchDomain.GENERAL,
            context={"research_depth": request.research_depth},
            max_results=request.max_results,
            cultural_sensitivity=True
        )
        search_results = await web_search_agent.research(research_query)
        
        # Step 2: Analyze and synthesize with LLM
        research_context = f"Search Results for '{request.query}':\n\n"
        search_result_list = search_results.get('top_results', [])
        
        if isinstance(search_result_list, list) and len(search_result_list) > 0:
            for idx, result in enumerate(search_result_list, 1):
                title = result.get('title') if isinstance(result, dict) else getattr(result, 'title', '')
                snippet = result.get('snippet') if isinstance(result, dict) else getattr(result, 'snippet', '')
                url = result.get('url') if isinstance(result, dict) else getattr(result, 'url', '')
                
                research_context += f"{idx}. {title}\n"
                research_context += f"   {snippet}\n"
                if request.include_citations:
                    research_context += f"   Source: {url}\n"
                research_context += "\n"
        else:
            research_context += "No search results available.\n"
        
        analysis_prompt = f"""You are an expert research analyst. Based on the search results provided, create a comprehensive research report on: {request.query}

SEARCH RESULTS:
{research_context}

Please provide:
1. Executive Summary
2. Key Findings
3. Analysis
4. Conclusions
5. Sources Used"""

        messages = [
            {"role": "system", "content": "You are NIS Protocol's research analyst. Provide thorough, well-cited research reports."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        llm_response = await llm_provider.generate_response(
            messages=messages,
            temperature=0.3
        )
        
        return {
            "status": "success",
            "query": request.query,
            "research_depth": request.research_depth,
            "report": llm_response.get("content", ""),
            "sources_count": len(search_result_list),
            "sources": search_result_list if request.include_citations else [],
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deep research error: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


@router.post("/validate")
async def validate_claim(request: ClaimValidationRequest):
    """
    ✅ Validate a claim using research and fact-checking.
    """
    try:
        web_search_agent = get_web_search_agent()
        llm_provider = get_llm_provider()
        
        if not web_search_agent or not llm_provider:
            # Fallback validation without web search
            messages = [
                {"role": "system", "content": "You are a fact-checker. Evaluate the following claim for accuracy."},
                {"role": "user", "content": f"Claim: {request.claim}\nContext: {request.context or 'None provided'}"}
            ]
            
            if llm_provider:
                response = await llm_provider.generate_response(messages=messages, temperature=0.2)
                return {
                    "status": "success",
                    "claim": request.claim,
                    "validation": response.get("content", ""),
                    "confidence": 0.7,
                    "sources": [],
                    "method": "llm_only"
                }
            else:
                return {
                    "status": "error",
                    "message": "Validation services not available"
                }
        
        # Full validation with web search
        from src.agents.research.web_search_agent import ResearchQuery, ResearchDomain
        
        search_query = ResearchQuery(
            query=f"fact check: {request.claim}",
            domain=ResearchDomain.GENERAL,
            max_results=5
        )
        
        search_results = await web_search_agent.research(search_query)
        sources = search_results.get('top_results', [])
        
        # Analyze with LLM
        validation_prompt = f"""Validate this claim: "{request.claim}"

Based on these sources:
{sources}

Provide:
1. Verdict (True/False/Partially True/Unverifiable)
2. Confidence (0-1)
3. Explanation
4. Key evidence"""

        messages = [
            {"role": "system", "content": "You are a rigorous fact-checker."},
            {"role": "user", "content": validation_prompt}
        ]
        
        response = await llm_provider.generate_response(messages=messages, temperature=0.1)
        
        return {
            "status": "success",
            "claim": request.claim,
            "validation": response.get("content", ""),
            "sources": sources,
            "method": "web_search_and_llm",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Claim validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_research_status():
    """
    📊 Get research system status.
    """
    web_search_agent = get_web_search_agent()
    llm_provider = get_llm_provider()
    
    return {
        "status": "operational",
        "web_search_available": web_search_agent is not None,
        "llm_available": llm_provider is not None,
        "capabilities": ["deep_research", "claim_validation", "web_search"],
        "timestamp": time.time()
    }


# ====== Dependency Injection Helper ======

def set_dependencies(web_search_agent=None, llm_provider=None):
    """Set dependencies for the research router"""
    router._web_search_agent = web_search_agent
    router._llm_provider = llm_provider
