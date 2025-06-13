"""
NIS Protocol Web Search Agent

Integrates web search capabilities with the Cognitive Orchestra for deep research.
Supports multiple search providers and uses GPT-4o/Gemini for intelligent analysis.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp

# Try to import optional dependencies
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class SearchProvider(Enum):
    """Available search providers."""
    GOOGLE_CSE = "google_cse"
    SERPER = "serper"
    TAVILY = "tavily"
    BING = "bing"


class ResearchDomain(Enum):
    """Research domain specializations."""
    ARCHAEOLOGICAL = "archaeological"
    CULTURAL = "cultural"
    HISTORICAL = "historical"
    SCIENTIFIC = "scientific"
    ENVIRONMENTAL = "environmental"
    GENERAL = "general"


@dataclass
class SearchResult:
    """Represents a search result."""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    domain: str
    timestamp: float
    metadata: Dict[str, Any] = None


@dataclass
class ResearchQuery:
    """Represents a research query with context."""
    query: str
    domain: ResearchDomain
    context: Dict[str, Any]
    max_results: int = 10
    academic_sources_only: bool = False
    cultural_sensitivity: bool = True
    language: str = "en"


class WebSearchAgent:
    """Advanced web search agent with cognitive orchestra integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the web search agent."""
        self.logger = logging.getLogger("web_search_agent")
        self.config = config or self._load_config()
        
        # Initialize search providers
        self.search_providers = {}
        self._initialize_search_providers()
        
        # Initialize LLM for query generation and synthesis
        self._initialize_llm_providers()
        
        # Research cache
        self.search_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Domain-specific configurations
        self.domain_configs = {
            ResearchDomain.ARCHAEOLOGICAL: {
                "academic_sources": True,
                "cultural_sensitivity": True,
                "preferred_domains": ["jstor.org", "cambridge.org", "academia.edu"],
                "keywords_boost": ["archaeology", "cultural heritage", "excavation"]
            },
            ResearchDomain.CULTURAL: {
                "academic_sources": True,
                "cultural_sensitivity": True,
                "preferred_domains": ["unesco.org", "smithsonianmag.com"],
                "keywords_boost": ["culture", "indigenous", "heritage"]
            }
        }
        
        self.logger.info("Web Search Agent initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "google_cse_id": os.getenv("GOOGLE_CSE_ID"),
            "serper_api_key": os.getenv("SERPER_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "bing_api_key": os.getenv("BING_SEARCH_API_KEY"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "max_results_per_provider": 10,
            "timeout": 30,
            "cultural_sensitivity": True,
            "academic_priority": True
        }
    
    def _initialize_search_providers(self):
        """Initialize available search providers."""
        # Google Custom Search Engine
        if self.config.get("google_api_key") and self.config.get("google_cse_id"):
            self.search_providers[SearchProvider.GOOGLE_CSE] = self._search_google_cse
            self.logger.info("Google CSE search provider initialized")
        
        # Serper API
        if self.config.get("serper_api_key"):
            self.search_providers[SearchProvider.SERPER] = self._search_serper
            self.logger.info("Serper search provider initialized")
        
        # Tavily API (research-focused)
        if self.config.get("tavily_api_key"):
            self.search_providers[SearchProvider.TAVILY] = self._search_tavily
            self.logger.info("Tavily search provider initialized")
        
        if not self.search_providers:
            self.logger.warning("No search providers configured - using mock search")
            self.search_providers[SearchProvider.GOOGLE_CSE] = self._search_mock
    
    def _initialize_llm_providers(self):
        """Initialize LLM providers for query generation and synthesis."""
        self.llm_providers = {}
        
        # Initialize Gemini for query generation
        if GEMINI_AVAILABLE and self.config.get("google_api_key"):
            try:
                genai.configure(api_key=self.config["google_api_key"])
                self.llm_providers["gemini"] = genai.GenerativeModel('gemini-pro')
                self.logger.info("Gemini LLM provider initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize Gemini: {e}")
    
    async def research(
        self,
        query: Union[str, ResearchQuery],
        domain: ResearchDomain = ResearchDomain.GENERAL,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic."""
        # Convert string query to ResearchQuery object
        if isinstance(query, str):
            research_query = ResearchQuery(
                query=query,
                domain=domain,
                context=context or {},
                cultural_sensitivity=True
            )
        else:
            research_query = query
        
        self.logger.info(f"Starting research: {research_query.query}")
        
        try:
            # Generate enhanced search queries
            enhanced_queries = await self._generate_enhanced_queries(research_query)
            
            # Execute searches across multiple providers
            search_results = await self._execute_multi_provider_search(enhanced_queries, research_query)
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(search_results, research_query)
            
            # Synthesize findings
            synthesis = await self._synthesize_research_findings(filtered_results, research_query)
            
            # Generate research report
            research_report = {
                "query": research_query.query,
                "domain": research_query.domain.value,
                "enhanced_queries": enhanced_queries,
                "total_results": len(search_results),
                "filtered_results": len(filtered_results),
                "top_results": filtered_results[:10],
                "synthesis": synthesis,
                "sources": [result.url for result in filtered_results[:10]],
                "timestamp": time.time()
            }
            
            self.logger.info(f"Research completed: {len(filtered_results)} results")
            return research_report
            
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            return {
                "query": research_query.query,
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }
    
    async def _generate_enhanced_queries(self, research_query: ResearchQuery) -> List[str]:
        """Generate enhanced search queries."""
        base_query = research_query.query
        domain = research_query.domain
        
        enhanced_queries = [base_query]
        
        # Add domain-specific variations
        if domain == ResearchDomain.ARCHAEOLOGICAL:
            enhanced_queries.extend([
                f"{base_query} archaeological evidence",
                f"{base_query} cultural heritage preservation"
            ])
        elif domain == ResearchDomain.CULTURAL:
            enhanced_queries.extend([
                f"{base_query} cultural significance",
                f"{base_query} indigenous perspectives"
            ])
        
        return list(dict.fromkeys(enhanced_queries))[:5]
    
    async def _execute_multi_provider_search(
        self,
        queries: List[str],
        research_query: ResearchQuery
    ) -> List[SearchResult]:
        """Execute searches across multiple providers."""
        all_results = []
        
        for query in queries:
            for provider, search_func in self.search_providers.items():
                try:
                    results = await search_func(query, research_query)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.warning(f"Search failed for {provider.value}: {e}")
        
        # Remove duplicates based on URL
        unique_results = {}
        for result in all_results:
            if result.url not in unique_results:
                unique_results[result.url] = result
        
        return list(unique_results.values())
    
    async def _search_google_cse(self, query: str, research_query: ResearchQuery) -> List[SearchResult]:
        """Search using Google Custom Search Engine."""
        if not self.config.get("google_api_key") or not self.config.get("google_cse_id"):
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.config["google_api_key"],
            "cx": self.config["google_cse_id"],
            "q": query,
            "num": min(research_query.max_results, 10)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_results(data, "google_cse")
        except Exception as e:
            self.logger.error(f"Google CSE search failed: {e}")
        
        return []
    
    async def _search_serper(self, query: str, research_query: ResearchQuery) -> List[SearchResult]:
        """Search using Serper API."""
        if not self.config.get("serper_api_key"):
            return []
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.config["serper_api_key"],
            "Content-Type": "application/json"
        }
        data = {
            "q": query,
            "num": min(research_query.max_results, 10)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return self._parse_serper_results(result_data, "serper")
        except Exception as e:
            self.logger.error(f"Serper search failed: {e}")
        
        return []
    
    async def _search_tavily(self, query: str, research_query: ResearchQuery) -> List[SearchResult]:
        """Search using Tavily API."""
        if not self.config.get("tavily_api_key"):
            return []
        
        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {self.config['tavily_api_key']}",
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "search_depth": "advanced",
            "include_academic": True,
            "max_results": min(research_query.max_results, 10)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return self._parse_tavily_results(result_data, "tavily")
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
        
        return []
    
    async def _search_mock(self, query: str, research_query: ResearchQuery) -> List[SearchResult]:
        """Mock search for testing."""
        return [
            SearchResult(
                title=f"Mock Research: {query}",
                url="https://example.com/research",
                snippet=f"Research findings on {query}...",
                source="mock",
                relevance_score=0.9,
                domain="example.com",
                timestamp=time.time(),
                metadata={"mock": True}
            )
        ]
    
    def _parse_google_results(self, data: Dict[str, Any], source: str) -> List[SearchResult]:
        """Parse Google CSE results."""
        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source=source,
                relevance_score=1.0,
                domain=item.get("displayLink", ""),
                timestamp=time.time()
            ))
        return results
    
    def _parse_serper_results(self, data: Dict[str, Any], source: str) -> List[SearchResult]:
        """Parse Serper API results."""
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source=source,
                relevance_score=1.0,
                domain=item.get("domain", ""),
                timestamp=time.time()
            ))
        return results
    
    def _parse_tavily_results(self, data: Dict[str, Any], source: str) -> List[SearchResult]:
        """Parse Tavily API results."""
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=source,
                relevance_score=item.get("score", 0.5),
                domain=item.get("url", "").split('/')[2] if item.get("url") else "",
                timestamp=time.time()
            ))
        return results
    
    def _filter_and_rank_results(
        self,
        results: List[SearchResult],
        research_query: ResearchQuery
    ) -> List[SearchResult]:
        """Filter and rank search results."""
        domain_config = self.domain_configs.get(research_query.domain, {})
        
        # Score and filter results
        scored_results = []
        for result in results:
            score = result.relevance_score
            
            # Boost academic sources
            if domain_config.get("academic_sources"):
                academic_domains = ["jstor.org", "academia.edu", "researchgate.net"]
                if any(domain in result.url for domain in academic_domains):
                    score += 0.3
            
            # Cultural sensitivity filtering
            if research_query.cultural_sensitivity:
                sensitive_terms = ["primitive", "savage", "backward"]
                if any(term in result.snippet.lower() for term in sensitive_terms):
                    score -= 0.5
            
            result.relevance_score = min(score, 1.0)
            scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_results
    
    async def _synthesize_research_findings(
        self,
        results: List[SearchResult],
        research_query: ResearchQuery
    ) -> Dict[str, Any]:
        """Synthesize research findings."""
        if not results:
            return {"summary": "No relevant results found"}
        
        # Basic synthesis
        return {
            "summary": f"Found {len(results)} relevant sources on {research_query.query}",
            "key_findings": [r.snippet for r in results[:5] if r.snippet],
            "sources": [r.url for r in results[:10]],
            "synthesis_method": "basic"
        }
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get research agent statistics."""
        return {
            "search_providers": list(self.search_providers.keys()),
            "llm_providers": list(self.llm_providers.keys()),
            "domain_configs": list(self.domain_configs.keys()),
            "cache_size": len(self.search_cache)
        } 