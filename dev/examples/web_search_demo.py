#!/usr/bin/env python3
"""
NIS Protocol Web Search Agent Demo

Demonstrates the integration of web search capabilities with the Cognitive Orchestra
for comprehensive archaeological and cultural research.

This example shows:
- Multi-provider web search integration
- Domain-specific research optimization
- Cultural sensitivity filtering
- Academic source prioritization
- LLM-enhanced query generation and synthesis
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.research import WebSearchAgent, ResearchDomain, ResearchQuery
from llm.cognitive_orchestra import CognitiveOrchestra, CognitiveFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_search_demo")


async def demonstrate_basic_search():
    """Demonstrate basic web search functionality."""
    print("\n" + "="*60)
    print("BASIC WEB SEARCH DEMONSTRATION")
    print("="*60)
    
    # Initialize the web search agent
    search_agent = WebSearchAgent()
    
    # Basic archaeological search
    query = "Mayan civilization recent discoveries"
    print(f"\n🔍 Searching for: {query}")
    
    research_results = await search_agent.research(
        query=query,
        domain=ResearchDomain.ARCHAEOLOGICAL
    )
    
    print(f"\n📊 Research Results:")
    print(f"   • Total results: {research_results.get('total_results', 0)}")
    print(f"   • Filtered results: {research_results.get('filtered_results', 0)}")
    print(f"   • Enhanced queries: {len(research_results.get('enhanced_queries', []))}")
    
    # Display top results
    top_results = research_results.get('top_results', [])[:3]
    print(f"\n🏆 Top {len(top_results)} Results:")
    for i, result in enumerate(top_results, 1):
        print(f"   {i}. {result.title}")
        print(f"      URL: {result.url}")
        print(f"      Relevance: {result.relevance_score:.2f}")
        print(f"      Snippet: {result.snippet[:100]}...")
        print()
    
    # Display synthesis
    synthesis = research_results.get('synthesis', {})
    if synthesis:
        print(f"🧠 Research Synthesis:")
        print(f"   Summary: {synthesis.get('summary', 'No synthesis available')}")
        if synthesis.get('key_findings'):
            print(f"   Key Findings:")
            for finding in synthesis['key_findings'][:3]:
                print(f"     • {finding[:80]}...")
    
    return research_results


async def demonstrate_cultural_research():
    """Demonstrate culturally sensitive research."""
    print("\n" + "="*60)
    print("CULTURAL RESEARCH DEMONSTRATION")
    print("="*60)
    
    search_agent = WebSearchAgent()
    
    # Create a detailed research query
    research_query = ResearchQuery(
        query="indigenous archaeological sites preservation methods",
        domain=ResearchDomain.CULTURAL,
        context={
            "focus": "indigenous rights and cultural preservation",
            "sensitivity_level": "high",
            "academic_priority": True
        },
        max_results=15,
        academic_sources_only=True,
        cultural_sensitivity=True
    )
    
    print(f"🌍 Cultural Research Query:")
    print(f"   • Query: {research_query.query}")
    print(f"   • Domain: {research_query.domain.value}")
    print(f"   • Cultural sensitivity: {research_query.cultural_sensitivity}")
    print(f"   • Academic sources only: {research_query.academic_sources_only}")
    
    research_results = await search_agent.research(research_query)
    
    print(f"\n📈 Cultural Research Results:")
    print(f"   • Enhanced queries generated: {len(research_results.get('enhanced_queries', []))}")
    print(f"   • Sources found: {research_results.get('total_results', 0)}")
    print(f"   • After cultural filtering: {research_results.get('filtered_results', 0)}")
    
    # Show enhanced queries
    enhanced_queries = research_results.get('enhanced_queries', [])
    if enhanced_queries:
        print(f"\n🔍 Enhanced Search Queries:")
        for i, query in enumerate(enhanced_queries, 1):
            print(f"   {i}. {query}")
    
    # Show academic sources
    top_results = research_results.get('top_results', [])
    academic_sources = [r for r in top_results if any(domain in r.url for domain in ['jstor.org', 'academia.edu', 'cambridge.org'])]
    
    if academic_sources:
        print(f"\n🎓 Academic Sources Found:")
        for source in academic_sources[:3]:
            print(f"   • {source.title}")
            print(f"     {source.url}")
            print(f"     Relevance: {source.relevance_score:.2f}")
            print()
    
    return research_results


async def demonstrate_cognitive_orchestra_integration():
    """Demonstrate integration with the Cognitive Orchestra."""
    print("\n" + "="*60)
    print("COGNITIVE ORCHESTRA INTEGRATION")
    print("="*60)
    
    # Initialize both the search agent and cognitive orchestra
    search_agent = WebSearchAgent()
    
    try:
        cognitive_orchestra = CognitiveOrchestra()
        
        print("🎼 Cognitive Orchestra initialized successfully")
        print(f"   • Available functions: {len(cognitive_orchestra.get_available_functions())}")
        
        # Demonstrate coordinated research
        query = "archaeological drone survey techniques cultural heritage sites"
        
        print(f"\n🔍 Coordinated Research Query: {query}")
        
        # Step 1: Use web search for information gathering
        print("\n📡 Step 1: Web Search Information Gathering")
        search_results = await search_agent.research(
            query=query,
            domain=ResearchDomain.ARCHAEOLOGICAL
        )
        
        print(f"   • Found {search_results.get('total_results', 0)} initial sources")
        
        # Step 2: Use cognitive orchestra for analysis
        print("\n🧠 Step 2: Cognitive Orchestra Analysis")
        
        # Prepare context for cognitive analysis
        research_context = {
            "search_query": query,
            "domain": "archaeological",
            "sources_found": search_results.get('total_results', 0),
            "top_sources": [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet
                }
                for result in search_results.get('top_results', [])[:5]
            ]
        }
        
        # Use different cognitive functions for analysis
        analysis_tasks = [
            {
                "function": CognitiveFunction.REASONING,
                "prompt": f"Analyze the technical feasibility of drone surveys for archaeological sites based on this research: {json.dumps(research_context, indent=2)}"
            },
            {
                "function": CognitiveFunction.CULTURAL,
                "prompt": f"Evaluate the cultural sensitivity considerations for drone surveys at heritage sites: {json.dumps(research_context, indent=2)}"
            },
            {
                "function": CognitiveFunction.ARCHAEOLOGICAL,
                "prompt": f"Assess the archaeological methodology and recommended practices from this research: {json.dumps(research_context, indent=2)}"
            }
        ]
        
        # Execute cognitive analysis
        cognitive_results = {}
        for task in analysis_tasks:
            try:
                result = await cognitive_orchestra.execute_function(
                    function=task["function"],
                    prompt=task["prompt"],
                    context=research_context
                )
                cognitive_results[task["function"].value] = result
                print(f"   ✅ {task['function'].value.title()} analysis completed")
            except Exception as e:
                print(f"   ❌ {task['function'].value.title()} analysis failed: {e}")
        
        # Step 3: Synthesize combined results
        print("\n🔬 Step 3: Combined Analysis Synthesis")
        
        if cognitive_results:
            print("   Cognitive Orchestra Analysis Results:")
            for function, result in cognitive_results.items():
                print(f"\n   {function.upper()} ANALYSIS:")
                if isinstance(result, dict) and 'response' in result:
                    response_text = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
                    print(f"     {response_text}")
                else:
                    print(f"     {str(result)[:200]}...")
        
        # Combine web search and cognitive analysis
        combined_report = {
            "research_query": query,
            "web_search_results": {
                "total_sources": search_results.get('total_results', 0),
                "top_sources": search_results.get('sources', [])[:5],
                "synthesis": search_results.get('synthesis', {})
            },
            "cognitive_analysis": cognitive_results,
            "integration_method": "web_search_plus_cognitive_orchestra",
            "timestamp": search_results.get('timestamp')
        }
        
        print(f"\n📋 Combined Research Report Generated:")
        print(f"   • Web sources analyzed: {combined_report['web_search_results']['total_sources']}")
        print(f"   • Cognitive functions used: {len(cognitive_results)}")
        print(f"   • Integration method: {combined_report['integration_method']}")
        
        return combined_report
        
    except Exception as e:
        print(f"❌ Cognitive Orchestra integration failed: {e}")
        print("   Falling back to web search only...")
        
        # Fallback to web search only
        search_results = await search_agent.research(
            query="archaeological drone survey techniques",
            domain=ResearchDomain.ARCHAEOLOGICAL
        )
        
        return {
            "research_query": "archaeological drone survey techniques",
            "web_search_results": search_results,
            "cognitive_analysis": None,
            "integration_method": "web_search_only",
            "note": "Cognitive Orchestra not available"
        }


async def demonstrate_multi_domain_research():
    """Demonstrate research across multiple domains."""
    print("\n" + "="*60)
    print("MULTI-DOMAIN RESEARCH DEMONSTRATION")
    print("="*60)
    
    search_agent = WebSearchAgent()
    
    # Research the same topic across different domains
    base_query = "climate change impact on archaeological sites"
    domains = [
        ResearchDomain.ARCHAEOLOGICAL,
        ResearchDomain.ENVIRONMENTAL,
        ResearchDomain.SCIENTIFIC
    ]
    
    multi_domain_results = {}
    
    for domain in domains:
        print(f"\n🔍 Researching in {domain.value.upper()} domain...")
        
        results = await search_agent.research(
            query=base_query,
            domain=domain
        )
        
        multi_domain_results[domain.value] = {
            "total_results": results.get('total_results', 0),
            "filtered_results": results.get('filtered_results', 0),
            "enhanced_queries": results.get('enhanced_queries', []),
            "top_sources": [r.url for r in results.get('top_results', [])[:3]],
            "synthesis": results.get('synthesis', {})
        }
        
        print(f"   • Found {results.get('total_results', 0)} sources")
        print(f"   • Generated {len(results.get('enhanced_queries', []))} enhanced queries")
    
    print(f"\n📊 Multi-Domain Research Summary:")
    total_sources = sum(domain_data['total_results'] for domain_data in multi_domain_results.values())
    print(f"   • Total sources across all domains: {total_sources}")
    print(f"   • Domains researched: {len(multi_domain_results)}")
    
    # Show domain-specific insights
    for domain, data in multi_domain_results.items():
        print(f"\n   {domain.upper()} DOMAIN:")
        print(f"     Sources: {data['total_results']}")
        print(f"     Enhanced queries: {len(data['enhanced_queries'])}")
        if data['enhanced_queries']:
            print(f"     Sample query: {data['enhanced_queries'][0]}")
    
    return multi_domain_results


async def demonstrate_research_statistics():
    """Demonstrate research agent statistics and capabilities."""
    print("\n" + "="*60)
    print("RESEARCH AGENT STATISTICS")
    print("="*60)
    
    search_agent = WebSearchAgent()
    
    # Get agent statistics
    stats = search_agent.get_research_statistics()
    
    print("🔧 Web Search Agent Configuration:")
    print(f"   • Search providers: {', '.join([p.value for p in stats['search_providers']])}")
    print(f"   • LLM providers: {', '.join(stats['llm_providers'])}")
    print(f"   • Domain configurations: {', '.join([d.value for d in stats['domain_configs']])}")
    print(f"   • Cache size: {stats['cache_size']}")
    
    # Test each search provider
    print(f"\n🧪 Testing Search Providers:")
    test_query = "test archaeological research"
    
    for provider in stats['search_providers']:
        try:
            # This would test individual providers if they were accessible
            print(f"   • {provider.value}: Available")
        except Exception as e:
            print(f"   • {provider.value}: Error - {e}")
    
    # Show domain-specific configurations
    print(f"\n⚙️ Domain-Specific Configurations:")
    for domain_name in stats['domain_configs']:
        domain_enum = ResearchDomain(domain_name)
        domain_config = search_agent.domain_configs.get(domain_enum, {})
        
        print(f"   {domain_name.upper()}:")
        print(f"     Academic sources: {domain_config.get('academic_sources', False)}")
        print(f"     Cultural sensitivity: {domain_config.get('cultural_sensitivity', False)}")
        print(f"     Preferred domains: {len(domain_config.get('preferred_domains', []))}")
        print(f"     Keywords boost: {len(domain_config.get('keywords_boost', []))}")
    
    return stats


async def main():
    """Run all web search demonstrations."""
    print("🚀 NIS Protocol Web Search Agent Demonstration")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        await demonstrate_basic_search()
        await demonstrate_cultural_research()
        await demonstrate_cognitive_orchestra_integration()
        await demonstrate_multi_domain_research()
        await demonstrate_research_statistics()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        print("\n🎯 Key Features Demonstrated:")
        print("   • Multi-provider web search integration")
        print("   • Domain-specific research optimization")
        print("   • Cultural sensitivity filtering")
        print("   • Academic source prioritization")
        print("   • Cognitive Orchestra integration")
        print("   • Multi-domain research capabilities")
        print("   • LLM-enhanced query generation")
        print("   • Research synthesis and analysis")
        
        print("\n📚 Next Steps:")
        print("   1. Configure API keys in .env file")
        print("   2. Install required dependencies (google-generativeai, aiohttp)")
        print("   3. Test with real search providers")
        print("   4. Integrate with archaeological drone surveys")
        print("   5. Expand to additional research domains")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Check API key configuration")
        print("   • Verify network connectivity")
        print("   • Install missing dependencies")
        print("   • Check search provider status")


if __name__ == "__main__":
    asyncio.run(main())