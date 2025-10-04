#!/usr/bin/env python3
"""
NIS Protocol - Neuro-Inspired System Protocol
===============================================

A comprehensive autonomous AI framework with physics validation,
multi-agent orchestration, and LLM integration.

Example Usage::

    from nis_protocol import NISCore, AutonomousOrchestrator
    
    # Initialize core system
    nis = NISCore()
    
    # Create autonomous orchestrator
    orchestrator = AutonomousOrchestrator(nis.llm_provider)
    
    # Process user request autonomously
    intent = await orchestrator.analyze_intent("Calculate fibonacci(10)")
    plan = await orchestrator.create_execution_plan("Calculate fibonacci(10)", intent)
    results = await orchestrator.execute_plan(plan, "Calculate fibonacci(10)", {})

For more examples, see the documentation at:
https://github.com/Organica-Ai-Solutions/NIS_Protocol
"""

__version__ = "3.2.1"
__author__ = "Organica AI Solutions"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Organica AI Solutions"

# Import core components
import sys
from pathlib import Path

# Add src to path for backward compatibility
_src_path = Path(__file__).parent.parent / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Core imports
try:
    from src.llm.llm_manager import GeneralLLMProvider as LLMProvider
    from src.agents.autonomous_orchestrator import (
        get_autonomous_orchestrator,
        AutonomousOrchestrator,
        IntentType,
        ToolCapability
    )
    from src.meta.unified_coordinator import create_scientific_coordinator, BehaviorMode
    from src.core.state_manager import GlobalStateManager
    
    __all__ = [
        # Version info
        "__version__",
        "__author__",
        "__license__",
        
        # Core classes
        "NISCore",
        "LLMProvider",
        "AutonomousOrchestrator",
        "get_autonomous_orchestrator",
        
        # Enums
        "IntentType",
        "ToolCapability",
        "BehaviorMode",
        
        # Utilities
        "create_scientific_coordinator",
        "GlobalStateManager",
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Some NIS Protocol components could not be imported: {e}. "
        "Make sure all dependencies are installed. "
        "Run: pip install nis-protocol[full]",
        ImportWarning
    )
    __all__ = ["__version__", "__author__", "__license__"]


class NISCore:
    """
    Main NIS Protocol core system.
    
    This class provides a high-level interface to the NIS Protocol,
    making it easy to integrate into other projects.
    
    Example::
    
        from nis_protocol import NISCore
        
        # Initialize
        nis = NISCore()
        
        # Get LLM response
        response = await nis.llm_provider.generate_response(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        
        # Use autonomous orchestrator
        result = await nis.process_autonomously("Calculate fibonacci(10)")
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize NIS Protocol core.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize LLM provider
        try:
            self.llm_provider = LLMProvider()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize LLM provider: {e}", RuntimeWarning)
            self.llm_provider = None
        
        # Initialize autonomous orchestrator
        try:
            self.orchestrator = get_autonomous_orchestrator(self.llm_provider)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize autonomous orchestrator: {e}", RuntimeWarning)
            self.orchestrator = None
    
    async def process_autonomously(self, message: str, context: dict = None) -> dict:
        """
        Process a message autonomously.
        
        The system will automatically:
        1. Detect intent
        2. Select appropriate tools
        3. Execute them
        4. Return comprehensive results
        
        Args:
            message: User message to process
            context: Optional context dictionary
            
        Returns:
            Dictionary with results including:
            - intent: Detected intent type
            - tools_used: List of tools used
            - outputs: Results from each tool
            - response: Final response text
            - reasoning: Explanation of what was done
        """
        if not self.orchestrator:
            raise RuntimeError("Autonomous orchestrator not available")
        
        # Analyze intent
        intent = await self.orchestrator.analyze_intent(message)
        
        # Create execution plan
        plan = await self.orchestrator.create_execution_plan(message, intent)
        
        # Execute plan
        context = context or {}
        results = await self.orchestrator.execute_plan(plan, message, context)
        
        return {
            "intent": intent.value,
            "tools_used": results["tools_used"],
            "outputs": results["outputs"],
            "response": results["outputs"].get("llm_provider", {}).get("response", ""),
            "reasoning": f"Detected {intent.value} intent and used {', '.join(results['tools_used'])}",
            "success": results["success"]
        }
    
    def get_llm_response(self, message: str, provider: str = None) -> dict:
        """
        Get a simple LLM response.
        
        Args:
            message: User message
            provider: Optional provider name ('openai', 'anthropic', etc.)
            
        Returns:
            Dictionary with 'content' key containing the response
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not available")
        
        import asyncio
        return asyncio.run(
            self.llm_provider.generate_response(
                messages=[{"role": "user", "content": message}],
                requested_provider=provider
            )
        )


def get_version() -> str:
    """Get the current version of NIS Protocol."""
    return __version__


def get_info() -> dict:
    """Get information about the NIS Protocol package."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "copyright": __copyright__,
        "repository": "https://github.com/Organica-Ai-Solutions/NIS_Protocol",
        "documentation": "https://github.com/Organica-Ai-Solutions/NIS_Protocol/tree/main/system/docs",
    }


# Convenience function for quick setup
def quick_start():
    """
    Quick start guide for NIS Protocol.
    
    Prints installation and usage instructions.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║                  NIS Protocol Quick Start                      ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    1. Installation:
       pip install nis-protocol
       
       Or with all features:
       pip install nis-protocol[full]
    
    2. Basic Usage:
       
       from nis_protocol import NISCore
       
       # Initialize
       nis = NISCore()
       
       # Get LLM response
       response = nis.get_llm_response("Hello, NIS!")
       print(response['content'])
    
    3. Autonomous Mode:
       
       import asyncio
       from nis_protocol import NISCore
       
       async def main():
           nis = NISCore()
           result = await nis.process_autonomously(
               "Calculate fibonacci(10)"
           )
           print(result['response'])
       
       asyncio.run(main())
    
    4. Start Server:
       
       nis-server
       
       Then visit: http://localhost:8000
    
    For more information:
    https://github.com/Organica-Ai-Solutions/NIS_Protocol
    """)

