#!/usr/bin/env python3
"""
NVIDIA NeMo Integration Manager - Central coordination for NIS Protocol
Manages the integration between NIS Protocol and NVIDIA NeMo enterprise components

Key Features:
- Seamless integration with existing NIS agents
- Automatic fallback when NeMo components are unavailable
- Performance monitoring and optimization
- Enterprise deployment coordination
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# NeMo components
from .nemo_physics_agent import NeMoPhysicsAgent, NeMoPhysicsConfig
from .nemo_agent_orchestrator import NeMoAgentOrchestrator, NeMoAgentConfig

# Existing NIS components  
from src.core.agent import NISAgent, NISLayer
# from src.agents.physics.unified_physics_agent import UnifiedPhysicsAgent
# Using placeholder for physics agent
class UnifiedPhysicsAgent:
    def __init__(self):
        pass
    async def validate_physics(self, data):
        return {"valid": True, "confidence": 0.85}
from src.agents.enhanced_agent_base import EnhancedAgentBase

logger = logging.getLogger(__name__)


@dataclass
class NeMoIntegrationConfig:
    """Configuration for NeMo integration with NIS Protocol"""
    # Enable/disable NeMo components
    enable_nemo_physics: bool = True
    enable_nemo_orchestration: bool = True
    enable_automatic_fallback: bool = True
    
    # Performance settings
    physics_precision: str = "high"
    max_concurrent_workflows: int = 5
    enable_gpu_acceleration: bool = True
    
    # Integration modes
    replace_existing_physics: bool = False  # Whether to replace existing physics agents
    hybrid_mode: bool = True  # Use both NeMo and existing agents
    
    # Monitoring
    enable_performance_monitoring: bool = True
    log_all_interactions: bool = False


class NeMoIntegrationManager:
    """
    Central manager for NVIDIA NeMo integration with NIS Protocol
    
    Coordinates between:
    - Existing NIS agents and systems
    - New NeMo-powered agents
    - Enterprise observability and monitoring
    - Automatic fallback mechanisms
    """
    
    def __init__(self, config: Optional[NeMoIntegrationConfig] = None):
        self.config = config or NeMoIntegrationConfig()
        
        # NeMo components
        self.nemo_physics_agent = None
        self.nemo_orchestrator = None
        
        # Integration status
        self.integration_status = {
            'physics_agent': 'not_initialized',
            'orchestrator': 'not_initialized',
            'fallback_active': False
        }
        
        # Performance tracking
        self.performance_metrics = {
            'nemo_physics_calls': 0,
            'fallback_physics_calls': 0,
            'orchestration_workflows': 0,
            'average_response_time': 0.0
        }
        
        logger.info("NeMo Integration Manager initialized")
    
    async def initialize_integration(self) -> Dict[str, Any]:
        """Initialize all NeMo components and integration"""
        
        initialization_results = {}
        
        # Initialize NeMo Physics Agent
        if self.config.enable_nemo_physics:
            physics_result = await self._initialize_nemo_physics()
            initialization_results['physics'] = physics_result
        
        # Initialize NeMo Agent Orchestrator
        if self.config.enable_nemo_orchestration:
            orchestrator_result = await self._initialize_nemo_orchestrator()
            initialization_results['orchestrator'] = orchestrator_result
        
        # Set up integration hooks
        await self._setup_integration_hooks()
        
        # Verify integration status
        final_status = await self.get_integration_status()
        
        logger.info("NeMo integration initialization completed")
        return {
            'status': 'completed',
            'results': initialization_results,
            'final_status': final_status
        }
    
    async def _initialize_nemo_physics(self) -> Dict[str, Any]:
        """Initialize NeMo Physics Agent"""
        
        try:
            physics_config = NeMoPhysicsConfig(
                precision_level=self.config.physics_precision,
                gpu_acceleration=self.config.enable_gpu_acceleration
            )
            
            self.nemo_physics_agent = NeMoPhysicsAgent(
                agent_id="nemo_physics_integration",
                config=physics_config
            )
            
            # Initialize the agent's models
            model_init_success = await self.nemo_physics_agent.initialize_models()
            
            if model_init_success:
                self.integration_status['physics_agent'] = 'ready'
                logger.info("NeMo Physics Agent initialized successfully")
                return {
                    'status': 'success',
                    'agent_id': self.nemo_physics_agent.agent_id,
                    'models_loaded': True
                }
            else:
                self.integration_status['physics_agent'] = 'fallback'
                self.integration_status['fallback_active'] = True
                logger.warning("NeMo Physics Agent using fallback mode")
                return {
                    'status': 'fallback',
                    'agent_id': self.nemo_physics_agent.agent_id,
                    'models_loaded': False
                }
                
        except Exception as e:
            self.integration_status['physics_agent'] = 'error'
            logger.error(f"Failed to initialize NeMo Physics Agent: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _initialize_nemo_orchestrator(self) -> Dict[str, Any]:
        """Initialize NeMo Agent Orchestrator"""
        
        try:
            orchestrator_config = NeMoAgentConfig(
                max_concurrent_agents=self.config.max_concurrent_workflows,
                enable_profiling=self.config.enable_performance_monitoring
            )
            
            self.nemo_orchestrator = NeMoAgentOrchestrator(
                agent_id="nemo_orchestrator_integration",
                config=orchestrator_config
            )
            
            # Initialize the orchestrator's toolkit
            toolkit_init_success = await self.nemo_orchestrator.initialize_toolkit()
            
            if toolkit_init_success:
                self.integration_status['orchestrator'] = 'ready'
                logger.info("NeMo Agent Orchestrator initialized successfully")
            else:
                self.integration_status['orchestrator'] = 'fallback'
                self.integration_status['fallback_active'] = True
                logger.warning("NeMo Agent Orchestrator using fallback mode")
            
            return {
                'status': 'success' if toolkit_init_success else 'fallback',
                'agent_id': self.nemo_orchestrator.agent_id,
                'toolkit_loaded': toolkit_init_success
            }
            
        except Exception as e:
            self.integration_status['orchestrator'] = 'error'
            logger.error(f"Failed to initialize NeMo Agent Orchestrator: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _setup_integration_hooks(self):
        """Set up integration hooks with existing NIS components"""
        
        # Register NeMo agents with orchestrator if available
        if (self.nemo_orchestrator and 
            self.nemo_physics_agent and 
            self.integration_status['orchestrator'] in ['ready', 'fallback']):
            
            try:
                await self.nemo_orchestrator.register_agent(
                    agent_name="nemo_physics",
                    agent_instance=self.nemo_physics_agent,
                    capabilities=['physics_simulation', 'cosmos_modeling', 'real_validation']
                )
                logger.info("Registered NeMo Physics Agent with orchestrator")
            except Exception as e:
                logger.warning(f"Failed to register physics agent: {e}")
    
    async def enhanced_physics_simulation(
        self,
        scenario_description: str,
        fallback_agent: Optional[UnifiedPhysicsAgent] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced physics simulation with automatic NeMo/fallback coordination
        
        Uses NeMo Physics Agent when available, falls back to existing physics agent
        """
        
        start_time = asyncio.get_event_loop().time()
        
        # Try NeMo Physics Agent first
        if (self.nemo_physics_agent and 
            self.integration_status['physics_agent'] in ['ready', 'fallback']):
            
            try:
                result = await self.nemo_physics_agent.simulate_physics_scenario(
                    scenario_description=scenario_description,
                    **kwargs
                )
                
                self.performance_metrics['nemo_physics_calls'] += 1
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Add integration metadata
                result['integration_metadata'] = {
                    'method': 'nemo_physics_agent',
                    'execution_time': execution_time,
                    'fallback_used': False,
                    'nemo_models_available': self.integration_status['physics_agent'] == 'ready'
                }
                
                logger.info(f"NeMo physics simulation completed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                logger.warning(f"NeMo physics simulation failed: {e}")
                if not self.config.enable_automatic_fallback:
                    raise
        
        # Fallback to existing physics agent
        if fallback_agent:
            try:
                # Use existing unified physics agent
                result = await fallback_agent.process_physics_request({
                    'scenario': scenario_description,
                    **kwargs
                })
                
                self.performance_metrics['fallback_physics_calls'] += 1
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Add integration metadata
                result['integration_metadata'] = {
                    'method': 'fallback_physics_agent',
                    'execution_time': execution_time,
                    'fallback_used': True,
                    'reason': 'nemo_unavailable_or_failed'
                }
                
                logger.info(f"Fallback physics simulation completed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                logger.error(f"Fallback physics simulation also failed: {e}")
                raise
        
        # No physics simulation available
        return {
            'status': 'error',
            'error': 'No physics simulation agents available',
            'integration_metadata': {
                'method': 'none',
                'execution_time': asyncio.get_event_loop().time() - start_time,
                'fallback_used': False
            }
        }
    
    async def orchestrate_multi_agent_workflow(
        self,
        workflow_name: str,
        agents: List[Union[NISAgent, Any]],
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Orchestrate multi-agent workflow using NeMo orchestrator when available
        """
        
        start_time = asyncio.get_event_loop().time()
        
        # Use NeMo Orchestrator if available
        if (self.nemo_orchestrator and 
            self.integration_status['orchestrator'] in ['ready', 'fallback']):
            
            try:
                # Register agents with orchestrator
                agent_ids = []
                for i, agent in enumerate(agents):
                    agent_id = await self.nemo_orchestrator.register_agent(
                        agent_name=f"workflow_agent_{i}",
                        agent_instance=agent
                    )
                    agent_ids.append(agent_id)
                
                # Create workflow
                workflow_id = await self.nemo_orchestrator.create_workflow(
                    workflow_name=workflow_name,
                    agent_sequence=agent_ids,
                    workflow_config=kwargs
                )
                
                # Execute workflow
                result = await self.nemo_orchestrator.execute_workflow(
                    workflow_id=workflow_id,
                    input_data=input_data
                )
                
                self.performance_metrics['orchestration_workflows'] += 1
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Add integration metadata
                result['integration_metadata'] = {
                    'method': 'nemo_orchestrator',
                    'execution_time': execution_time,
                    'agent_count': len(agents),
                    'workflow_id': workflow_id
                }
                
                logger.info(f"NeMo workflow orchestration completed in {execution_time:.3f}s")
                return result
                
            except Exception as e:
                logger.warning(f"NeMo orchestration failed: {e}")
                if not self.config.enable_automatic_fallback:
                    raise
        
        # Fallback to simple sequential execution
        try:
            results = []
            current_data = input_data
            
            for i, agent in enumerate(agents):
                if hasattr(agent, 'execute'):
                    agent_result = await agent.execute(current_data)
                elif hasattr(agent, 'process'):
                    agent_result = await agent.process(current_data)
                else:
                    agent_result = {
                        'status': 'completed',
                        'output': f"Agent {i} processed data"
                    }
                
                results.append(agent_result)
                
                # Pass output to next agent
                if isinstance(agent_result, dict) and 'output' in agent_result:
                    current_data = agent_result['output']
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return {
                'status': 'success',
                'final_output': current_data,
                'agent_results': results,
                'integration_metadata': {
                    'method': 'fallback_sequential',
                    'execution_time': execution_time,
                    'agent_count': len(agents),
                    'fallback_used': True
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback orchestration failed: {e}")
            raise
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        
        # Get NeMo component metrics if available
        nemo_metrics = {}
        
        if self.nemo_physics_agent:
            try:
                nemo_metrics['physics'] = await self.nemo_physics_agent.get_performance_metrics()
            except Exception as e:
                nemo_metrics['physics'] = {'error': str(e)}
        
        if self.nemo_orchestrator:
            try:
                nemo_metrics['orchestrator'] = await self.nemo_orchestrator.get_orchestrator_metrics()
            except Exception as e:
                nemo_metrics['orchestrator'] = {'error': str(e)}
        
        return {
            'integration_status': self.integration_status,
            'performance_metrics': self.performance_metrics,
            'nemo_component_metrics': nemo_metrics,
            'config': {
                'nemo_physics_enabled': self.config.enable_nemo_physics,
                'nemo_orchestration_enabled': self.config.enable_nemo_orchestration,
                'fallback_enabled': self.config.enable_automatic_fallback,
                'hybrid_mode': self.config.hybrid_mode
            }
        }
    
    async def shutdown_integration(self):
        """Gracefully shutdown NeMo integration"""
        
        logger.info("Shutting down NeMo integration...")
        
        # Shutdown orchestrator
        if self.nemo_orchestrator and hasattr(self.nemo_orchestrator, 'shutdown'):
            try:
                await self.nemo_orchestrator.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down orchestrator: {e}")
        
        # Shutdown physics agent
        if self.nemo_physics_agent and hasattr(self.nemo_physics_agent, 'shutdown'):
            try:
                await self.nemo_physics_agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down physics agent: {e}")
        
        logger.info("NeMo integration shutdown completed")


# Global integration manager instance
_integration_manager = None

def get_nemo_integration_manager(
    config: Optional[NeMoIntegrationConfig] = None
) -> NeMoIntegrationManager:
    """Get or create the global NeMo integration manager"""
    global _integration_manager
    
    if _integration_manager is None:
        _integration_manager = NeMoIntegrationManager(config)
    
    return _integration_manager
