#!/usr/bin/env python3
"""
Multi-Agent Negotiation Protocol for NIS Protocol
Enables agents to communicate and collaborate

Copyright 2025 Organica AI Solutions
Licensed under Apache License 2.0
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import time

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages."""
    PROPOSAL = "proposal"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    COUNTER_PROPOSAL = "counter_proposal"
    QUERY = "query"
    RESPONSE = "response"
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"


@dataclass
class AgentMessage:
    """Message between agents."""
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }


class MultiAgentNegotiator:
    """
    Multi-agent negotiation and collaboration system.
    
    Features:
    - Agent-to-agent messaging
    - Task delegation
    - Consensus building
    - Collaborative problem solving
    
    Honest Assessment:
    - Simple message passing (not complex negotiation algorithms)
    - No game theory or auction mechanisms
    - Basic consensus (majority vote)
    - Good for coordination, not true negotiation
    - 60% real - it enables communication but not sophisticated bargaining
    """
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize multi-agent negotiator.
        
        Args:
            agents: Dictionary of agent_name -> agent_instance
        """
        self.agents = agents
        self.message_queue: List[AgentMessage] = []
        self.conversation_history: Dict[str, List[AgentMessage]] = {}
        
        logger.info(f"🤝 Multi-agent negotiator initialized with {len(agents)} agents")
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send message from one agent to another.
        
        Args:
            from_agent: Sender agent name
            to_agent: Recipient agent name
            message_type: Type of message
            content: Message content
            
        Returns:
            Dict with success status
        """
        try:
            if from_agent not in self.agents:
                return {
                    "success": False,
                    "error": f"Unknown sender: {from_agent}"
                }
            
            if to_agent not in self.agents:
                return {
                    "success": False,
                    "error": f"Unknown recipient: {to_agent}"
                }
            
            # Create message
            message = AgentMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                content=content,
                timestamp=time.time(),
                message_id=f"{from_agent}_{to_agent}_{int(time.time() * 1000)}"
            )
            
            # Add to queue
            self.message_queue.append(message)
            
            # Add to conversation history
            conv_key = f"{from_agent}_{to_agent}"
            if conv_key not in self.conversation_history:
                self.conversation_history[conv_key] = []
            self.conversation_history[conv_key].append(message)
            
            logger.info(f"✅ Message sent: {from_agent} → {to_agent} ({message_type.value})")
            
            return {
                "success": True,
                "message_id": message.message_id
            }
            
        except Exception as e:
            logger.error(f"❌ Send message error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delegate_task(
        self,
        from_agent: str,
        to_agent: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delegate task from one agent to another.
        
        Args:
            from_agent: Agent delegating the task
            to_agent: Agent receiving the task
            task: Task specification
            
        Returns:
            Dict with task result
        """
        try:
            # Send task request
            await self.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=MessageType.TASK_REQUEST,
                content=task
            )
            
            # Execute task on target agent
            target_agent = self.agents[to_agent]
            
            # Assuming agents have execute_autonomous_task method
            if hasattr(target_agent, 'execute_autonomous_task'):
                result = await target_agent.execute_autonomous_task(task)
            else:
                result = {
                    "status": "error",
                    "error": f"Agent {to_agent} cannot execute tasks"
                }
            
            # Send result back
            await self.send_message(
                from_agent=to_agent,
                to_agent=from_agent,
                message_type=MessageType.TASK_RESULT,
                content=result
            )
            
            logger.info(f"✅ Task delegated: {from_agent} → {to_agent}")
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Delegate task error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def build_consensus(
        self,
        initiator: str,
        proposal: Dict[str, Any],
        agents_to_consult: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build consensus among agents for a proposal.
        
        Args:
            initiator: Agent proposing
            proposal: Proposal content
            agents_to_consult: List of agents to consult (default: all)
            
        Returns:
            Dict with consensus result
        """
        try:
            if agents_to_consult is None:
                agents_to_consult = [a for a in self.agents.keys() if a != initiator]
            
            # Send proposal to all agents
            responses = []
            for agent_name in agents_to_consult:
                await self.send_message(
                    from_agent=initiator,
                    to_agent=agent_name,
                    message_type=MessageType.PROPOSAL,
                    content=proposal
                )
                
                # Simulate agent response (in real system, agents would respond)
                # For now, simple majority vote based on agent type
                response = self._simulate_agent_response(agent_name, proposal)
                responses.append(response)
                
                await self.send_message(
                    from_agent=agent_name,
                    to_agent=initiator,
                    message_type=MessageType.ACCEPTANCE if response else MessageType.REJECTION,
                    content={"proposal_id": proposal.get("id", "unknown")}
                )
            
            # Calculate consensus
            acceptances = sum(1 for r in responses if r)
            total = len(responses)
            consensus_reached = acceptances > total / 2
            
            logger.info(f"✅ Consensus: {acceptances}/{total} accepted")
            
            return {
                "success": True,
                "consensus_reached": consensus_reached,
                "acceptances": acceptances,
                "rejections": total - acceptances,
                "total_agents": total,
                "consensus_ratio": acceptances / total if total > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Consensus error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _simulate_agent_response(self, agent_name: str, proposal: Dict[str, Any]) -> bool:
        """
        Simulate agent response to proposal.
        
        HONEST: This is a placeholder. Real agents would use LLM to evaluate proposals.
        Currently just returns True (accept) for simplicity.
        """
        # In production, agents would:
        # 1. Analyze proposal with LLM
        # 2. Check against their goals/constraints
        # 3. Make informed decision
        
        # For now, simple heuristic
        return True  # Accept by default
    
    def get_conversation_history(
        self,
        agent1: str,
        agent2: str
    ) -> Dict[str, Any]:
        """Get conversation history between two agents."""
        try:
            conv_key = f"{agent1}_{agent2}"
            reverse_key = f"{agent2}_{agent1}"
            
            messages = []
            if conv_key in self.conversation_history:
                messages.extend(self.conversation_history[conv_key])
            if reverse_key in self.conversation_history:
                messages.extend(self.conversation_history[reverse_key])
            
            # Sort by timestamp
            messages.sort(key=lambda m: m.timestamp)
            
            return {
                "success": True,
                "messages": [m.to_dict() for m in messages],
                "count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"❌ Get history error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
_negotiator: Optional[MultiAgentNegotiator] = None


def get_multi_agent_negotiator(agents: Dict[str, Any]) -> MultiAgentNegotiator:
    """Get or create multi-agent negotiator instance."""
    global _negotiator
    if _negotiator is None:
        _negotiator = MultiAgentNegotiator(agents=agents)
    return _negotiator
