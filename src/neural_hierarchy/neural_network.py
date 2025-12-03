from typing import Dict, List, Optional, Any
from collections import defaultdict
from .base_neural_agent import NeuralAgent, NeuralLayer, NeuralSignal

class NeuralNetwork:
    """Coordinator for the neural network of agents"""
    
    def __init__(self):
        # Agents organized by layer
        self.agents: Dict[NeuralLayer, List[NeuralAgent]] = defaultdict(list)
        
        # Signal queues for each layer
        self.signal_queues: Dict[NeuralLayer, List[NeuralSignal]] = defaultdict(list)
        
        # Network state
        self.global_activation = 0.0
        self.active_signals = []
        
    def add_agent(self, agent: NeuralAgent):
        """Add an agent to the network"""
        self.agents[agent.layer].append(agent)
    
    def connect_agents(
        self,
        source_agent: NeuralAgent,
        target_agent: NeuralAgent,
        connection_strength: float = 1.0
    ):
        """Create a connection between two agents"""
        source_agent.connect_to(target_agent, connection_strength)
    
    def process_input(self, input_data: Any) -> Dict[str, Any]:
        """Process input through the neural network"""
        # Start with sensory layer
        sensory_signal = NeuralSignal(
            source_layer=NeuralLayer.SENSORY,
            target_layer=NeuralLayer.SENSORY,
            content=input_data
        )
        
        self.signal_queues[NeuralLayer.SENSORY].append(sensory_signal)
        
        # Process through each layer in order
        layer_order = [
            NeuralLayer.SENSORY,
            NeuralLayer.PERCEPTION,
            NeuralLayer.MEMORY,
            NeuralLayer.EMOTIONAL,
            NeuralLayer.EXECUTIVE,
            NeuralLayer.MOTOR
        ]
        
        results = {}
        for layer in layer_order:
            layer_results = self._process_layer(layer)
            if layer_results:
                results[layer.value] = layer_results
        
        return results
    
    def _process_layer(self, layer: NeuralLayer) -> List[Dict[str, Any]]:
        """Process all signals for a given layer"""
        if not self.signal_queues[layer]:
            return []
            
        results = []
        # Process each signal through all agents in the layer
        while self.signal_queues[layer]:
            signal = self.signal_queues[layer].pop(0)
            
            for agent in self.agents[layer]:
                # Update agent activation based on signal priority
                agent.update_activation(signal.priority)
                
                # Only process if agent is active
                if agent.is_active():
                    response = agent.process_signal(signal)
                    if response:
                        # Add response to appropriate queue
                        self.signal_queues[response.target_layer].append(response)
                        results.append({
                            'agent_id': agent.agent_id,
                            'response': response.content
                        })
        
        return results
    
    def get_layer_activation(self, layer: NeuralLayer) -> float:
        """Get average activation level for a layer"""
        if not self.agents[layer]:
            return 0.0
            
        return sum(
            agent.activation_level
            for agent in self.agents[layer]
        ) / len(self.agents[layer])
    
    def reset_network(self):
        """Reset all agents and clear signal queues"""
        for agents in self.agents.values():
            for agent in agents:
                agent.reset()
        
        self.signal_queues.clear()
        self.global_activation = 0.0
        self.active_signals = []
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get current state of the neural network"""
        return {
            'layer_activations': {
                layer.value: self.get_layer_activation(layer)
                for layer in NeuralLayer
            },
            'active_agents': [
                {
                    'agent_id': agent.agent_id,
                    'layer': agent.layer.value,
                    'activation': agent.activation_level
                }
                for agents in self.agents.values()
                for agent in agents
                if agent.is_active()
            ],
            'queue_sizes': {
                layer.value: len(queue)
                for layer, queue in self.signal_queues.items()
            }
        } 