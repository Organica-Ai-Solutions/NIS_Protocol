"""
NVIDIA Cosmos Integration for NIS Protocol

World Foundation Models for Physical AI:
- Cosmos Predict: Future state prediction (30s video generation)
- Cosmos Transfer: Synthetic data augmentation
- Cosmos Reason: Vision-language reasoning for robotics

Usage:
    from src.agents.cosmos import get_cosmos_generator, get_cosmos_reasoner
"""

from .cosmos_data_generator import CosmosDataGenerator, get_cosmos_generator
from .cosmos_reasoner import CosmosReasoner, get_cosmos_reasoner

__all__ = [
    'CosmosDataGenerator',
    'CosmosReasoner',
    'get_cosmos_generator',
    'get_cosmos_reasoner'
]
