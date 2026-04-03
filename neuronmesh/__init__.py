"""
NeuronMesh - Distributed Intelligent Autoagent Platform

Core package for building AI agents with intelligent memory.
"""

__version__ = "0.1.0"

from neuronmesh.agent import Agent
from neuronmesh.memory import Memory, MemoryEntry
from neuronmesh.brain import Brain, ModelConfig
from neuronmesh.openloop import OpenLoopClient

__all__ = [
    "Agent",
    "Memory", 
    "MemoryEntry",
    "Brain",
    "ModelConfig",
    "OpenLoopClient",
]
