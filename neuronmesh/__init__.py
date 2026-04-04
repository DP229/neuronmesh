"""
NeuronMesh - Distributed Intelligent Autoagent Platform

Core package for building AI agents with intelligent memory and tools.
"""

__version__ = "0.1.0"

from neuronmesh.agent import (
    Agent,
    AgentConfig,
    AgentMode,
    AgentResponse,
    Turn,
    ToolCall,
    Message,
    create_agent,
    create_coder_agent,
    create_researcher_agent,
)

from neuronmesh.memory import Memory, MemoryEntry, MemoryType, MemoryImportance

from neuronmesh.brain import (
    Brain,
    ModelConfig,
    ModelInfo,
    ModelProvider,
    ModelRegistry,
    create_brain,
)

from neuronmesh.tools import (
    ToolRegistry,
    ToolDefinition,
    ToolResult,
    ToolCategory,
    BaseTool,
    create_default_tools,
    # Built-in tools
    BashTool,
    ReadFileTool,
    WriteFileTool,
    GlobTool,
    GrepTool,
    WebSearchTool,
    MemorySearchTool,
    MemoryStoreTool,
)

from neuronmesh.openloop import OpenLoopClient

__all__ = [
    # Version
    "__version__",
    
    # Agent
    "Agent",
    "AgentConfig",
    "AgentMode",
    "AgentResponse",
    "Turn",
    "ToolCall",
    "Message",
    "create_agent",
    "create_coder_agent",
    "create_researcher_agent",
    
    # Memory
    "Memory",
    "MemoryEntry",
    "MemoryType",
    
    # Brain
    "Brain",
    "ModelConfig",
    "ModelInfo",
    "ModelProvider",
    "ModelRegistry",
    "create_brain",
    
    # Tools
    "ToolRegistry",
    "ToolDefinition",
    "ToolResult",
    "ToolCategory",
    "BaseTool",
    "create_default_tools",
    "BashTool",
    "ReadFileTool",
    "WriteFileTool",
    "GlobTool",
    "GrepTool",
    "WebSearchTool",
    "MemorySearchTool",
    "MemoryStoreTool",
    
    # OpenLoop
    "OpenLoopClient",
]
