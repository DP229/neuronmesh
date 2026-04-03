"""
NeuronMesh Agent - Core agent implementation
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

from neuronmesh.brain import Brain, ModelConfig
from neuronmesh.memory import Memory


class AgentMode(Enum):
    """Agent execution modes"""
    LOCAL = "local"           # Run on local machine
    DISTRIBUTED = "distributed"  # Run via OpenPool
    HYBRID = "hybrid"         # Both local + distributed


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str = "agent"
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful AI assistant."
    mode: AgentMode = AgentMode.LOCAL
    tools: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "mode": self.mode.value,
        }


@dataclass
class Message:
    """A message in the conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent"""
    content: str
    messages: List[Message]
    latency_ms: int
    cost: float = 0.0
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "model": self.model,
            "metadata": self.metadata,
        }


class Agent:
    """
    Core NeuronMesh agent with intelligent memory.
    
    Example:
        agent = Agent(model="llama3")
        memory = Memory()
        
        response = agent.run("Hello!", memory=memory)
        print(response.content)
    """
    
    def __init__(
        self,
        name: str = "agent",
        model: str = "llama3",
        brain: Optional[Brain] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.name = name
        self.model = model
        self.brain = brain or Brain()
        self.config = config or AgentConfig(name=name, model=model)
        
        # Conversation history
        self.messages: List[Message] = []
        
        # Initialize system message
        self.messages.append(Message(
            role="system",
            content=self.config.system_prompt
        ))
    
    def run(
        self,
        prompt: str,
        memory: Optional[Memory] = None,
        mode: Optional[AgentMode] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Run the agent with a prompt.
        
        Args:
            prompt: User input
            memory: Optional memory for context retrieval
            mode: Execution mode (local/distributed/hybrid)
            **kwargs: Additional arguments passed to model
            
        Returns:
            AgentResponse with content and metadata
        """
        start_time = time.time()
        
        # Add user message
        self.messages.append(Message(role="user", content=prompt))
        
        # Build context from memory if provided
        context = ""
        if memory:
            relevant = memory.retrieve(prompt, limit=5)
            if relevant:
                context = "\n\n## Relevant Memory:\n" + "\n".join([
                    f"- {entry.content}" 
                    for entry in relevant
                ])
        
        # Build final prompt with context
        full_prompt = self._build_prompt(context)
        
        # Call brain (LLM)
        response_text, usage = self.brain.generate(
            prompt=full_prompt,
            model=self.model,
            config=ModelConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
            mode=mode.value if mode else self.config.mode.value,
        )
        
        # Add assistant response
        self.messages.append(Message(role="assistant", content=response_text))
        
        # Store in memory if provided
        if memory:
            # Extract facts from conversation (simple heuristic)
            self._store_interactions(memory)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return AgentResponse(
            content=response_text,
            messages=self.messages[-10:],  # Last 10 messages
            latency_ms=latency_ms,
            cost=usage.get("cost", 0) if usage else 0,
            model=self.model,
            metadata={"mode": mode.value if mode else self.config.mode.value},
        )
    
    async def run_async(
        self,
        prompt: str,
        memory: Optional[Memory] = None,
        **kwargs,
    ) -> AgentResponse:
        """Async version of run()"""
        # For now, just wrap sync version
        # In production, this would be truly async
        return self.run(prompt, memory, **kwargs)
    
    def _build_prompt(self, context: str = "") -> str:
        """Build the full prompt with context"""
        messages_for_context = self.messages[-20:]  # Last 20 messages
        
        prompt_parts = []
        for msg in messages_for_context:
            if msg.role == "system":
                continue
            prefix = "User" if msg.role == "user" else "Assistant"
            prompt_parts.append(f"{prefix}: {msg.content}")
        
        if context:
            prompt_parts.append(f"\n{context}")
        
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    def _store_interactions(self, memory: Memory):
        """Store recent interactions in memory"""
        # Simple heuristic: store last 2 user-assistant pairs
        if len(self.messages) >= 2:
            user_msg = None
            for msg in reversed(self.messages[:-1]):
                if msg.role == "user":
                    user_msg = msg
                    break
            
            if user_msg:
                memory.add(
                    content=f"User preference: {user_msg.content}",
                    entry_type="preference",
                    importance=0.5,
                )
    
    def reset(self):
        """Reset conversation history (keep system message)"""
        self.messages = [self.messages[0]]  # Keep system message
    
    def get_history(self) -> List[Message]:
        """Get conversation history"""
        return self.messages.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state"""
        return {
            "name": self.name,
            "model": self.model,
            "config": self.config.to_dict(),
            "message_count": len(self.messages),
        }


# Convenience function for quick agents
def create_agent(
    name: str = "agent",
    model: str = "llama3",
    instructions: str = "You are a helpful assistant.",
) -> Agent:
    """
    Create a simple agent with default settings.
    
    Example:
        agent = create_agent("coder", model="codellama")
        response = agent.run("Write a hello world in Python")
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=instructions,
    )
    return Agent(name=name, model=model, config=config)
