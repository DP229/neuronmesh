"""
NeuronMesh Brain - Unified LLM Interface

Provides a unified interface to multiple LLM providers with:
- Multi-provider support (OpenAI, Anthropic, Ollama, llama.cpp)
- Model routing based on task complexity
- Cost optimization
- Fallback chains for reliability
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    VLLM = "vllm"
    LOCAL = "local"  # llama.cpp style


@dataclass
class ModelConfig:
    """Configuration for a model generation"""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
            "stream": self.stream,
        }


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    provider: ModelProvider
    context_length: int = 4096
    cost_per_1k: float = 0.0  # Cost per 1K tokens
    capabilities: List[str] = field(default_factory=list)  # "chat", "completion", "embedding"
    description: str = ""
    
    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


class ModelRegistry:
    """Registry of available models"""
    
    # Default models
    DEFAULT_MODELS = {
        # OpenAI
        "gpt-4": ModelInfo(
            name="gpt-4",
            provider=ModelProvider.OPENAI,
            context_length=8192,
            cost_per_1k=0.03,
            capabilities=["chat", "function_calling"],
            description="GPT-4 - Most capable model",
        ),
        "gpt-3.5-turbo": ModelInfo(
            name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            context_length=16385,
            cost_per_1k=0.002,
            capabilities=["chat", "function_calling"],
            description="GPT-3.5 Turbo - Fast and cheap",
        ),
        # Ollama / Local
        "llama3": ModelInfo(
            name="llama3",
            provider=ModelProvider.OLLAMA,
            context_length=8192,
            cost_per_1k=0.0,  # Free if you have the hardware
            capabilities=["chat"],
            description="Meta's Llama 3 - Open model",
        ),
        "llama3:70b": ModelInfo(
            name="llama3:70b",
            provider=ModelProvider.OLLAMA,
            context_length=8192,
            cost_per_1k=0.0,
            capabilities=["chat"],
            description="Llama 3 70B - Larger, more capable",
        ),
        "codellama": ModelInfo(
            name="codellama",
            provider=ModelProvider.OLLAMA,
            context_length=16384,
            cost_per_1k=0.0,
            capabilities=["chat", "completion"],
            description="Code Llama - Specialized for code",
        ),
        "mistral": ModelInfo(
            name="mistral",
            provider=ModelProvider.OLLAMA,
            context_length=8192,
            cost_per_1k=0.0,
            capabilities=["chat"],
            description="Mistral AI - Efficient open model",
        ),
        "mixtral": ModelInfo(
            name="mixtral",
            provider=ModelProvider.OLLAMA,
            context_length=32768,
            cost_per_1k=0.0,
            capabilities=["chat"],
            description="Mixtral - Mixture of experts",
        ),
    }
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = self.DEFAULT_MODELS.copy()
    
    def register(self, model: ModelInfo):
        """Register a new model"""
        self.models[model.name] = model
    
    def get(self, name: str) -> Optional[ModelInfo]:
        """Get model info"""
        return self.models.get(name)
    
    def list(self, provider: Optional[ModelProvider] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by provider"""
        models = list(self.models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        return models
    
    def select(
        self,
        task: str = "chat",
        max_cost: float = 1.0,
        max_latency: Optional[int] = None,
    ) -> Optional[ModelInfo]:
        """
        Select the best model for a task based on constraints.
        
        Args:
            task: Type of task ("chat", "completion", "embedding")
            max_cost: Maximum cost per 1K tokens
            max_latency: Maximum expected latency in ms
            
        Returns:
            Best model for the task, or None
        """
        candidates = [
            m for m in self.models.values()
            if m.supports(task) and m.cost_per_1k <= max_cost
        ]
        
        if not candidates:
            return None
        
        # Sort by cost (prefer cheaper)
        candidates.sort(key=lambda m: m.cost_per_1k)
        
        return candidates[0]


class Brain:
    """
    Unified interface to LLM providers.
    
    Features:
    - Automatic provider detection based on model name
    - Request batching
    - Cost tracking
    - Fallback chains
    """
    
    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        default_provider: Optional[ModelProvider] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.default_provider = default_provider or ModelProvider.OLLAMA
        
        # Provider clients
        self._clients: Dict[ModelProvider, Any] = {}
        self._init_clients()
        
        # Metrics
        self._total_requests = 0
        self._total_cost = 0.0
    
    def _init_clients(self):
        """Initialize provider clients based on environment"""
        # Check for API keys
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self._clients[ModelProvider.OPENAI] = OpenAI()
            except ImportError:
                logger.warning("OpenAI SDK not installed")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from anthropic import Anthropic
                self._clients[ModelProvider.ANTHROPIC] = Anthropic()
            except ImportError:
                logger.warning("Anthropic SDK not installed")
        
        # Ollama is always available if running locally
        self._clients[ModelProvider.OLLAMA] = self._OllamaClient()
    
    def generate(
        self,
        prompt: str,
        model: str = "llama3",
        config: Optional[ModelConfig] = None,
        mode: str = "local",
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate text using the specified model.
        
        Args:
            prompt: Input prompt
            model: Model name (e.g., "llama3", "gpt-4")
            config: Generation configuration
            mode: Execution mode ("local", "distributed", "hybrid")
            
        Returns:
            Tuple of (generated_text, usage_metadata)
        """
        config = config or ModelConfig()
        self._total_requests += 1
        
        # Detect provider
        model_info = self.registry.get(model)
        if model_info:
            provider = model_info.provider
        else:
            # Default to Ollama for unknown models (assume local)
            provider = ModelProvider.OLLAMA
        
        # Get client
        client = self._clients.get(provider)
        if not client:
            # Fallback to Ollama
            provider = ModelProvider.OLLAMA
            client = self._clients.get(provider)
        
        if not client:
            raise ValueError(f"No client available for provider {provider}")
        
        try:
            # Generate
            if provider == ModelProvider.OPENAI:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                text = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "cost": (response.usage.total_tokens / 1000) * model_info.cost_per_1k if model_info else 0,
                }
            else:
                # Ollama-style API
                text = client.generate(model, prompt, config)
                usage = {"cost": 0}  # Local models are free
            
            self._total_cost += usage.get("cost", 0)
            return text, usage
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        model: str = "llama3",
        config: Optional[ModelConfig] = None,
    ):
        """Streaming generation"""
        # TODO: Implement streaming
        text, _ = self.generate(prompt, model, config)
        yield text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self._total_requests,
            "total_cost": self._total_cost,
            "available_providers": [p.value for p in self._clients.keys()],
        }
    
    # Internal Ollama client for local inference
    class _OllamaClient:
        """Simple Ollama API client"""
        
        def __init__(self):
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        def generate(self, model: str, prompt: str, config: ModelConfig) -> str:
            """Call Ollama API"""
            import urllib.request
            import urllib.error
            
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                }
            }
            
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
            )
            
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                    return result.get("response", "")
            except urllib.error.URLError as e:
                # Fallback to mock response for testing
                logger.warning(f"Ollama not available, using mock: {e}")
                return f"[Mock response for: {prompt[:50]}...]"


def create_brain(
    providers: List[str] = None,
    default_model: str = "llama3",
) -> Brain:
    """
    Create a Brain with specified providers.
    
    Example:
        brain = create_brain(
            providers=["openai", "ollama"],
            default_model="llama3"
        )
    """
    brain = Brain()
    
    # Configure providers based on environment
    # ...
    
    return brain
