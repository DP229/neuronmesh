"""
NeuronMesh OpenLoop - Distributed Execution Client

Integrates with OpenPool network for distributed task execution:
- Task submission and result retrieval
- Node discovery and selection
- Streaming responses
- Credit management
- Model sharding

Inspired by OpenPool's architecture.
"""

import os
import json
import time
import asyncio
import hashlib
from typing import Any, AsyncIterator, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a distributed task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of distributed tasks"""
    AGENT_INFER = "agent_infer"
    AGENT_TRAIN = "agent_train"
    AGENT_OPTIMIZE = "agent_optimize"
    EMBEDDING = "embedding"
    INFERENCE = "inference"
    BATCH = "batch"
    CUSTOM = "custom"


@dataclass
class TaskResult:
    """Result from a distributed task"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    latency_ms: int = 0
    cost: float = 0.0
    node_id: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "node_id": self.node_id,
            "steps": self.steps,
            "metadata": self.metadata,
        }


@dataclass
class NodeInfo:
    """Information about a compute node"""
    id: str
    multiaddr: str
    country: str = ""
    cpu_cores: int = 0
    has_gpu: bool = False
    gpu_model: str = ""
    memory_gb: int = 0
    price: int = 0  # Credits per task
    score: float = 0.0
    reliability: float = 1.0
    avg_latency_ms: int = 0
    status: str = "online"
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(
            id=data.get("id", ""),
            multiaddr=data.get("multiaddr", ""),
            country=data.get("country", ""),
            cpu_cores=data.get("cpu_cores", 0),
            has_gpu=data.get("has_gpu", False),
            gpu_model=data.get("gpu_model", ""),
            memory_gb=data.get("memory_gb", 0),
            price=data.get("price", 0),
            score=data.get("score", 0.0),
            reliability=data.get("reliability", 1.0),
            avg_latency_ms=data.get("avg_latency_ms", 0),
            status=data.get("status", "online"),
            capabilities=data.get("capabilities", []),
            last_seen=data.get("last_seen", time.time()),
        )
    
    def supports(self, task_type: TaskType) -> bool:
        """Check if node supports a task type"""
        if task_type == TaskType.AGENT_INFER and "agent" in self.capabilities:
            return True
        if task_type == TaskType.EMBEDDING and "embedding" in self.capabilities:
            return True
        if self.has_gpu and "gpu" in self.capabilities:
            return True
        return len(self.capabilities) == 0  # No requirements means supports all


@dataclass
class NetworkStats:
    """Statistics about the OpenPool network"""
    total_nodes: int = 0
    online_nodes: int = 0
    gpu_nodes: int = 0
    avg_latency_ms: int = 0
    total_tasks: int = 0
    network_load: float = 0.0


class OpenLoopClient:
    """
    Client for OpenPool distributed compute network.
    
    Features:
    - Task submission and result retrieval
    - Node discovery with scoring
    - Streaming responses
    - Credit management
    - Fallback to local execution
    
    Example:
        client = OpenLoopClient("http://localhost:8080")
        
        # Submit a task
        result = await client.run_agent(
            model="llama3",
            prompt="Hello"
        )
        
        print(result.result)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        node_id: Optional[str] = None,
        prefer_local: bool = True,
        fallback_enabled: bool = True,
    ):
        self.base_url = base_url or os.getenv("OPENPOOL_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("OPENPOOL_API_KEY", "")
        self.node_id = node_id or self._generate_node_id()
        self.prefer_local = prefer_local
        self.fallback_enabled = fallback_enabled
        
        # Cache of known nodes
        self._nodes: Dict[str, NodeInfo] = {}
        self._last_discovery = 0.0
        self._discovery_cache_ttl = 60.0  # seconds
        
        # Local execution fallback
        self._local_brain = None
        self._local_memory = None
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID"""
        return hashlib.md5(f"{os.getpid()}{time.time()}".encode()).hexdigest()[:16]
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Make HTTP request to OpenPool node"""
        url = f"{self.base_url}{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise OpenLoopError(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise OpenLoopError(f"Connection error: {e.reason}")
        except TimeoutError:
            raise OpenLoopError("Request timeout")
    
    async def _make_request_async(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Async HTTP request"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._make_request(method, endpoint, data, timeout)
        )
    
    # === Network Discovery ===
    
    async def discover_nodes(self, force: bool = False) -> List[NodeInfo]:
        """Discover available nodes in the network"""
        # Use cache if valid
        if not force and (time.time() - self._last_discovery) < self._discovery_cache_ttl:
            return list(self._nodes.values())
        
        try:
            response = await self._make_request_async("GET", "/discover")
            nodes_data = response.get("peers", [])
            
            nodes = [NodeInfo.from_dict(n) for n in nodes_data]
            
            # Update cache
            self._nodes = {n.id: n for n in nodes}
            self._last_discovery = time.time()
            
            return nodes
            
        except OpenLoopError as e:
            logger.warning(f"Node discovery failed: {e}")
            return list(self._nodes.values())
    
    def get_best_node(
        self,
        require_gpu: bool = False,
        task_type: Optional[TaskType] = None,
    ) -> Optional[NodeInfo]:
        """Get the best available node for a task"""
        nodes = list(self._nodes.values())
        
        if not nodes:
            return None
        
        # Filter by requirements
        if require_gpu:
            nodes = [n for n in nodes if n.has_gpu]
        
        if task_type:
            nodes = [n for n in nodes if n.supports(task_type)]
        
        # Filter by status
        nodes = [n for n in nodes if n.status == "online"]
        
        if not nodes:
            return None
        
        # Sort by score (descending)
        nodes.sort(key=lambda n: n.score, reverse=True)
        
        return nodes[0]
    
    async def get_network_stats(self) -> NetworkStats:
        """Get network statistics"""
        try:
            response = await self._make_request_async("GET", "/registry/stats")
            return NetworkStats(
                total_nodes=response.get("total_peers", 0),
                online_nodes=response.get("total_peers", 0),
                gpu_nodes=response.get("gpu_nodes", 0),
                total_tasks=response.get("total_tasks", 0),
            )
        except OpenLoopError:
            return NetworkStats()
    
    # === Task Execution ===
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        timeout: int = 60,
        priority: int = 0,
        require_gpu: bool = False,
    ) -> TaskResult:
        """
        Submit a task to the distributed network.
        
        Args:
            task_type: Type of task (e.g., "agent_infer")
            payload: Task payload (model, prompt, etc.)
            timeout: Timeout in seconds
            priority: Task priority
            require_gpu: Whether GPU is required
            
        Returns:
            TaskResult with status and output
        """
        start_time = time.time()
        
        # Try distributed execution first
        if self.base_url and self.base_url != "http://localhost:8080":
            try:
                result = await self._submit_distributed(
                    task_type, payload, timeout, priority
                )
                return result
            except OpenLoopError as e:
                logger.warning(f"Distributed execution failed: {e}")
                if not self.fallback_enabled:
                    return TaskResult(
                        task_id="",
                        status=TaskStatus.FAILED,
                        error=str(e),
                        latency_ms=int((time.time() - start_time) * 1000),
                    )
        
        # Fallback to local execution
        if self.fallback_enabled:
            return await self._execute_local(task_type, payload, timeout)
        
        return TaskResult(
            task_id="",
            status=TaskStatus.FAILED,
            error="No distributed execution available and local fallback disabled",
            latency_ms=int((time.time() - start_time) * 1000),
        )
    
    async def _submit_distributed(
        self,
        task_type: str,
        payload: Dict[str, Any],
        timeout: int,
        priority: int,
    ) -> TaskResult:
        """Submit task to distributed network"""
        task_data = {
            "type": task_type,
            "payload": payload,
            "timeout": timeout,
            "priority": priority,
        }
        
        response = await self._make_request_async(
            "POST", "/run", task_data, timeout=timeout + 10
        )
        
        return TaskResult(
            task_id=response.get("task_id", ""),
            status=TaskStatus.COMPLETED,
            result=response.get("result"),
            latency_ms=response.get("latency_ms", 0),
            cost=response.get("cost", 0),
            node_id=response.get("node_id"),
            steps=response.get("steps", []),
            metadata=response,
        )
    
    async def _execute_local(
        self,
        task_type: str,
        payload: Dict[str, Any],
        timeout: int,
    ) -> TaskResult:
        """Execute task locally"""
        start_time = time.time()
        
        try:
            if task_type == "agent_infer":
                # Import here to avoid circular imports
                from neuronmesh.agent import Agent
                from neuronmesh.memory import Memory
                
                model = payload.get("model", "llama3")
                prompt = payload.get("prompt", "")
                system_prompt = payload.get("system_prompt", "You are a helpful assistant.")
                
                agent = Agent(
                    name="local",
                    model=model,
                    memory=self._local_memory,
                )
                agent.config.system_prompt = system_prompt
                
                response = await agent.run_async(prompt)
                
                return TaskResult(
                    task_id=f"local_{int(time.time())}",
                    status=TaskStatus.COMPLETED,
                    result={"content": response.content},
                    latency_ms=int((time.time() - start_time) * 1000),
                    cost=0,  # Local is free
                    node_id=self.node_id,
                    steps=[{"step": "local_execution", "latency_ms": response.latency_ms}],
                )
            else:
                return TaskResult(
                    task_id="",
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task_type}",
                    latency_ms=int((time.time() - start_time) * 1000),
                )
                
        except Exception as e:
            return TaskResult(
                task_id="",
                status=TaskStatus.FAILED,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000),
            )
    
    # === Agent Execution ===
    
    async def run_agent(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        timeout: int = 60,
        require_gpu: bool = False,
    ) -> TaskResult:
        """
        Run an agent on the distributed network.
        
        Args:
            model: Model to use (e.g., "llama3", "gpt-4")
            prompt: User prompt
            system_prompt: System instructions
            timeout: Timeout in seconds
            require_gpu: Whether GPU is required
            
        Returns:
            TaskResult with agent response
        """
        return await self.submit_task(
            task_type="agent_infer",
            payload={
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
            },
            timeout=timeout,
            require_gpu=require_gpu,
        )
    
    # === Sync Versions ===
    
    def submit_task_sync(
        self,
        task_type: str,
        payload: Dict[str, Any],
        timeout: int = 60,
        priority: int = 0,
    ) -> TaskResult:
        """Synchronous version of submit_task"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.submit_task(task_type, payload, timeout, priority)
            )
        finally:
            loop.close()
    
    def run_agent_sync(
        self,
        model: str,
        prompt: str,
        **kwargs,
    ) -> TaskResult:
        """Synchronous version of run_agent"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.run_agent(model, prompt, **kwargs)
            )
        finally:
            loop.close()
    
    # === Status & Info ===
    
    async def get_status(self) -> Dict[str, Any]:
        """Get node status"""
        try:
            return await self._make_request_async("GET", "/status")
        except OpenLoopError as e:
            return {"status": "offline", "error": str(e)}
    
    def is_available(self) -> bool:
        """Check if OpenPool network is available"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                status = loop.run_until_complete(self.get_status())
                return status.get("status") != "offline"
            finally:
                loop.close()
        except Exception:
            return False
    
    def estimate_cost(
        self,
        task_type: str,
        payload: Dict[str, Any],
    ) -> float:
        """Estimate cost for a task"""
        base_costs = {
            "agent_infer": 1,
            "agent_train": 10,
            "agent_optimize": 5,
            "embedding": 0.1,
        }
        
        base = base_costs.get(task_type, 1)
        
        # Adjust for model
        model = payload.get("model", "")
        if "gpt-4" in model:
            base *= 10
        elif "gpt-3.5" in model:
            base *= 2
        
        return base
    
    def set_local_brain(self, brain):
        """Set local brain for fallback execution"""
        self._local_brain = brain
    
    def set_local_memory(self, memory):
        """Set local memory for fallback execution"""
        self._local_memory = memory


class OpenLoopError(Exception):
    """OpenLoop client error"""
    pass


# === Convenience Functions ===

def create_client(
    url: Optional[str] = None,
    **kwargs,
) -> OpenLoopClient:
    """
    Create an OpenLoop client.
    
    Example:
        client = create_client()  # Uses environment variables
        client = create_client(url="http://node2:8080")
    """
    return OpenLoopClient(base_url=url, **kwargs)


async def run_distributed(
    model: str,
    prompt: str,
    url: Optional[str] = None,
) -> TaskResult:
    """Quick distributed execution"""
    client = OpenLoopClient(base_url=url)
    return await client.run_agent(model, prompt)
