"""
NeuronMesh OpenLoop - Distributed Execution Client

Integrates with OpenPool network for distributed task execution.
"""

import os
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a distributed task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "node_id": self.node_id,
            "metadata": self.metadata,
        }


@dataclass  
class NodeInfo:
    """Information about a compute node"""
    id: str
    multiaddr: str
    cpu_cores: int = 0
    has_gpu: bool = False
    gpu_model: str = ""
    memory_gb: int = 0
    price: int = 0  # Credits per task
    score: float = 0.0
    status: str = "online"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(
            id=data.get("id", ""),
            multiaddr=data.get("multiaddr", ""),
            cpu_cores=data.get("cpu_cores", 0),
            has_gpu=data.get("has_gpu", False),
            gpu_model=data.get("gpu_model", ""),
            memory_gb=data.get("memory_gb", 0),
            price=data.get("price", 0),
            score=data.get("score", 0.0),
            status=data.get("status", "online"),
        )


class OpenLoopClient:
    """
    Client for OpenPool distributed compute network.
    
    Features:
    - Task submission and result retrieval
    - Node discovery
    - Credit management
    - Result verification
    
    Example:
        client = OpenLoopClient("http://localhost:8080")
        
        # Submit a task
        result = client.submit_task(
            task_type="agent_infer",
            payload={"model": "llama3", "task": "Hello"}
        )
        
        print(result.result)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("OPENPOOL_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("OPENPOOL_API_KEY", "")
        self.node_id = node_id or os.getenv("OPENPOOL_NODE_ID", "")
        
        self._session = urllib.request.urlopen
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise Exception(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Connection error: {e.reason}")
    
    # === Node Management ===
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status"""
        try:
            return self._make_request("GET", "/status")
        except Exception as e:
            logger.warning(f"Failed to get status: {e}")
            return {"status": "offline", "error": str(e)}
    
    def list_nodes(self) -> List[NodeInfo]:
        """List available nodes in the network"""
        try:
            data = self._make_request("GET", "/discover")
            nodes = data.get("peers", [])
            return [NodeInfo.from_dict(n) for n in nodes]
        except Exception as e:
            logger.warning(f"Failed to list nodes: {e}")
            return []
    
    def get_best_node(self, require_gpu: bool = False) -> Optional[NodeInfo]:
        """Get the best available node for a task"""
        nodes = self.list_nodes()
        
        if not nodes:
            return None
        
        # Filter by GPU requirement
        if require_gpu:
            nodes = [n for n in nodes if n.has_gpu]
        
        if not nodes:
            return None
        
        # Sort by score (descending)
        nodes.sort(key=lambda n: n.score, reverse=True)
        
        return nodes[0]
    
    # === Task Execution ===
    
    def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        timeout: int = 60,
        priority: int = 0,
    ) -> TaskResult:
        """
        Submit a task to the distributed network.
        
        Args:
            task_type: Type of task (e.g., "agent_infer", "agent_train")
            payload: Task payload (model, prompt, etc.)
            timeout: Timeout in seconds
            priority: Task priority (higher = more urgent)
            
        Returns:
            TaskResult with status and output
        """
        start_time = time.time()
        
        task_data = {
            "type": task_type,
            "payload": payload,
            "timeout": timeout,
            "priority": priority,
        }
        
        try:
            response = self._make_request("POST", "/run", task_data)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return TaskResult(
                task_id=response.get("task_id", ""),
                status=TaskStatus.COMPLETED,
                result=response.get("result"),
                latency_ms=latency_ms,
                cost=response.get("cost", 0),
                node_id=response.get("node_id"),
                metadata=response,
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            return TaskResult(
                task_id="",
                status=TaskStatus.FAILED,
                error=str(e),
                latency_ms=latency_ms,
            )
    
    def run_agent(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        timeout: int = 60,
    ) -> TaskResult:
        """
        Run an agent on the distributed network.
        
        Args:
            model: Model to use (e.g., "llama3", "gpt-4")
            prompt: User prompt
            system_prompt: System instructions
            timeout: Timeout in seconds
            
        Returns:
            TaskResult with agent response
        """
        return self.submit_task(
            task_type="agent_infer",
            payload={
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
            },
            timeout=timeout,
        )
    
    # === Credit Management ===
    
    def get_credits(self) -> int:
        """Get current credit balance"""
        try:
            data = self._make_request("GET", "/ledger")
            return data.get("balance", 0)
        except Exception:
            return 0
    
    def add_credits(self, amount: int) -> bool:
        """Add credits to account"""
        try:
            self._make_request("POST", "/ledger/add", {"amount": amount})
            return True
        except Exception:
            return False
    
    # === Agent Mode (Local + Distributed) ===
    
    def is_available(self) -> bool:
        """Check if OpenPool network is available"""
        try:
            status = self.get_status()
            return status.get("status") != "offline"
        except Exception:
            return False
    
    def estimate_cost(
        self,
        task_type: str,
        payload: Dict[str, Any],
    ) -> float:
        """Estimate cost for a task"""
        # Simple estimation based on task type
        base_costs = {
            "agent_infer": 1,
            "agent_train": 10,
            "agent_optimize": 5,
        }
        
        base = base_costs.get(task_type, 1)
        
        # Adjust for model
        model = payload.get("model", "")
        if "gpt-4" in model:
            base *= 10
        elif "gpt-3.5" in model:
            base *= 2
        
        return base


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
