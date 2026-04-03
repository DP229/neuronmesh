"""
Example 3: Distributed Execution

Run agent tasks on the OpenPool network.

Prerequisites:
1. OpenPool node running (./node2 --http 8080)
2. Or set OPENPOOL_URL environment variable

Run:
    python examples/03_distributed.py
"""

import os
from neuronmesh import OpenLoopClient


def distributed_execution():
    """Execute tasks on distributed network"""
    print("=" * 60)
    print("Distributed Execution with OpenPool")
    print("=" * 60)
    
    # Create client
    client = OpenLoopClient()
    
    # Check network status
    print("\n1️⃣ Checking network status...")
    status = client.get_status()
    print(f"   Status: {status.get('status', 'unknown')}")
    print(f"   URL: {client.base_url}")
    print(f"   Credits: {client.get_credits()}")
    
    # Check available nodes
    print("\n2️⃣ Discovering nodes...")
    nodes = client.list_nodes()
    print(f"   Found {len(nodes)} nodes")
    
    if nodes:
        # Get best node
        best = client.get_best_node(require_gpu=False)
        if best:
            print(f"   Best node: {best.id[:8]}... (score: {best.score:.2f})")
    
    # Submit a task
    print("\n3️⃣ Submitting agent task...")
    result = client.run_agent(
        model="llama3",
        prompt="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
        timeout=30,
    )
    
    print(f"   Status: {result.status.value}")
    print(f"   Latency: {result.latency_ms}ms")
    
    if result.result:
        print(f"   Result: {result.result[:200]}...")
    
    if result.error:
        print(f"   Error: {result.error}")
    
    print("\n✅ Distributed execution complete!")


if __name__ == "__main__":
    distributed_execution()
