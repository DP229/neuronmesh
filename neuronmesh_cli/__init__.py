"""
NeuronMesh CLI - Command line interface
"""

import os
import sys
import json
import argparse
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuronmesh import Agent, Memory, Brain, OpenLoopClient


def main():
    parser = argparse.ArgumentParser(
        description="NeuronMesh - Distributed Intelligent Autoagent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # === Agent Commands ===
    
    agent_parser = subparsers.add_parser("agent", help="Run an agent")
    agent_parser.add_argument("prompt", help="Prompt for the agent")
    agent_parser.add_argument("--model", "-m", default="llama3", help="Model to use")
    agent_parser.add_argument("--memory", action="store_true", help="Enable memory")
    agent_parser.add_argument("--distributed", "-d", action="store_true", help="Use distributed execution")
    
    # === Memory Commands ===
    
    memory_parser = subparsers.add_parser("memory", help="Manage memory")
    memory_parser.add_argument("action", choices=["list", "add", "clear"], help="Action")
    memory_parser.add_argument("--content", help="Memory content (for add)")
    memory_parser.add_argument("--type", default="fact", help="Memory type")
    
    # === Network Commands ===
    
    network_parser = subparsers.add_parser("network", help="Manage network")
    network_parser.add_argument("action", choices=["status", "nodes", "submit"], help="Action")
    network_parser.add_argument("--task", help="Task type (for submit)")
    network_parser.add_argument("--payload", help="Task payload JSON (for submit)")
    
    # === Info Command ===
    
    subparsers.add_parser("info", help="Show NeuronMesh info")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == "agent":
        run_agent(args)
    elif args.command == "memory":
        manage_memory(args)
    elif args.command == "network":
        manage_network(args)
    elif args.command == "info":
        show_info()


def run_agent(args):
    """Run an agent with the given prompt"""
    memory = None
    
    if args.memory:
        memory = Memory()
        print("🧠 Memory enabled")
    
    agent = Agent(model=args.model)
    
    print(f"🤖 Running agent (model: {args.model})...")
    print(f"📝 Prompt: {args.prompt[:100]}...")
    
    if args.distributed:
        print("🔄 Using distributed execution")
        # TODO: Implement distributed mode
        print("⚠️  Distributed mode not yet implemented")
        return
    
    response = agent.run(args.prompt, memory=memory)
    
    print("\n" + "=" * 50)
    print("🤖 Response:")
    print("=" * 50)
    print(response.content)
    print("=" * 50)
    print(f"⏱️  Latency: {response.latency_ms}ms")
    print(f"💰 Cost: ${response.cost:.4f}")


def manage_memory(args):
    """Manage agent memory"""
    memory = Memory()
    
    if args.action == "list":
        stats = memory.get_stats()
        print("📊 Memory Statistics:")
        print(f"   Short-term: {stats['stm_size']}")
        print(f"   Working: {stats['working_size']}")
        print(f"   Long-term: {stats['ltm_size']}")
        print(f"   Total: {stats['total_memories']}")
    
    elif args.action == "add":
        if not args.content:
            print("❌ --content required for add action")
            return
        
        entry = memory.add(args.content, entry_type=args.type)
        print(f"✅ Added memory: {entry.id}")
    
    elif args.action == "clear":
        memory.clear()
        print("✅ Memory cleared")


def manage_network(args):
    """Manage OpenPool network"""
    client = OpenLoopClient()
    
    if args.action == "status":
        status = client.get_status()
        print("🌐 Network Status:")
        print(f"   URL: {client.base_url}")
        print(f"   Status: {status.get('status', 'unknown')}")
        print(f"   Credits: {client.get_credits()}")
    
    elif args.action == "nodes":
        nodes = client.list_nodes()
        print(f"📡 Found {len(nodes)} nodes:")
        for node in nodes:
            gpu = "🎮 GPU" if node.has_gpu else "💻 CPU"
            print(f"   {node.id[:8]}... | {gpu} | Score: {node.score:.2f}")
    
    elif args.action == "submit":
        if not args.task or not args.payload:
            print("❌ --task and --payload required for submit")
            return
        
        payload = json.loads(args.payload)
        print(f"📤 Submitting task: {args.task}")
        
        result = client.submit_task(args.task, payload)
        
        print(f"✅ Task completed:")
        print(f"   Status: {result.status.value}")
        print(f"   Latency: {result.latency_ms}ms")
        if result.error:
            print(f"   Error: {result.error}")


def show_info():
    """Show NeuronMesh information"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                     NeuronMesh v0.1.0                         ║
║         Distributed Intelligent Autoagent Platform             ║
╠══════════════════════════════════════════════════════════════╣
║                                                                ║
║  🧠  Intelligent Memory    - Persistent, searchable context   ║
║  🔄  Distributed Execution - Run on OpenPool network         ║
║  💰  Cost Efficient       - Use local + cloud models         ║
║  🔌  Extensible          - Add custom tools & integrations   ║
║                                                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                                ║
║  Commands:                                                     ║
║    neuronmesh agent "prompt"      Run an agent                  ║
║    neuronmesh memory list        List memories                 ║
║    neuronmesh network status     Check network                 ║
║                                                                ║
║  Documentation: https://neuronmesh.dev                         ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
