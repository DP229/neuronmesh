"""
NeuronMesh CLI - Enhanced Command Line Interface

Provides a complete CLI for NeuronMesh:
- Agent creation and execution
- Memory management
- Network operations
- Configuration management
- Interactive mode

Run:
    neuronmesh --help
"""

import os
import sys
import json
import time
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuronmesh import (
    Agent, Memory, Brain,
    Orchestrator, AgentSpec,
    OpenLoopClient,
    create_agent, create_coder_agent,
    ModelRegistry, ModelProvider,
)


@dataclass
class CLIConfig:
    """CLI configuration"""
    model: str = "llama3"
    memory_enabled: bool = True
    tools_enabled: bool = True
    verbose: bool = False
    json_output: bool = False
    color: bool = True


class NeuronMeshCLI:
    """NeuronMesh Command Line Interface"""
    
    def __init__(self, config: Optional[CLIConfig] = None):
        self.config = config or CLIConfig()
        self.memory = None
        self.agent = None
        self.brain = Brain()
    
    # === Agent Commands ===
    
    def cmd_agent_create(self, name: str, model: str = None, 
                        instructions: str = None) -> Dict[str, Any]:
        """Create a new agent"""
        model = model or self.config.model
        instructions = instructions or "You are a helpful AI assistant."
        
        self.agent = create_agent(
            name=name,
            model=model,
            instructions=instructions,
            memory=self.config.memory_enabled,
            tools=self.config.tools_enabled,
        )
        
        if not self.config.json_output:
            print(f"✅ Created agent: {name}")
            print(f"   Model: {model}")
            print(f"   Memory: {'enabled' if self.config.memory_enabled else 'disabled'}")
        
        return {"name": name, "model": model, "status": "created"}
    
    def cmd_agent_run(self, prompt: str) -> Dict[str, Any]:
        """Run agent with prompt"""
        if not self.agent:
            self.agent = create_agent(
                model=self.config.model,
                memory=self.config.memory_enabled,
            )
        
        if self.config.verbose:
            print(f"📝 Prompt: {prompt[:100]}...")
        
        start = time.time()
        response = self.agent.run(prompt)
        duration = int((time.time() - start) * 1000)
        
        if self.config.json_output:
            return {
                "content": response.content,
                "latency_ms": duration,
                "cost": response.cost,
                "turns": len(response.turns),
            }
        
        print("\n" + "=" * 60)
        print("🤖 Response:")
        print("=" * 60)
        print(response.content)
        print("=" * 60)
        print(f"⏱️  Latency: {duration}ms | 💰 Cost: ${response.cost:.4f}")
        
        return {"content": response.content, "latency_ms": duration}
    
    def cmd_agent_chat(self):
        """Interactive chat mode"""
        print("\n🎭 Entering chat mode. Type 'exit' to quit.\n")
        
        if not self.agent:
            self.agent = create_agent(
                model=self.config.model,
                memory=self.config.memory_enabled,
            )
        
        while True:
            try:
                prompt = input("👤 You: ")
                if prompt.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt.strip():
                    continue
                
                response = self.agent.run(prompt)
                print(f"\n🤖 Agent: {response.content}\n")
                
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
    
    # === Memory Commands ===
    
    def cmd_memory_list(self) -> Dict[str, Any]:
        """List memory statistics"""
        if not self.memory:
            self.memory = Memory()
        
        stats = self.memory.get_stats()
        
        if self.config.json_output:
            return stats
        
        print("\n📊 Memory Statistics:")
        print(f"   Short-term: {stats['stm_size']}")
        print(f"   Working: {stats['working_size']}")
        print(f"   Long-term: {stats['ltm_size']}")
        print(f"   Total: {stats['total_memories']}")
        
        return stats
    
    def cmd_memory_add(self, content: str, entry_type: str = "fact",
                       importance: float = 0.5) -> Dict[str, Any]:
        """Add memory"""
        if not self.memory:
            self.memory = Memory()
        
        entry = self.memory.add(
            content=content,
            entry_type=entry_type,
            importance=importance,
        )
        
        if not self.config.json_output:
            print(f"✅ Added memory: {entry.id}")
            print(f"   Type: {entry_type}")
            print(f"   Importance: {importance}")
        
        return {"id": entry.id, "status": "added"}
    
    def cmd_memory_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search memory"""
        if not self.memory:
            self.memory = Memory()
        
        results = self.memory.retrieve(query, limit=limit)
        
        if self.config.json_output:
            return {"results": [r.to_dict() for r in results]}
        
        if not results:
            print("❌ No results found")
            return {"results": []}
        
        print(f"\n🔍 Results for: '{query}'")
        for i, entry in enumerate(results, 1):
            print(f"\n{i}. [{entry.entry_type}] {entry.content[:100]}...")
            print(f"   Importance: {'⭐' * int(entry.importance * 5)}")
        
        return {"results": [r.content for r in results]}
    
    def cmd_memory_clear(self) -> Dict[str, Any]:
        """Clear all memories"""
        if not self.memory:
            self.memory = Memory()
        
        count = self.memory.get_stats()['total_memories']
        self.memory.clear()
        
        if not self.config.json_output:
            print(f"✅ Cleared {count} memories")
        
        return {"cleared": count, "status": "cleared"}
    
    # === Network Commands ===
    
    def cmd_network_status(self) -> Dict[str, Any]:
        """Check network status"""
        client = OpenLoopClient()
        
        status = client.get_status()
        stats = client.get_network_stats()
        
        if self.config.json_output:
            return {"status": status, "stats": stats}
        
        print("\n🌐 Network Status:")
        print(f"   URL: {client.base_url}")
        print(f"   Status: {status.get('status', 'unknown')}")
        print(f"   Credits: {client.get_credits()}")
        print(f"   Nodes: {stats.total_nodes}")
        
        return {"status": status, "stats": stats}
    
    def cmd_network_nodes(self) -> Dict[str, Any]:
        """List available nodes"""
        client = OpenLoopClient()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nodes = loop.run_until_complete(client.discover_nodes())
        loop.close()
        
        if self.config.json_output:
            return {"nodes": [n.__dict__ for n in nodes]}
        
        if not nodes:
            print("❌ No nodes found")
            return {"nodes": []}
        
        print(f"\n📡 Found {len(nodes)} nodes:")
        for node in nodes:
            gpu = "🎮 GPU" if node.has_gpu else "💻 CPU"
            print(f"   {node.id[:12]}... | {gpu} | Score: {node.score:.2f} | {node.status}")
        
        return {"nodes": [n.id for n in nodes]}
    
    # === Model Commands ===
    
    def cmd_model_list(self, provider: str = None) -> Dict[str, Any]:
        """List available models"""
        registry = self.brain.registry
        
        if provider:
            prov = ModelProvider(provider)
            models = registry.list(provider=prov)
        else:
            models = registry.list()
        
        if self.config.json_output:
            return {"models": [m.name for m in models]}
        
        print("\n🤖 Available Models:")
        for m in models:
            cost = f"${m.cost_per_1k:.4f}/1K" if m.cost_per_1k > 0 else "FREE"
            print(f"   {m.name:<30} [{m.provider.value:<10}] {cost}")
        
        return {"models": [m.name for m in models]}
    
    def cmd_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        model = self.brain.registry.get(model_name)
        
        if not model:
            return {"error": f"Model not found: {model_name}"}
        
        if self.config.json_output:
            return {
                "name": model.name,
                "provider": model.provider.value,
                "context_length": model.context_length,
                "cost_per_1k": model.cost_per_1k,
                "capabilities": model.capabilities,
                "description": model.description,
            }
        
        print(f"\n📋 Model: {model.name}")
        print(f"   Provider: {model.provider.value}")
        print(f"   Context: {model.context_length} tokens")
        print(f"   Cost: ${model.cost_per_1k:.4f}/1K tokens")
        print(f"   Capabilities: {', '.join(model.capabilities)}")
        print(f"   Description: {model.description}")
        
        return {"name": model.name}
    
    # === Orchestrator Commands ===
    
    def cmd_orchestrate(self, pattern: str, task: str) -> Dict[str, Any]:
        """Run orchestration pattern"""
        orchestrator = Orchestrator()
        
        if pattern == "sequential":
            agents = [
                AgentSpec("researcher", "researcher", "Research and find key information."),
                AgentSpec("writer", "writer", "Write a clear summary based on research."),
            ]
            result = orchestrator.sync_sequential(agents, task)
        
        elif pattern == "parallel":
            agents = [
                AgentSpec("analyst1", "analyst", "Analyze from perspective A."),
                AgentSpec("analyst2", "analyst", "Analyze from perspective B."),
            ]
            result = orchestrator.sync_parallel(agents, task)
        
        else:
            return {"error": f"Unknown pattern: {pattern}"}
        
        if self.config.json_output:
            return result.to_dict()
        
        print(f"\n✅ Orchestration complete ({pattern})")
        print(f"   Duration: {result.duration_ms}ms")
        print(f"   Agents: {len(result.agent_results)}")
        
        for name, output in result.outputs.items():
            if name not in ['aggregate']:
                print(f"\n📄 {name}:")
                print(f"   {output[:200]}...")
        
        return {"pattern": pattern, "duration_ms": result.duration_ms}
    
    # === Config Commands ===
    
    def cmd_config_show(self) -> Dict[str, Any]:
        """Show current configuration"""
        config = {
            "model": self.config.model,
            "memory_enabled": self.config.memory_enabled,
            "tools_enabled": self.config.tools_enabled,
            "verbose": self.config.verbose,
            "json_output": self.config.json_output,
        }
        
        if self.config.json_output:
            return config
        
        print("\n⚙️  Current Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        return config
    
    def cmd_config_set(self, key: str, value: str) -> Dict[str, Any]:
        """Set configuration value"""
        if key == "model":
            self.config.model = value
        elif key == "memory":
            self.config.memory_enabled = value.lower() == "true"
        elif key == "tools":
            self.config.tools_enabled = value.lower() == "true"
        elif key == "verbose":
            self.config.verbose = value.lower() == "true"
        elif key == "json":
            self.config.json_output = value.lower() == "true"
        else:
            return {"error": f"Unknown config key: {key}"}
        
        return {"key": key, "value": value, "status": "set"}


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NeuronMesh - Distributed Intelligent Autoagent Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global options
    parser.add_argument("--model", "-m", default="llama3", help="Default model")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    parser.add_argument("--no-tools", action="store_true", help="Disable tools")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Agent commands
    agent_parser = subparsers.add_parser("agent", help="Agent operations")
    agent_sub = agent_parser.add_subparsers(dest="action")
    
    agent_create = agent_sub.add_parser("create", help="Create agent")
    agent_create.add_argument("name", help="Agent name")
    agent_create.add_argument("--model", "-m", help="Model name")
    agent_create.add_argument("--instructions", "-i", help="System instructions")
    
    agent_run = agent_sub.add_parser("run", help="Run agent")
    agent_run.add_argument("prompt", help="Prompt text")
    
    agent_sub.add_parser("chat", help="Interactive chat")
    
    # Memory commands
    memory_parser = subparsers.add_parser("memory", help="Memory operations")
    memory_sub = memory_parser.add_subparsers(dest="action")
    
    memory_sub.add_parser("list", help="List memory stats")
    
    memory_add = memory_sub.add_parser("add", help="Add memory")
    memory_add.add_argument("content", help="Memory content")
    memory_add.add_argument("--type", "-t", default="fact", help="Memory type")
    memory_add.add_argument("--importance", "-i", type=float, default=0.5, help="Importance 0-1")
    
    memory_search = memory_sub.add_parser("search", help="Search memory")
    memory_search.add_argument("query", help="Search query")
    memory_search.add_argument("--limit", "-l", type=int, default=5, help="Result limit")
    
    memory_sub.add_parser("clear", help="Clear all memories")
    
    # Network commands
    network_parser = subparsers.add_parser("network", help="Network operations")
    network_sub = network_parser.add_subparsers(dest="action")
    
    network_sub.add_parser("status", help="Check network status")
    network_sub.add_parser("nodes", help="List nodes")
    
    # Model commands
    model_parser = subparsers.add_parser("model", help="Model operations")
    model_sub = model_parser.add_subparsers(dest="action")
    
    model_list = model_sub.add_parser("list", help="List models")
    model_list.add_argument("--provider", "-p", help="Filter by provider")
    
    model_info = model_sub.add_parser("info", help="Model info")
    model_info.add_argument("name", help="Model name")
    
    # Orchestrator commands
    orch_parser = subparsers.add_parser("orchestrate", help="Multi-agent orchestration")
    orch_parser.add_argument("pattern", choices=["sequential", "parallel"], help="Pattern")
    orch_parser.add_argument("task", help="Task description")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration")
    config_sub = config_parser.add_subparsers(dest="action")
    
    config_sub.add_parser("show", help="Show config")
    
    config_set = config_sub.add_parser("set", help="Set config")
    config_set.add_argument("key", help="Config key")
    config_set.add_argument("value", help="Config value")
    
    # Info command
    subparsers.add_parser("info", help="Show NeuronMesh info")
    
    args = parser.parse_args()
    
    # Create CLI
    config = CLIConfig(
        model=args.model,
        memory_enabled=not args.no_memory,
        tools_enabled=not args.no_tools,
        verbose=args.verbose,
        json_output=args.json,
        color=not args.no_color,
    )
    cli = NeuronMeshCLI(config)
    
    # Execute command
    result = None
    
    if args.command is None:
        parser.print_help()
        return
    
    elif args.command == "agent":
        if args.action == "create":
            result = cli.cmd_agent_create(args.name, args.model, args.instructions)
        elif args.action == "run":
            result = cli.cmd_agent_run(args.prompt)
        elif args.action == "chat":
            cli.cmd_agent_chat()
    
    elif args.command == "memory":
        if args.action == "list":
            result = cli.cmd_memory_list()
        elif args.action == "add":
            result = cli.cmd_memory_add(args.content, args.type, args.importance)
        elif args.action == "search":
            result = cli.cmd_memory_search(args.query, args.limit)
        elif args.action == "clear":
            result = cli.cmd_memory_clear()
    
    elif args.command == "network":
        if args.action == "status":
            result = cli.cmd_network_status()
        elif args.action == "nodes":
            result = cli.cmd_network_nodes()
    
    elif args.command == "model":
        if args.action == "list":
            result = cli.cmd_model_list(args.provider)
        elif args.action == "info":
            result = cli.cmd_model_info(args.name)
    
    elif args.command == "orchestrate":
        result = cli.cmd_orchestrate(args.pattern, args.task)
    
    elif args.command == "config":
        if args.action == "show":
            result = cli.cmd_config_show()
        elif args.action == "set":
            result = cli.cmd_config_set(args.key, args.value)
    
    elif args.command == "info":
        print("""
╔══════════════════════════════════════════════════════════════╗
║              NeuronMesh v0.1.0                             ║
║       Distributed Intelligent Autoagent Platform               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🧠  Intelligent Memory     - RAG, semantic search        ║
║  🔧  Claude Code Tools      - Bash, files, search, web     ║
║  🤖  Multi-Agent           - Sequential, parallel, hierarchical║
║  🔄  OpenPool Integration   - Distributed execution         ║
║  💰  Cost Optimization      - Smart model routing           ║
║                                                              ║
║  Docs: https://github.com/dp229/neuronmesh                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    # Print JSON result if enabled
    if result and args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
