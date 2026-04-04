# NeuronMesh 🤖

**Distributed Intelligent Autoagent Platform**

Build AI agents that remember, collaborate, and scale across your hardware network.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/dp229/neuronmesh)](https://github.com/dp229/neuronmesh/stargazers)

## What is NeuronMesh?

NeuronMesh is an open-source platform for building intelligent AI agents with:

- 🧠 **Intelligent Memory** - Agents that remember context across sessions
- 🔧 **Claude Code-Inspired Tools** - Bash, file operations, search, web
- 🤖 **Multi-Agent Orchestration** - Sequential, parallel, hierarchical patterns
- 🔄 **OpenPool Integration** - Distributed execution across your hardware
- 💰 **Cost Optimization** - Smart model routing, caching, free local models

## Quick Start

```bash
# Install
pip install neuronmesh

# Or from source
git clone https://github.com/dp229/neuronmesh.git
cd neuronmesh
pip install -e .

# Start Ollama (for local inference)
ollama serve
ollama pull llama3

# Run example
python examples/00_welcome.py
```

## Features

### 🤖 Intelligent Agents

```python
from neuronmesh import Agent, Memory

# Create agent with memory
memory = Memory()
agent = Agent(model="llama3", memory=memory)

# Run - agent remembers context
response = agent.run("I'm working on a Python project")
response = agent.run("What am I working on?")  # Remembers!
```

### 🔧 Tools

```python
from neuronmesh import create_coder_agent

coder = create_coder_agent()
response = coder.run("List all Python files in this project")
```

### 🤝 Multi-Agent

```python
from neuronmesh import Orchestrator, AgentSpec

orchestrator = Orchestrator()

agents = [
    AgentSpec("researcher", "researcher", "Research thoroughly."),
    AgentSpec("writer", "writer", "Write clearly."),
]

result = orchestrator.sync_sequential(agents, "What is AI?")
```

### 🔄 Distributed

```python
from neuronmesh import OpenLoopClient

client = OpenLoopClient()
result = await client.run_agent("llama3", "Hello")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       NeuronMesh                             │
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│  │  Agent  │◄──►│ Memory  │◄──►│  Brain  │                │
│  │         │    │  (RAG)  │    │ (LLMs) │                │
│  └────┬────┘    └─────────┘    └────┬────┘                │
│       │                              │                      │
│       └──────────┬─────────────────┘                      │
│                  │                                        │
│          ┌──────┴──────┐                                │
│          │ Orchestrator │                                │
│          └─────────────┘                                │
│                  │                                        │
│          ┌──────┴──────┐                                │
│          │  OpenLoop   │                                │
│          │ (Network)  │                                │
│          └─────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

## Examples

| Example | Description |
|---------|-------------|
| [00_welcome.py](examples/00_welcome.py) | Feature showcase |
| [01_quickstart.py](examples/01_quickstart.py) | Basic agent + memory |
| [02_multi_agent.py](examples/02_multi_agent.py) | Multiple agents |
| [03_distributed.py](examples/03_distributed.py) | OpenPool integration |
| [04_tools.py](examples/04_tools.py) | Tool system |
| [05_memory_rag.py](examples/05_memory_rag.py) | RAG memory |
| [06_orchestrator.py](examples/06_orchestrator.py) | Multi-agent |

## CLI

```bash
# Agent
neuronmesh agent run "What is Python?"
neuronmesh agent chat

# Memory
neuronmesh memory add "Important fact" --type fact
neuronmesh memory search "fact"

# Models
neuronmesh model list
neuronmesh model info llama3

# Network
neuronmesh network status
neuronmesh network nodes

# Configuration
neuronmesh config show
neuronmesh config set model gpt-4
```

## Supported Models

### Free (Local)

- **Ollama**: llama3, llama3.1, llama3.2, codellama, mistral, mixtral, qwen2.5, gemma2

### API (Paid)

- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: claude-sonnet-4, claude-3.5-sonnet, claude-3.5-haiku
- **OpenRouter**: deepseek-r1, various models
- **Groq**: llama-3.3-70b, mixtral-8x7b

## Documentation

- [Quick Start](docs/QUICKSTART.md) - Get started in 5 minutes
- [API Reference](docs/API.md) - Complete API documentation
- [Implementation Plan](PLAN.md) - 12-week development roadmap

## Tech Stack

- **Python 3.10+** - Core language
- **SQLite** - Memory persistence
- **Sentence-Transformers** - Embeddings (optional)
- **OpenAI/Anthropic SDKs** - Cloud LLM providers
- **Ollama** - Local inference

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

See [PLAN.md](PLAN.md) for the 12-week implementation plan:

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Agent + Tools + Brain | ✅ |
| 2 | Memory + RAG | ✅ |
| 3 | OpenLoop + Orchestrator | ✅ |
| 4 | CLI + Docs | 🔄 |
| 5-12 | Production features | ⏳ |

## License

MIT License - see [LICENSE](LICENSE)

## Links

- [GitHub](https://github.com/dp229/neuronmesh)
- [Documentation](docs/)
- [Plan](PLAN.md)

---

Built with ❤️ for the AI agent community
