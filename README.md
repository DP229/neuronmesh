# NeuronMesh

> Distributed intelligent autoagent platform. Run AI agents across your hardware network with intelligent memory.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What is NeuronMesh?

NeuronMesh lets you:
- **Build agents** with intelligent memory that persists across sessions
- **Run distributed** across your hardware (CPU/GPU nodes)
- **Save money** by using local inference instead of cloud APIs
- **Scale horizontally** by adding more nodes to your network

## Quick Start

```bash
# Install
pip install neuronmesh

# Create your first agent
from neuronmesh import Agent, Memory

agent = Agent(model="llama3")
memory = Memory()

response = agent.run("Hello, remember that I like coffee", memory=memory)
print(response)

# Later session - agent remembers
response = agent.run("What did I say I like?", memory=memory)
print(response)  # "You said you like coffee!"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   NeuronMesh                         │
│                                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │  Agent │◄──►│ Memory  │◄──►│  Brain  │         │
│  └─────────┘    └─────────┘    └─────────┘         │
│       │              │              │               │
│       └──────────────┴──────────────┘               │
│                      │                              │
│              ┌───────┴───────┐                     │
│              │  OpenLoop     │                     │
│              │  (Executor)   │                     │
│              └───────────────┘                     │
└─────────────────────────────────────────────────────┘
```

## Features

- 🧠 **Intelligent Memory** - Agents remember context, facts, and preferences
- 🔄 **Distributed Execution** - Run on any OpenPool node
- 💰 **Cost Efficient** - Use local inference, save on API costs
- 🔌 **Extensible** - Add custom tools and integrations
- 📊 **Observability** - Built-in metrics and tracing

## Installation

```bash
pip install neuronmesh
```

Or from source:

```bash
git clone https://github.com/dp229/neuronmesh.git
cd neuronmesh
pip install -e .
```

## Requirements

- Python 3.10+
- Redis (for memory layer)
- OpenPool node (for distributed execution)

## Documentation

Full documentation at: https://neuronmesh.dev

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)
