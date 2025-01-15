# ML Arena Pong 2024 Competition

Welcome to ML Arena Pong 2024, a multiplayer reinforcement learning competition where you'll develop intelligent agents to master the classic game of Pong with a competitive twist! 

## 🔗 Quick Links

- [Competition Page](https://ml-arena.com/viewcompetition/3)
- [PettingZoo Pong Environment](https://pettingzoo.farama.org/environments/atari/pong/)
- [ML Arena Platform](https://ml-arena.com)
- [Installation Guide](INSTALL.md)
- [Google Colab Starter](https://colab.research.google.com/drive/ml-arena-pong2024-starter)

## 🎮 Competition Overview

Compete against other AI agents in a multiplayer Pong environment where your agent needs to learn optimal strategies for both offensive and defensive play. The competition uses the PettingZoo Pong environment and features:

- 2-player sequential Pong
- ELO ranking system
- CPU, RAM, Memory constraints
- Support for PyTorch, JAX, and TensorFlow frameworks

## 📦 Repository Structure

```
ml-arena-pong2024/
├── src/
│   ├── pong/
│   │   ├── __init__.py
│   │   ├── environment.py    # Core Pong environment wrapper
│   │   ├── renderer.py       # Visualization utilities
│   │   └── types.py         # Type definitions
│   └── tools/
│       ├── __init__.py
│       ├── evaluate.py      # Local evaluation tools
│       └── replay.py        # Replay visualization
├── kit/                    # Python starter kit
│   ├── main.py
│   ├── agent.py
│   └── requirements.txt
├── examples/
│   ├── random_agent.py
│   ├── rule_based_agent.py
│   └── simple_dqn_agent.py
└── tests/
    ├── __init__.py
    ├── test_environment.py
    └── test_agents.py
```

## 🚀 Getting Started

1. Choose your development environment:
   - [Local Installation Guide](INSTALL.md)
   - [Google Colab Notebook](https://colab.research.google.com/drive/ml-arena-pong2024-starter)

2. Check out the [Python Starter Kit](kit/README.md) for implementation examples

3. Test your agent using the local evaluation tools (see [Environment Guide](docs/environment.md))

## 📝 Competition Rules

### Match Format
- 3000 steps match
- Points scored through successful paddle hits and opponent misses

### Agent Requirements
- Must respond within 50ms per step
- Maximum 1GB memory usage
- Supports partially observable state
- Must be compatible with runtime provided

### Evaluation Criteria
1. Win/loss/draw elo in tournament matches

## 📅 Important Dates

- **Competition Start**: January 22, 2025
- **Final Submission Deadline**: April 1, 2025

## 📖 Documentation

- [Environment Guide](docs/environment.md)
- [Agent Development](docs/agent.md)
- [Submission Guide](docs/submission.md)

## 💬 Community

- [GitHub Discussions](https://github.com/ml-arena/pong2024/discussions)

## 📊 Leaderboard

Current leaderboard available at: [ML Arena Pong 2024 Leaderboard](https://ml-arena.com/viewcompetition/3)

## 🤝 Contributing

Contributions are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md).

## 🙏 Acknowledgments

- Competition infrastructure powered by ML Arena
- Environment provided by PettingZoo
- Special thanks to our community contributors