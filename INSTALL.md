# Installation Guide

There are several ways to set up your development environment for the ML Arena Pong 2024 competition:

1. [Docker Installation](#docker-installation) (Recommended)
2. [Local Installation](#local-installation)
3. [Google Colab](#google-colab)

## Docker Installation (Recommended)

The Docker setup provides a consistent environment that matches the competition runtime.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/ml-arena/pong2024.git
cd pong2024
```

2. Build and run using Docker Compose:
```bash
docker-compose up --build
```

This will:
- Set up the PettingZoo environment with all dependencies
- Install required Python packages
- Configure the Atari ROM
- Mount your local directory for development

### Verify Docker Installation

```bash
# Run a test match
docker-compose run --rm ml-arena-pong python examples/random_agent.py examples/random_agent.py
```

## Local Installation

### Prerequisites
- Python 3.9
- pip or conda package manager
- C++ build tools (for PettingZoo installation)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update && sudo apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    swig \
    python3-pygame \
    libsdl2-dev \
    libjpeg-dev \
    zlib1g-dev
```

#### macOS
```bash
brew install python@3.9 swig sdl2
```

#### Windows
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install [SWIG](http://www.swig.org/download.html)
3. Add SWIG to your system PATH

### Python Environment Setup

1. Create a virtual environment:
```bash
# Using venv
python3.9 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Or using conda
conda create -n ml-arena-pong python=3.9
conda activate ml-arena-pong
```

2. Install Python dependencies:
```bash
# Install Poetry
pip install poetry

# Install project dependencies
poetry install

# Install PettingZoo with Atari support
poetry add "pettingzoo[classic,atari]==1.24.3"

# Install AutoROM and accept license
poetry run AutoROM --accept-license
```

### Verify Local Installation

```bash
# Run a test match
python examples/random_agent.py examples/random_agent.py

# Run the visualization tool
python -m ml_arena_pong.vis replay.json
```

## Google Colab

For those who prefer a cloud-based setup or have installation issues, we provide a Google Colab notebook:

[ML Arena Pong 2024 Starter Notebook](https://colab.research.google.com/drive/ml-arena-pong2024-starter)

The notebook includes:
- Pre-installed dependencies
- Example agent implementation
- Training and evaluation code
- Visualization tools

Note: While Colab is great for development and testing, you'll need to ensure your agent works in the competition runtime environment before submission.

## Troubleshooting

### Common Issues

1. **PettingZoo Installation Fails**
   - Ensure you have the required system dependencies
   - Try installing with `pip install --no-cache-dir "pettingzoo[atari]"`

2. **ROM Loading Error**
   - Run `AutoROM --accept-license` to download required ROMs
   - Check ROM path environment variable: `export ROM_PATH=/path/to/roms`

3. **SDL2 Error**
   - Install SDL2 development libraries for your system
   - For Ubuntu: `sudo apt-get install libsdl2-dev`

4. **Memory/GPU Issues**
   - Try running with CPU only: `export CUDA_VISIBLE_DEVICES=`
   - Reduce batch size in training scripts

For additional help:
- Open an issue on GitHub