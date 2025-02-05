# Installation Guide

There are several ways to set up your development environment for the ML Arena Pong 2024 competition:

1. [Docker Installation](#docker-installation) (comming soon)
2. [Local Installation](#local-installation)
3. [Google Colab](#google-colab)

## Docker Installation

(comming soon)

## Local Installation

### Prerequisites
- Python 3.10
- pip or conda package manager
- C++ build tools (for PettingZoo installation)

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update && sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    swig \
    python3-pygame \
    libsdl2-dev \
    libjpeg-dev \
    zlib1g-dev
```

#### macOS
```bash
brew install python@3.10 swig sdl2
```

#### Windows
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install [SWIG](http://www.swig.org/download.html)
3. Add SWIG to your system PATH

### Python Environment Setup

1. Create a virtual environment:
```bash
# Using venv
python3.10 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

```

2. Install Python dependencies:
```bash
# Install Python packages
pip install "pettingzoo[atari]==1.24.3" gymnasium[atari] numpy pygame

# Install and setup AutoROM
pip install autorom
AutoROM --accept-license
```

3. Install pong2024 pkg:
```bash
# Install Poetry
pip install git+https://github.com/ml-arena/pong2024.git

```

## Google Colab

For those who prefer a cloud-based setup or have installation issues, we provide a Google Colab notebook:

[ML Arena Pong 2024 Starter Notebook](https://colab.research.google.com/github/ml-arena/pong2024/blob/main/notebook/getting_started.ipynb)

The notebook includes:
- Pre-installed dependencies
- Example agent implementation
- Training and evaluation code
- Visualization tools