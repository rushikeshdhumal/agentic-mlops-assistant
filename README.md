# ATHENA MLOps Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A**daptive **T**raining & **H**yperparameter **E**xploration **N**atural **A**ssistant

> Agentic MLOps assistant with natural language interface for edge AI optimization. Enables ML engineers to query experiments, orchestrate training runs, and get optimization recommendations through conversational AI.

## Overview

ATHENA is an intelligent MLOps platform that combines:
- **Agentic AI**: LangChain/LangGraph-powered agent with ReAct pattern for autonomous task execution
- **Natural Language Interface**: Query experiments, launch training, and get insights through conversation
- **MLOps Automation**: Integrated experiment tracking (MLflow), model management, and training orchestration
- **Edge AI Focus**: Optimization recommendations for edge deployment (quantization, pruning, latency analysis)
- **Semantic Search**: Vector-based search over experiments and documentation using embeddings

## Key Features

### Natural Language Querying
```
"What were my best performing models last week?"
"Compare YOLOv8 variants on accuracy vs latency"
"Show experiments with >90% accuracy and <50ms inference time"
```

### Intelligent Training Orchestration
```
"Train ResNet18 on CIFAR-10 with lr=0.001, batch_size=64"
"Run hyperparameter sweep for learning rates [0.0001, 0.001, 0.01]"
"Resume training from checkpoint_epoch_10"
```

### Edge AI Optimization
```
"How can I improve inference speed by 2x?"
"Recommend quantization strategy for mobile deployment"
"Optimize this model for Jetson Nano with <10ms latency"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│           (Streamlit Chat + Visualization Panels)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Agentic Orchestration Layer                     │
│  (Intent Classifier → ReAct Agent → Tool Selection)         │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│               Tool Integration Layer                         │
│  Query │ Training │ Analysis │ Optimization │ Visualization │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  MLOps Backend Layer                         │
│        MLflow │ Model Registry │ Training Scripts           │
└─────────────────────────────────────────────────────────────┘
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Agentic Framework** | LangChain + LangGraph (ReAct pattern) |
| **LLM Backend** | Ollama (Llama 3.1 8B - local, zero-cost) |
| **Experiment Tracking** | MLflow with SQLite + local artifacts |
| **ML Framework** | PyTorch |
| **Vector Store** | Chroma with sentence-transformers |
| **UI Framework** | Streamlit |
| **API Layer** | FastAPI |
| **Database** | SQLite |
| **Testing** | Pytest (>85% coverage target) |
| **CI/CD** | GitHub Actions |

## Project Structure

```
agentic-mlops-assistant/
├── src/athena/              # Main application code
│   ├── agent/               # Agentic orchestration (intent, planner, memory)
│   ├── tools/               # Tool implementations (query, training, analysis)
│   ├── mlops/               # MLOps backend (MLflow, training scripts)
│   ├── ui/                  # Streamlit interface components
│   ├── api/                 # FastAPI endpoints
│   ├── config/              # Configuration and storage management
│   └── utils/               # Shared utilities (logging, metrics)
├── tests/                   # Unit and integration tests
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── scripts/                 # Training and utility scripts
├── docs/                    # Documentation (MkDocs)
├── configs/                 # Configuration files
├── notebooks/               # Jupyter notebooks for exploration
├── .github/workflows/       # CI/CD pipelines
├── data/                    # Symlink to external storage datasets
├── mlruns/                  # Symlink to external storage MLflow artifacts
├── models/                  # Symlink to external storage model checkpoints
└── local_storage/           # Local storage (<10 GB)
    ├── sqlite_db/           # SQLite databases
    ├── vector_db/           # Chroma vector store
    └── logs/                # Application logs
```

## External Storage Configuration

ATHENA intelligently manages storage by detecting the **DHUMAL** external drive and storing large files there:

**External Storage (~13 GB)**:
- Ollama models (~4.7 GB)
- Training datasets (~2 GB)
- MLflow artifacts (~2 GB)
- Model checkpoints (~3 GB)
- Cache files (~2 GB)

**Local Storage (<10 GB)**:
- Code and dependencies
- SQLite databases
- Vector store (Chroma)
- Configuration files

The system automatically detects the DHUMAL drive on:
- **Windows**: `D:/`, `E:/`, `F:/`, etc.
- **macOS**: `/Volumes/DHUMAL`
- **Linux**: `/mnt/dhumal`

## Quick Start

### Prerequisites

- Python 3.9 or higher
- 16 GB RAM recommended
- External drive (DHUMAL) for large file storage (optional but recommended)
- Ollama installed ([ollama.ai](https://ollama.ai))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rushikeshdhumal/agentic-mlops-assistant.git
   cd agentic-mlops-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Production dependencies
   pip install -r requirements.txt

   # Development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

4. **Install Ollama and download model**
   ```bash
   # Install Ollama from https://ollama.ai

   # Pull Llama 3.1 8B model
   ollama pull llama3.1:8b
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (optional - defaults work out of the box)
   ```

6. **Initialize storage**
   ```bash
   python -m athena --check-storage
   ```
   This will:
   - Detect the DHUMAL external drive (if available)
   - Create necessary directories
   - Create symlinks to external storage

7. **Install pre-commit hooks** (for developers)
   ```bash
   pre-commit install
   ```

### Running Sample Training

Run baseline experiments to populate MLflow:

```bash
# Train on MNIST
python scripts/train_mnist.py --epochs 5

# Train on CIFAR-10
python scripts/train_cifar10.py --epochs 10 --model cifar10_cnn

# Train ResNet variant
python scripts/train_cifar10.py --epochs 15 --model simple_resnet --lr 0.01
```

### Launching ATHENA

```bash
# Launch Streamlit UI (when implemented)
python -m athena --mode ui

# Launch FastAPI server (when implemented)
python -m athena --mode api

# Interactive CLI (when implemented)
python -m athena --mode cli
```

## Development Roadmap

### Phase 1: Foundation & Infrastructure ✅ (Complete)
- [x] Project structure setup
- [x] External storage detection (DHUMAL drive)
- [x] MLflow integration
- [x] Sample training scripts (MNIST/CIFAR-10)
- [x] Vector store setup (Chroma)
- [x] Pre-commit hooks and code quality tools
- [x] GitHub Actions CI/CD

### Phase 2: Core Agentic System (In Progress)
- [ ] Ollama integration with Llama 3.1 8B
- [ ] MLflow query tools
- [ ] Intent classification system
- [ ] ReAct agent with LangChain/LangGraph
- [ ] Conversation manager with context tracking
- [ ] Tool execution framework
- [ ] Semantic search implementation
- [ ] Memory systems

### Phase 3: Analysis & Orchestration
- [ ] Training orchestration tools
- [ ] Analysis and visualization tools
- [ ] Recommendation engine for edge optimization
- [ ] Report generation
- [ ] Anomaly detection

### Phase 4: User Interface & Experience
- [ ] Streamlit chat interface
- [ ] Visualization panels
- [ ] Example query templates
- [ ] Export functionality
- [ ] Interactive dashboards

### Phase 5: Production Readiness
- [ ] Comprehensive testing (>85% coverage)
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Docker configuration
- [ ] Demo video and presentation materials

## Configuration

### Storage Configuration

Edit `config/storage.py` or set environment variables:

```python
EXTERNAL_STORAGE_PATH=D:/  # Windows
EXTERNAL_STORAGE_PATH=/Volumes/DHUMAL  # macOS
EXTERNAL_STORAGE_PATH=/mnt/dhumal  # Linux
```

### MLflow Configuration

```bash
MLFLOW_TRACKING_URI=sqlite:///local_storage/sqlite_db/mlflow.db
MLFLOW_ARTIFACT_ROOT=  # Auto-configured to external storage
```

### Ollama Configuration

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/athena --cov-report=html

# Run specific test file
pytest tests/unit/test_storage.py

# Run integration tests
pytest tests/integration/
```

## Code Quality

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/

# Type check
mypy src/

# Run all checks
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rushikesh Dhumal**
- Portfolio Project for ML/AI Engineer Role
- Demonstrates: Agentic AI, MLOps, Edge AI Optimization, Production-Quality Code

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- LLM powered by [Ollama](https://ollama.ai)
- Experiment tracking with [MLflow](https://mlflow.org)
- Vector store with [Chroma](https://www.trychroma.com/)

---

**Status**: Phase 1 Complete | Active Development

For questions or feedback, please open an issue on GitHub.
