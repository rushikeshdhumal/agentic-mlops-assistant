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
| **Experiment Tracking** | MLflow with SQLite + artifact storage |
| **ML Framework** | PyTorch |
| **Vector Store** | Chroma with sentence-transformers |
| **UI Framework** | Streamlit |
| **API Layer** | FastAPI |
| **Database** | SQLite |
| **Testing** | Pytest |
| **CI/CD** | GitHub Actions |

## Quick Start

### Prerequisites

- Python 3.9 or higher
- 16 GB RAM recommended
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
   pip install -r requirements.txt
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
   # Edit .env with your configuration (optional - defaults work)
   ```

6. **Initialize storage**
   ```bash
   python -m athena --check-storage
   ```

### Running Sample Training

```bash
# Train on MNIST
python scripts/train_mnist.py --epochs 10

# Train on CIFAR-10
python scripts/train_cifar10.py --epochs 20 --model cifar10_cnn

# Train ResNet variant
python scripts/train_cifar10.py --epochs 20 --model simple_resnet --lr 0.01
```

### View Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

## Project Structure

```
agentic-mlops-assistant/
├── src/athena/              # Main application code
│   ├── agent/               # Agentic orchestration layer
│   ├── tools/               # Tool implementations
│   ├── mlops/               # MLOps backend (MLflow, training)
│   ├── ui/                  # Streamlit interface
│   ├── api/                 # FastAPI endpoints
│   ├── config/              # Configuration management
│   └── utils/               # Shared utilities
├── tests/                   # Unit and integration tests
├── scripts/                 # Training and utility scripts
├── docs/                    # Documentation
└── configs/                 # Configuration files
```

## Configuration

### MLflow Configuration

```bash
MLFLOW_TRACKING_URI=sqlite:///local_storage/sqlite_db/mlflow.db
MLFLOW_ARTIFACT_ROOT=  # Auto-configured
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
- GitHub: [@rushikeshdhumal](https://github.com/rushikeshdhumal)
- Portfolio project demonstrating: Agentic AI, MLOps, Edge AI Optimization, Production-Quality Code

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- LLM powered by [Ollama](https://ollama.ai)
- Experiment tracking with [MLflow](https://mlflow.org)
- Vector store with [Chroma](https://www.trychroma.com/)

---

For questions or feedback, please open an issue on GitHub.
