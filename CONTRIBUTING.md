# Contributing to ATHENA MLOps Platform

Thank you for your interest in contributing to ATHENA! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/rushikeshdhumal/agentic-mlops-assistant.git
   cd agentic-mlops-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   # Format code
   black src/ tests/ scripts/
   isort src/ tests/ scripts/

   # Run linters
   flake8 src/ tests/ scripts/
   mypy src/

   # Run tests
   pytest tests/ --cov=src/athena
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions or changes
   - `refactor:` Code refactoring
   - `style:` Code style changes
   - `chore:` Build process or tooling changes

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Maximum complexity: 10 (enforced by flake8)
- Add type hints to all function signatures
- Use Google-style docstrings

### Documentation

Every public function should have a docstring:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.

    Longer description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.
    """
    pass
```

### Testing

- Write unit tests for all new functionality
- Aim for >85% code coverage
- Use pytest fixtures for common test setup
- Test edge cases and error conditions
- Integration tests for multi-component features

Example test structure:

```python
def test_my_function():
    """Test my_function with valid input."""
    # Arrange
    input_data = "test"

    # Act
    result = my_function(input_data)

    # Assert
    assert result is True
```

## Project Structure

When adding new functionality:

- **Agent code**: `src/athena/agent/`
- **Tools**: `src/athena/tools/`
- **MLOps backend**: `src/athena/mlops/`
- **API endpoints**: `src/athena/api/`
- **UI components**: `src/athena/ui/`
- **Configuration**: `src/athena/config/`
- **Utilities**: `src/athena/utils/`
- **Tests**: Mirror the structure in `tests/unit/` and `tests/integration/`

## Pull Request Process

1. **Ensure all checks pass**
   - All tests pass
   - Code is formatted correctly
   - Linting passes
   - Coverage is maintained or improved

2. **Update documentation**
   - Update README if needed
   - Add docstrings to new code
   - Update relevant documentation pages

3. **Write a clear PR description**
   - Describe what changes were made
   - Explain why the changes are needed
   - Reference any related issues

4. **Wait for review**
   - Address reviewer feedback
   - Keep the PR up to date with main branch

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant logs or error messages

## Feature Requests

Feature requests are welcome! Please:

- Check if the feature already exists or is planned
- Describe the use case clearly
- Explain how it fits with the project goals
- Be open to discussion about implementation

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Reach out to the maintainers

Thank you for contributing to ATHENA!
