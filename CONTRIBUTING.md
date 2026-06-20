# Contributing to Loan Approval Prediction

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person

## How to Contribute

### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/LoanApproval_Prediction.git
cd LoanApproval_Prediction
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Follow PEP 8 style guidelines
- Write clear, descriptive commit messages
- Add docstrings to functions and classes
- Include unit tests for new features

### 4. Code Quality
```bash
# Format code
black src/ app.py train_model.py

# Check linting
flake8 src/ app.py train_model.py

# Run tests
pytest tests/
```

### 5. Submit a Pull Request
- Describe the changes clearly
- Reference any related issues
- Ensure all tests pass

## Development Setup

1. Create virtual environment
2. Install dev dependencies: `pip install -r requirements-dev.txt`
3. Follow code standards above
4. Test thoroughly

## Issues

- Before opening an issue, check if it already exists
- Provide clear description and reproduction steps
- Include relevant error messages and logs

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing!
