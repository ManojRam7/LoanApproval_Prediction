# Development & Testing Guide

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test

```bash
pytest tests/test_model_manager.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Code Quality

### Format Code with Black

```bash
black src/ app.py train_model.py
```

### Check Linting with Flake8

```bash
flake8 src/ app.py train_model.py --max-line-length=100
```

### Sort Imports with isort

```bash
isort src/ app.py train_model.py
```

### Combined Check

```bash
black --check src/ app.py train_model.py
flake8 src/ app.py train_model.py
```

## Documentation

### Generate Docstrings

All functions should have docstrings in NumPy style:

```python
def example_function(param1: str, param2: int) -> dict:
    """
    Brief description of function.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    dict
        Description of return value
    """
    pass
```

## Pre-commit Hooks

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
black --check src/ app.py train_model.py
flake8 src/ app.py train_model.py
pytest tests/ -q
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Performance Testing

### Profile Model Inference

```python
import time
from src.model_manager import get_model_manager
import pandas as pd

manager = get_model_manager()
test_data = pd.DataFrame(...) # Your test data

# Measure prediction time
start = time.time()
for _ in range(1000):
    manager.predict(test_data)
end = time.time()

print(f"1000 predictions in {end-start:.2f}s")
```

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements-dev.txt
      - run: black --check src/ app.py train_model.py
      - run: flake8 src/ app.py train_model.py
      - run: pytest tests/ --cov=src
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Interactive Debugging with pdb

```python
import pdb; pdb.set_trace()
```

### Inspect Model

```python
from src.model_manager import get_model_manager

manager = get_model_manager()
info = manager.get_model_info()
print(info)
```

## Performance Tips

1. **Model Loading**: Uses caching (singleton pattern)
2. **Data Processing**: Vectorized with pandas/numpy
3. **Prediction**: Batch predictions supported
4. **Logging**: Rotating file handler prevents disk issues

## Common Issues

| Issue | Solution |
|-------|----------|
| Tests fail | Run `pytest -v` for details |
| Black format fails | Check line length (default 88) |
| Import errors in tests | Ensure sys.path includes project root |
| Model pickle incompatible | Retrain with current scikit-learn version |

---

For more information, see README.md and INSTALLATION.md
