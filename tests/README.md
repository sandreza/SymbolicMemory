# Tests

This directory contains tests for the transformer model implementation.

## Project Structure

The project is organized as a Python package with the following structure:
```
.
├── models/
│   ├── __init__.py
│   ├── transformer.py
│   └── utils.py
├── training/
│   ├── __init__.py
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_generation.py
└── pyproject.toml
```

## Running Tests

First, make sure you're in the project root directory. Then run:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_generation.py

# Run with verbose output
pytest -v
```

The `pyproject.toml` file is configured to automatically:
- Add the project root to PYTHONPATH
- Set the default test directory
- Enable verbose output

After running the tests, you'll see a summary showing:
- Number of passed, failed, and skipped tests
- Names of all passed tests
- Detailed test results (if using -v flag)

## Test Coverage

### Generation Tests (`test_generation.py`)

Tests the sequence generation functionality with different configurations:

- Single token generation
- Sequence generation
- Batched generation (multiple sequences in parallel)
- Long context handling
- Temperature effects on generation
- Input validation

Each test uses a small transformer model to ensure fast execution while still testing the core functionality.

## Adding New Tests

When adding new tests:
1. Create a new test file if testing a new component
2. Use the `small_model` fixture for transformer tests
3. Add clear docstrings explaining what each test checks
4. Include assertions that verify both shapes and values

## Test Output

Example test summary:
```
=== Test Summary ===
✓ Passed: 7 tests

Passed Tests:
  ✓ tests/test_generation.py::test_single_token_generation
  ✓ tests/test_generation.py::test_sequence_generation
  ✓ tests/test_generation.py::test_batched_generation
  ✓ tests/test_generation.py::test_batched_sequence_generation
  ✓ tests/test_generation.py::test_long_context_generation
  ✓ tests/test_generation.py::test_temperature_effect
  ✓ tests/test_generation.py::test_input_validation 