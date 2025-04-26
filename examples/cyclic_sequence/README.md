# Cyclic Sequence Example

This example demonstrates training a transformer model on a simple cyclic sequence task,
where the model must learn to predict the next token in a repeating sequence.

## Task Description

The dataset consists of a repeating sequence of tokens (e.g., 1,2,3,1,2,3... or 1,1,2,2,3,3,1,1,2,2,3,3).
The model must learn to predict the next token in the sequence, which requires understanding
both the current token and its position in the cycle.

## Running the Example

From the project root directory, run:
```bash
python examples/cyclic_sequence/train_cyclic.py
```

The script will:
1. Generate the cyclic dataset
2. Initialize and train the model
3. Display training progress
4. Show example predictions
5. Visualize attention patterns

## Features

The example demonstrates several key capabilities:
- Training on synthetic sequential data
- Monitoring training progress through loss visualization
- Analyzing attention patterns
- Evaluating model predictions
- Computing attention entropy

## Usage

You can modify the training configuration by editing the parameters in `train_cyclic.py`.
The script provides visualization of:
- Training progress through loss curves
- Attention patterns showing what the model attends to
- Model predictions on example sequences 