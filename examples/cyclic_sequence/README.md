# Cyclic Sequence Example

This example demonstrates training a transformer model on a simple cyclic sequence task,
where the model must learn to predict the next token in a repeating sequence, and then
analyzing its internal representations using a sparse autoencoder.

## Task Description

The dataset consists of a repeating sequence of tokens (e.g., 1,2,3,1,2,3... or 1,1,2,2,3,3,1,1,2,2,3,3).
The model must learn to predict the next token in the sequence, which requires understanding
both the current token and its position in the cycle.

## Running the Example

From the project root directory, run the following scripts in order:

1. Train the transformer model:
```bash
python examples/cyclic_sequence/train_cyclic.py
```

2. Train the sparse autoencoder on transformer activations:
```bash
python examples/cyclic_sequence/train_sae.py
```

3. Run mechanistic interventions:
```bash
python examples/cyclic_sequence/sae_mechanistic_intervention.py
```

4. Check for hallucinations:
```bash
python examples/cyclic_sequence/check_hallucinations.py
```

## Features

The example demonstrates several key capabilities:

### Transformer Training
- Training on synthetic sequential data
- Monitoring training progress through loss visualization
- Analyzing attention patterns
- Evaluating model predictions
- Computing attention entropy

### Sparse Autoencoder Analysis
- Training on transformer layer activations
- Visualizing activation reconstructions
- Analyzing sparse representations
- Computing reconstruction loss
- Monitoring training progress

### Mechanistic Interventions
- Identifying important activation dimensions
- Testing interventions on different sequences
- Generating sequences with interventions
- Analyzing intervention effects
- Testing individual candidate indices

### Hallucination Testing
- Testing on normal sequences
- Testing on forbidden patterns
- Testing on random sequences
- Testing on repeated patterns
- Testing on edge cases

## Usage

You can modify the training configuration by editing the parameters in the respective scripts:

### Transformer Training (`train_cyclic.py`)
- Model dimensions
- Number of layers
- Training steps
- Learning rate
- Batch size

### Sparse Autoencoder Training (`train_sae.py`)
- Expansion factor
- Layer to analyze
- Training steps
- Batch size
- Learning rate
- Weight decay

### Mechanistic Interventions (`sae_mechanistic_intervention.py`)
- Intervention strength
- Number of candidate indices
- Sequence generation length
- Test sequence types
- Layer to intervene on

## Outputs

The scripts provide visualization of:
- Training progress through loss curves
- Attention patterns showing what the model attends to
- Model predictions on example sequences
- Activation reconstructions
- Intervention effects on predictions
- Generated sequences with interventions 