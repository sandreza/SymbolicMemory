# Fixed Point Lorenz Example

This example demonstrates training a transformer model on sequences generated from the Lorenz system,
where the model must learn to predict the next state in the chaotic time series, and then analyzing
its internal representations using a sparse autoencoder.

## Task Description

The dataset consists of time series data from the Lorenz system, a set of three coupled differential equations
that exhibit chaotic behavior. The model must learn to predict the next state in the sequence, which requires
understanding the complex dynamics of the system.

## Running the Example

From the project root directory, run the following scripts in order:

1. Train the transformer model:
```bash
python examples/fixed_point_lorenz/train_lorenz.py
```

2. Train the sparse autoencoder on transformer activations:
```bash
python examples/fixed_point_lorenz/train_sae.py
```

3. Run mechanistic interventions:
```bash
python examples/fixed_point_lorenz/sae_mechanistic_intervention.py
```

4. Check for hallucinations:
```bash
python examples/fixed_point_lorenz/check_hallucinations.py
```

## Features

The example demonstrates several key capabilities:

### Transformer Training
- Training on chaotic time series data
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

### Transformer Training (`train_lorenz.py`)
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