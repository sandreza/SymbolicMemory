# SymbolicMemory
Using Transformers to Model Symbol Sequences with Memory

## Project Structure

```
.
├── models/
│   ├── attention.py     # Attention mechanism implementation
│   ├── mlp.py           # MLP layer implementation
│   ├── transformer.py   # Main transformer model
│   ├── autoencoder.py   # Base autoencoder implementation
│   └── utils.py         # Utility functions for model operations
├── training/
│   ├── trainer.py        # Training infrastructure
│   ├── loss.py           # Loss functions and metrics
│   ├── sae_trainer.py    # Sparse autoencoder training infrastructure
│   └── utils.py          # Utility functions for model saving/loading
└── examples/
    └── cyclic_sequence/  # Example of training on cyclic sequences
        ├── README.md     # Example-specific documentation
        ├── train_cyclic.py  # Training script for transformer
        ├── train_sae.py     # Training script for sparse autoencoder
        ├── sae_mechanistic_intervention.py  # Intervention experiments
        └── check_hallucinations.py  # Hallucination testing
```

## Setup Instructions

First change into the `SymbolicMemory` directory and follow the instructions below to set up the repository:

1. Install `uv`:
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install Python 3.11 using `uv`:
    ```sh
    uv python install 3.11
    ```

3. Create a virtual environment with Python 3.11 and sync:
    ```sh
    uv venv --python 3.11
    source .venv/bin/activate
    uv sync
    ```

## Running the Examples

### Cyclic Sequence Example

This example demonstrates training a transformer to predict the next token in a repeating sequence, and then training a sparse autoencoder to analyze its internal representations.

1. First, train the transformer model:
```bash
python examples/cyclic_sequence/train_cyclic.py
```

2. Then, train the sparse autoencoder on the transformer's activations:
```bash
python examples/cyclic_sequence/train_sae.py
```

3. Finally, run mechanistic interventions using the trained models:
```bash
python examples/cyclic_sequence/sae_mechanistic_intervention.py
```

The scripts will:
- Generate cyclic sequence datasets
- Train the transformer model
- Train the sparse autoencoder on transformer activations
- Perform mechanistic interventions to analyze the model's behavior
- Show attention and activation visualizations
- Plot training metrics

### Model Components

1. **Transformer Model** (`models/transformer.py`):
   - Implements a simple transformer architecture
   - Handles sequence prediction tasks
   - Supports mechanistic interventions

2. **Sparse Autoencoder** (`models/autoencoder.py`):
   - Implements a sparse autoencoder architecture
   - Trains on transformer layer activations
   - Supports expansion factors for different inflation ratios

3. **Training Infrastructure** (`training/`):
   - `trainer.py`: Base training infrastructure
   - `sae_trainer.py`: Specialized trainer for sparse autoencoders
   - `loss.py`: Loss functions for both models
   - `utils.py`: Model saving/loading utilities

### Configuration

You can modify the following parameters in the training scripts:

1. In `examples/cyclic_sequence/train_cyclic.py`:
   - Model dimensions
   - Number of layers
   - Training steps
   - Learning rate

2. In `examples/cyclic_sequence/train_sae.py`:
   - Expansion factor
   - Layer to analyze
   - Training steps
   - Batch size

3. In `examples/cyclic_sequence/sae_mechanistic_intervention.py`:
   - Intervention strength
   - Number of candidate indices
   - Sequence generation length
