# SymbolicMemory
Using Transformers to Model Symbol Sequences with Memory

## Project Structure

```
.
├── models/
│   ├── attention.py       # Attention mechanism implementation
│   ├── embedding.py       # Token and positional embeddings
│   ├── mlp.py            # MLP layer implementation
│   ├── transformer.py     # Main transformer model
│   ├── autoencoder.py    # Base autoencoder implementation
│   ├── autoencoder_networks.py  # Encoder/decoder networks
│   └── sparse_autoencoder.py    # Sparse autoencoder implementation
├── training/
│   ├── trainer.py        # Training infrastructure
│   └── loss.py           # Loss functions and metrics
└── examples/
    └── cyclic_sequence/  # Example of training on cyclic sequences
        ├── README.md     # Example-specific documentation
        └── train_cyclic.py  # Training script
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

This example demonstrates training a transformer to predict the next token in a repeating sequence.

1. Run the training script:
```bash
python examples/cyclic_sequence/train_cyclic.py
```

The script will:
- Generate a cyclic sequence dataset
- Initialize a transformer model
- Train the model while displaying progress
- Show attention visualizations
- Plot training metrics

You can change the model parameters in `examples/cyclic_sequence/train_cyclic.py`.
