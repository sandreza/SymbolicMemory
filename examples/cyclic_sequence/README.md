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

## Model Parameters

The default configuration uses:
- Transformer model with 4 layers and 9 attention heads
- Model dimension of 72 (n_heads * 8)
- Sequence length of 10
- Batch size of 32
- Learning rate of 1e-3
- 2000 training steps

You can modify these parameters directly in `train_cyclic.py`.

## Model Architecture

- Transformer with 4 layers
- 9 attention heads
- Model dimension of 72 (n_heads * 8)
- Maximum sequence length of 10

## Training

The model is trained using:
- AdamW optimizer with learning rate 1e-3
- Batch size of 32
- Sequence length of 10
- 2000 training steps

## Visualization

During training, the example:
1. Prints loss every 100 steps
2. Visualizes attention patterns every 1000 steps
3. Computes and displays attention entropy

## Usage

Run the example with:
```bash
python train_cyclic.py
```

The script will:
1. Generate the cyclic dataset
2. Initialize and train the model
3. Display training progress
4. Show example predictions
5. Visualize attention patterns 