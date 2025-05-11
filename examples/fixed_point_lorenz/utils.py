"""Utility functions for the fixed point Lorenz example."""

import h5py
import jax.numpy as jnp
from pathlib import Path

def load_lorenz_data(file_path="examples/fixed_point_lorenz/data/fixed_point_lorenz.hdf5", train_split=0.8, n_samples=None):
    """Load and preprocess Lorenz system data.
    
    Args:
        file_path: Path to the HDF5 file containing the Lorenz system data
        train_split: Fraction of data to use for training (default: 0.8)
        n_samples: Number of samples to load (default: None, load all)
    
    Returns:
        If n_samples is None:
            train_data, val_data: Tuple of training and validation data
        Otherwise:
            sequence: The loaded and preprocessed sequence
    """
    with h5py.File(file_path, "r") as f:
        if n_samples is not None:
            sequence = f["sequence"][:n_samples] - 1
        else:
            sequence = f["sequence"][:] - 1
    
    
    if n_samples is not None:
        return sequence
    
    # Split into train and validation
    n_train = int(len(sequence) * train_split)
    train_data = sequence[:n_train]
    val_data = sequence[n_train:]
    
    return train_data, val_data 