"""Utility functions for model serialization and deserialization.

This module provides functions to save and load transformer models along with their
hyperparameters. It handles serialization of both the model parameters and configuration,
making it easy to save trained models and restore them later.
"""

import json
from typing import Any, Dict, TypeVar, Type, Callable
import equinox as eqx
import jax.random as jr

# Type variable for the model class
ModelT = TypeVar('ModelT', bound=eqx.Module)

def save(
    filename: str,
    hyperparams: Dict[str, Any],
    model: eqx.Module
) -> None:
    """Save a model and its hyperparameters to a file.
    
    The function saves both the model's hyperparameters as JSON and the model's
    parameters using equinox's serialization. The hyperparameters are stored
    as a JSON string in the first line, followed by the serialized model parameters.
    
    Args:
        filename: Path where the model should be saved
        hyperparams: Dictionary of model hyperparameters
        model: The equinox model to save
    """
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def load(
    filename: str,
    model_name: Type[ModelT]
) -> ModelT:
    """Load a model and its hyperparameters from a file.
    
    The function reconstructs a model by:
    1. Reading and parsing the hyperparameters from the first line
    2. Creating a new model instance with these hyperparameters
    3. Loading the saved parameters into the model
    
    Args:
        filename: Path to the saved model file
        model_name: The model class to instantiate
        
    Returns:
        A model instance with the loaded parameters
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_name(key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

def load_hyperparameters(
    filename: str
) -> Dict[str, Any]:
    """Load only the hyperparameters from a saved model file.
    
    This is useful when you need to inspect or modify the hyperparameters
    without loading the full model.
    
    Args:
        filename: Path to the saved model file
        
    Returns:
        Dictionary containing the model's hyperparameters
    """
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        return hyperparams