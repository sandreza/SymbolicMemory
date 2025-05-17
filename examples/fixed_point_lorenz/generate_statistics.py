from training.utils import load
from models.transformer import Transformer
from models.utils import generate_predictions
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import load_lorenz_data
import h5py
from pathlib import Path
import datetime

model = load("lorenz_model/model.mo", Transformer) # load("lorenz_model/st_model.mo", SimpleTransformer)
data, val_data = load_lorenz_data()

print("\n=== Testing model predictions on different windows ===")
print("\nWindow 1 (start=0):")
istart = 0
block = 10
x = jnp.array(data[istart:istart+block])
p = jax.vmap(jax.nn.softmax)(model(x))
print("Input sequence:", x)
print("Predicted probabilities:", p)

predictions = generate_predictions(
    model,
    initial_seq = jnp.array(data[0]),
    max_new_tokens= 10000,
    block_size= 150,
    key = jr.key(0),
    batch_size = 1,
)

# Save predictions to HDF5 file
save_path = "lorenz_model/generated_sequences.hdf5"
save_dir = Path(save_path).parent
save_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(save_path, 'w') as f:
    # Save predictions
    f.create_dataset('predictions', data=np.array(predictions))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'max_new_tokens': 10000,
        'block_size': 150,
        'batch_size': 1,
        'model_type': 'Transformer'
    }
    
    # Save metadata as attributes
    for key, value in metadata.items():
        f.attrs[key] = str(value)

print(f"\nPredictions saved to {save_path}")