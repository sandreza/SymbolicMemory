import requests
import h5py
import os
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path("examples/fixed_point_lorenz/data")
data_dir.mkdir(parents=True, exist_ok=True)

# Set up file paths
url = "https://zenodo.org/records/15384531/files/fixed_point_lorenz.hdf5?download=1"
output_path = data_dir / "fixed_point_lorenz.hdf5"

# Download data if it doesn't exist
if not output_path.exists():
    print("Downloading data from Zenodo...")
    response = requests.get(url)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")
else:
    print("Data already exists, skipping download.")

# Load and inspect the 'sequence' dataset
with h5py.File(output_path, "r") as f:
    print("Keys in HDF5 file:", list(f.keys()))
    sequence = f["sequence"][:] - 1
    print("Shape of sequence:", sequence.shape)
    print("First 10 entries:", sequence[:10]) 