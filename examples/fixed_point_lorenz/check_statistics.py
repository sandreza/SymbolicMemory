import h5py
import numpy as np
from utils import load_lorenz_data
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent.parent.parent))

save_path = "lorenz_model/generated_sequences.hdf5"

with h5py.File(save_path, 'r') as f:
    predictions = f['predictions'][:]

data, val_data = load_lorenz_data()

# Calculate statistics 

p_mean = np.mean(predictions)
p_std = np.std(predictions)
d_mean = np.mean(data)
d_std = np.std(data)

print(f"Data mean: {d_mean}")
print(f"Data std: {d_std}")
print(f"Prediction mean: {p_mean}")
print(f"Prediction std: {p_std}")

# Calculate autocorrelations for each value
def calculate_autocorr(sequence):
    return np.real(np.fft.ifft(np.fft.fft(sequence) * np.fft.ifft(sequence))) - np.mean(sequence)**2

# Create figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot autocorrelations for data
for value in [0, 1, 2]:
    data_mask = (data == value)
    pred_mask = (predictions[0, :] == value)
    
    data_autocorr = calculate_autocorr(data_mask)
    pred_autocorr = calculate_autocorr(pred_mask)
    
    # Plot on first subplot (data)
    axs[value].plot(data_autocorr[:400], label=f'Data (value={value})', alpha=0.7)
    axs[value].plot(pred_autocorr[:400], label=f'Predictions (value={value})', alpha=0.7)


# Customize plots
axs[0].set_ylabel('Autocorrelation of 0-Index')
axs[1].set_ylabel('Autocorrelation of 1-Index')
axs[2].set_ylabel('Autocorrelation of 2-Index')
axs[2].set_xlabel('Lag')
# legend
axs[0].legend()

plt.tight_layout()
plt.show()

