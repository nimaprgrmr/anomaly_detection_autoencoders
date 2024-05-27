#Step 1: Create Synthetic Data
import numpy as np


def create_data():
    # Set the random seed for reproducibility
    np.random.seed(42)
    # Generate normal data
    normal_data = np.random.normal(loc=500000, scale=200000, size=1000) # generate numbers with
    normal_data = np.clip(normal_data, 0, None)  # Ensure no negative values
    # Inject anomalies
    anomalies = np.random.normal(loc=1500000, scale=500000, size=50)
    anomalies = np.clip(anomalies, 0, None) # Ensure no negative values

    # Combine the data
    data = np.concatenate([normal_data, anomalies])
    np.random.shuffle(data)  # Shuffle the data

    data = data.reshape(-1, 1)  # Reshape for the Autoencoder
    return data