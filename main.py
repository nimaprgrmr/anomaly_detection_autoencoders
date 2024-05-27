# Step 3: Detect Anomalies
from Buisness.data_preprocess import create_data
from Buisness.model import create_model
import numpy as np


data = create_data()
autoencoder, history, normalized_data = create_model(data)


def detect_anomalies(autoencoder, normalized_data):
    # Predict the reconstruction
    reconstructed_data = autoencoder.predict(normalized_data)

    # Calculate reconstruction error
    reconstruction_error = np.mean(np.abs(normalized_data - reconstructed_data), axis=1)
    print(f"reconstruction_error: {reconstruction_error}\n")

    # Determine threshold for anomaly detection
    threshold = np.percentile(reconstruction_error, 95)
    print(f"Threshhold: {threshold}\n")

    # Detect anomalies
    anomalies = reconstruction_error > threshold
    print("Anomalies detected:", np.sum(anomalies),"\n")

    # Show some examples
    anomalous_data = data[anomalies]
    print("Anomalous data:", anomalous_data[:10])

    return anomalous_data


anomaly_data = detect_anomalies(autoencoder, normalized_data)
