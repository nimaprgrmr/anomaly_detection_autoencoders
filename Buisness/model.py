# Step 2: Build and Train the Autoencoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from typing import Dict
from tensorflow.keras.regularizers import l2


def create_model(data: np.ndarray) -> [Model, MinMaxScaler, Dict, np.ndarray]:
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Define Autoencoder architecture
    input_dim = normalized_data.shape[1]
    encoding_dim = 8  # Size of the encoded representation

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
    encoder = Dense(encoding_dim // 2, activation='relu', kernel_regularizer=l2(0.01))(encoder)
    encoder = Dropout(0.2)(encoder)  # Adding dropout for regularization

    # Decoder
    decoder = Dense(encoding_dim // 2, activation='relu', kernel_regularizer=l2(0.01))(encoder)
    decoder = Dense(input_dim, activation='sigmoid', kernel_regularizer=l2(0.01))(decoder)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # Train the Autoencoder
    history = autoencoder.fit(normalized_data, normalized_data,
                              epochs=100,
                              batch_size=32,
                              validation_split=0.2,
                              shuffle=True)

    return autoencoder, history, normalized_data
