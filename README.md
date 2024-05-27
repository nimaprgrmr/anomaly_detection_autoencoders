# Anomaly Detection with Autoencoders

This repository contains a project for detecting anomalies in time series data using Autoencoders. The project is implemented in Python using TensorFlow and Keras libraries.

## Project Structure

- `Anomaly_Detection/`
  - `main.py` - The main script to run the anomaly detection process.
  - `Buisness/`
    - `model.py` - Contains the definition of the Autoencoder model.
    - `preprocess.py` - Contains functions for preprocessing the data.
- `.gitignore` - Specifies files and directories to be ignored by Git.
- `requirements.txt` - All require packages for run this code.
- `README.md` - This file.

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/nimaprgrmr/anomaly_detection_autoencoders.git
   cd anomaly_detection_autoencoders/Anomaly_Detection

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt

## Usage
1. **Prepare your data:**
      Place your time series data in the appropriate format as required by the preprocess.py script.
   
3. **Run the main script:**
   ```sh
   python main.py

The script will preprocess the data, train the Autoencoder model, and detect anomalies in the data.

**Contributing**
  
   Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

**License**

   This project is licensed under the MIT License.

**Contact**

   If you have any questions or suggestions, feel free to reach out to me at nimatoqiri@gmail.com.
