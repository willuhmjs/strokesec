import json
import pickle
import torch
import numpy as np
from config import MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from pathlib import Path
from utils import load_artifacts

def export_artifacts():
    print("Loading artifacts...")
    
    # Load artifacts using the shared utility which handles PyTorch loading
    model, scaler, threshold = load_artifacts(MODEL_FILE, SCALER_FILE, THRESHOLD_FILE)

    print("Exporting to JSON...")

    # helper to convert numpy arrays to lists
    def to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr

    # Extract weights and biases from PyTorch model
    weights = []
    biases = []
    
    # Iterate through named parameters to find weights and biases
    # The order depends on the definition in model.py
    # Updated Architecture:
    # Encoder: Linear -> BN -> LeakyReLU -> Dropout -> Linear -> BN -> LeakyReLU -> Dropout -> Linear -> LeakyReLU
    # Decoder: Linear -> BN -> LeakyReLU -> Linear -> BN -> LeakyReLU -> Linear
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'batch_norm' not in name: 
                 weights.append(to_list(param.detach().numpy().T)) # Transpose for standard matrix mult convention (Input x Hidden)
        elif 'bias' in name:
             if 'batch_norm' not in name:
                biases.append(to_list(param.detach().numpy()))

    # Prepare data structure
    data = {
        "threshold": float(threshold),
        "scaler": {
            "mean": to_list(scaler.mean_),
            "scale": to_list(scaler.scale_)
        },
        "model": {
            "weights": weights,
            "biases": biases,
            "activation": "leaky_relu",
            "leaky_relu_slope": 0.1
        }
    }

    # Save to web_collector directory
    output_path = Path("web_collector/model_data.json")
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"Successfully exported model data to {output_path}")

if __name__ == "__main__":
    export_artifacts()