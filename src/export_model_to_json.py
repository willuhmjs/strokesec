import json
import pickle
import numpy as np
from config import MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from pathlib import Path

def export_artifacts():
    print("Loading artifacts...")
    
    # Load pickles
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(THRESHOLD_FILE, "rb") as f:
        threshold = pickle.load(f)

    print("Exporting to JSON...")

    # helper to convert numpy arrays to lists
    def to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr

    # Prepare data structure
    data = {
        "threshold": float(threshold),
        "scaler": {
            "mean": to_list(scaler.mean_),
            "scale": to_list(scaler.scale_)
        },
        "model": {
            # coefs_ is a list of weight matrices, intercepts_ is a list of bias vectors
            "weights": [to_list(w) for w in model.coefs_],
            "biases": [to_list(b) for b in model.intercepts_],
            "activation": model.activation, # 'relu' expected
            "layer_sizes": model.hidden_layer_sizes
        }
    }

    # Save to web_collector directory
    output_path = Path("web_collector/model_data.json")
    with open(output_path, "w") as f:
        json.dump(data, f)
    
    print(f"Successfully exported model data to {output_path}")

if __name__ == "__main__":
    export_artifacts()