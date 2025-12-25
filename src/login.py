"""
Login verification script using trained Autoencoder model.
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from config import TARGET_PHRASE, REQUIRED_LENGTH, MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from capture import KeystrokeCapture

def load_artifacts(model_path, scaler_path, threshold_path):
    """Loads the trained model, scaler, and threshold."""
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train.py first!")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found. Run train.py first!")
        sys.exit(1)
    if not os.path.exists(threshold_path):
        print(f"Error: {threshold_path} not found. Run train.py first!")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)
        
    return model, scaler, threshold

def capture_login_attempt():
    """Captures a single login attempt string."""
    print(f"\n[SECURITY CHECK] Type phrase: '{TARGET_PHRASE}' and hit ENTER.")
    
    capture = KeystrokeCapture()
    return capture.capture_sequence()

def verify_user(attempt_data, model, scaler, threshold):
    """Verifies the user based on the captured keystroke data."""
    # Strict Length Check
    if len(attempt_data) != REQUIRED_LENGTH:
        print(f"❌ Login Failed: Incorrect Length ({len(attempt_data)} keys). Did you typo?")
        return

    row = {}
    for i in range(REQUIRED_LENGTH):
        stroke = attempt_data[i]
        row[f'k{i}_hold'] = stroke['hold']
        row[f'k{i}_ud'] = stroke['ud']
        row[f'k{i}_dd'] = stroke['dd']
        
    df = pd.DataFrame([row])
    
    # Scale the input
    try:
        df_scaled = scaler.transform(df)
        
        # Reconstruct
        reconstructed = model.predict(df_scaled)
        
        # Calculate Error (MSE)
        mse = mean_squared_error(df_scaled[0], reconstructed[0])
        
        print("\n" + "="*30)
        print(f"Reconstruction Error (MSE): {mse:.6f}")
        print(f"Access Threshold:          {threshold:.6f}")
        
        if mse <= threshold:
            print("✅ ACCESS GRANTED. Pattern Matched.")
        else:
            print("⛔ ACCESS DENIED. Abnormal typing pattern detected.")
        print("="*30 + "\n")
        
    except ValueError as e:
        print(f"\n❌ Error during verification: {e}")
        print("This usually means the feature count doesn't match the model.")
        print(f"Expected features: {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'Unknown'}")
        print(f"Provided features: {df.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Login verification.")
    parser.add_argument("--model", default=MODEL_FILE, help="Path to the model file (default: from config).")
    parser.add_argument("--scaler", default=SCALER_FILE, help="Path to the scaler file (default: from config).")
    parser.add_argument("--threshold", default=THRESHOLD_FILE, help="Path to the threshold file (default: from config).")
    args = parser.parse_args()

    try:
        model, scaler, threshold = load_artifacts(args.model, args.scaler, args.threshold)
        print(f"Loaded artifacts. Threshold set to: {threshold:.6f}")
        
        while True:
            try:
                data = capture_login_attempt()
                verify_user(data, model, scaler, threshold)
                if input("Test again? (y/n): ").lower() != 'y':
                    break
            except KeyboardInterrupt:
                break
    except Exception as e:
        print(f"Error initializing login system: {e}")
        sys.exit(1)
