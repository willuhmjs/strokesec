"""
Login verification script using PyTorch Autoencoder.
"""
import argparse
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from config import TARGET_PHRASE, REQUIRED_LENGTH, MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from capture import KeystrokeCapture
from utils import load_artifacts
from logger import logger

def capture_login_attempt():
    print(f"\n[SECURITY CHECK] Type phrase: '{TARGET_PHRASE}' and hit ENTER.")
    capture = KeystrokeCapture()
    return capture.capture_sequence()

def verify_user(attempt_data, model, scaler, threshold):
    if len(attempt_data) != REQUIRED_LENGTH:
        logger.warning(f"Login Failed: Incorrect Length ({len(attempt_data)}).")
        return

    # 1. Preprocess
    row = {}
    for i in range(REQUIRED_LENGTH):
        stroke = attempt_data[i]
        row[f'k{i}_hold'] = stroke['hold']
        row[f'k{i}_ud'] = stroke['ud']
        row[f'k{i}_dd'] = stroke['dd']
        
    df = pd.DataFrame([row])
    
    try:
        # 2. Scale
        input_vector = scaler.transform(df.values)
        
        # 3. Convert to Tensor
        input_tensor = torch.FloatTensor(input_vector)
        
        # 4. Predict (Reconstruct)
        model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            reconstructed_tensor = model(input_tensor)
            
        reconstructed = reconstructed_tensor.numpy()
        
        # 5. Calculate Error
        mse = mean_squared_error(input_vector[0], reconstructed[0])
        
        print("\n" + "="*30)
        logger.info(f"MSE Error:      {mse:.6f}")
        logger.info(f"Auth Threshold: {threshold:.6f}")
        
        if mse <= threshold:
            print("✅ ACCESS GRANTED.")
            logger.info("Access Granted")
        else:
            print("⛔ ACCESS DENIED.")
            logger.warning("Access Denied (High Anomaly Score)")
        print("="*30 + "\n")
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_FILE)
    parser.add_argument("--scaler", default=SCALER_FILE)
    parser.add_argument("--threshold", default=THRESHOLD_FILE)
    args = parser.parse_args()

    try:
        model, scaler, threshold = load_artifacts(args.model, args.scaler, args.threshold)
        logger.info(f"System Ready. Threshold: {threshold:.6f}")
        
        while True:
            try:
                data = capture_login_attempt()
                verify_user(data, model, scaler, threshold)
                if input("Test again? (y/n): ").lower() != 'y':
                    break
            except KeyboardInterrupt:
                logger.info("Exiting login loop.")
                break
    except Exception as e:
        logger.critical(f"Startup Error: {e}")
        sys.exit(1)
