"""
Login verification script using trained neural network model.
"""
import time
import os
import sys
import argparse
import pickle
import pandas as pd
from config import TARGET_PHRASE, REQUIRED_LENGTH, MODEL_FILE, SCALER_FILE
from capture import KeystrokeCapture

def load_model_and_scaler(model_path, scaler_path):
    """Loads the trained authentication model and scaler."""
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train_model.py first!")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found. Run train_model.py first!")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    return model, scaler

def capture_login_attempt():
    """Captures a single login attempt string."""
    print(f"\n[SECURITY CHECK] Type phrase: '{TARGET_PHRASE}' and hit ENTER.")
    
    capture = KeystrokeCapture()
    return capture.capture_sequence()

def verify_user(attempt_data, model, scaler):
    """Verifies the user based on the captured keystroke data."""
    # Strict Length Check
    if len(attempt_data) != REQUIRED_LENGTH:
        print(f"❌ Login Failed: Incorrect Length ({len(attempt_data)} keys). Did you typo?")
        return

    row = {}
    for i in range(REQUIRED_LENGTH):
        stroke = attempt_data[i]
        row[f'k{i}_dwell'] = stroke['dwell']
        row[f'k{i}_flight'] = stroke['flight']
        
    df = pd.DataFrame([row])
    
    # Scale the input
    df_scaled = scaler.transform(df)
    
    # Get probability [Imposter_Prob, User_Prob]
    proba = model.predict_proba(df_scaled)[0] 
    confidence = proba[1] 
    
    print("\n" + "="*30)
    print(f"Biometric Confidence: {confidence*100:.2f}%")
    
    if confidence > 0.85:
        print("✅ ACCESS GRANTED. Welcome back, Will.")
    else:
        print("⛔ ACCESS DENIED. Rhythm does not match.")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Login verification.")
    parser.add_argument("model", nargs="?", default=MODEL_FILE, help="Path to the model file (default: from config).")
    parser.add_argument("--scaler", default=SCALER_FILE, help="Path to the scaler file (default: from config).")
    args = parser.parse_args()

    ai_brain, scaler = load_model_and_scaler(args.model, args.scaler)
    
    while True:
        try:
            data = capture_login_attempt()
            verify_user(data, ai_brain, scaler)
            if input("Test again? (y/n): ").lower() != 'y':
                break
        except KeyboardInterrupt:
            break
