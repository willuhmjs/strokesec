"""
Login verification script using trained neural network model.
"""
import time
import os
import sys
import pickle
import pandas as pd
from config import TARGET_PHRASE, REQUIRED_LENGTH, MODEL_FILE
from capture import KeystrokeCapture

def load_model():
    """Loads the trained authentication model."""
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Run train_model.py first!")
        sys.exit(1)
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def capture_login_attempt():
    """Captures a single login attempt string."""
    print(f"\n[SECURITY CHECK] Type phrase: '{TARGET_PHRASE}' and hit ENTER.")
    
    capture = KeystrokeCapture()
    return capture.capture_sequence()

def verify_user(attempt_data, model):
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
    
    # Get probability [Imposter_Prob, User_Prob]
    proba = model.predict_proba(df)[0] 
    confidence = proba[1] 
    
    print("\n" + "="*30)
    print(f"Biometric Confidence: {confidence*100:.2f}%")
    
    if confidence > 0.85:
        print("✅ ACCESS GRANTED. Welcome back, Will.")
    else:
        print("⛔ ACCESS DENIED. Rhythm does not match.")
    print("="*30 + "\n")

if __name__ == "__main__":
    ai_brain = load_model()
    
    while True:
        try:
            data = capture_login_attempt()
            verify_user(data, ai_brain)
            if input("Test again? (y/n): ").lower() != 'y':
                break
        except KeyboardInterrupt:
            break
