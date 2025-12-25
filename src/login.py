import keyboard
import time
import pandas as pd
import pickle
import sys
import os

# CONFIG
TARGET_PHRASE = "the quick brown fox"
MODEL_FILE = "auth_model.pkl"
EXPECTED_LENGTH = 20  # Updated for new phrase

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Run train_model.py first!")
        sys.exit(1)
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def capture_login_attempt():
    print(f"\n[SECURITY CHECK] Type phrase: '{TARGET_PHRASE}' and hit ENTER.")
    
    current_record = []
    press_times = {}
    recorded_sequence = []
    done = False

    def on_action(event):
        nonlocal done
        k = event.name
        now = event.time
        
        if event.event_type == 'down':
            if k not in press_times:
                press_times[k] = now
        
        elif event.event_type == 'up':
            if k in press_times:
                start = press_times.pop(k)
                dwell = now - start
                
                flight = 0.0
                if len(recorded_sequence) > 0:
                    last_release = recorded_sequence[-1]['release_ts']
                    flight = start - last_release
                
                recorded_sequence.append({
                    'dwell': dwell,
                    'flight': flight,
                    'release_ts': now
                })
                
                if k == 'enter':
                    done = True

    hook = keyboard.hook(on_action)
    
    while not done:
        time.sleep(0.01)
        
    keyboard.unhook(hook)
    return recorded_sequence

def verify_user(attempt_data, model):
    # Strict Length Check
    if len(attempt_data) != EXPECTED_LENGTH:
        print(f"❌ Login Failed: Incorrect Length ({len(attempt_data)} keys). Did you typo?")
        return

    row = {}
    for i in range(EXPECTED_LENGTH):
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