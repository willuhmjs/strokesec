"""
Data collector script for recording keystroke dynamics.
"""
import os
import shutil
from datetime import datetime
import pandas as pd
import keyboard
from config import TARGET_PHRASE, REQUIRED_LENGTH, KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE
from capture import KeystrokeCapture

def select_profile():
    """Selects which user profile to record data for."""
    print("\n" + "="*30)
    print("   KEYSTROKE DATA COLLECTOR   ")
    print("="*30)
    print("Who is typing right now?")
    print(f"1. Will (The Master) -> {os.path.basename(KEYSTROKE_DATA_FILE)}")
    print(f"2. Stranger (Imposter) -> {os.path.basename(IMPOSTER_DATA_FILE)}")
    print("3. Custom Name (Creates new synced data file)")

    choice = input("\nSelect Option (1-3): ")

    if choice == '1':
        return KEYSTROKE_DATA_FILE
    if choice == '2':
        return IMPOSTER_DATA_FILE
    
    from config import DATA_DIR
    name = input("Enter your name (e.g. 'alice'): ").strip().lower()
    return os.path.join(DATA_DIR, f"{name}_data.csv")

# --- MAIN SETUP ---
filename = select_profile()

# --- VALIDATION LOGIC ---
if os.path.exists(filename):
    try:
        existing_df = pd.read_csv(filename)
        expected_cols = REQUIRED_LENGTH * 2
        if len(existing_df.columns) != expected_cols:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filename.replace('.csv', '')}_backup_{timestamp}.csv"
            print(f"\n⚠️  Column mismatch detected! Expected {expected_cols}, "
                  f"found {len(existing_df.columns)}.")
            print(f"   Renaming '{filename}' to '{backup_name}' and starting fresh.")
            shutil.move(filename, backup_name)
    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"\n⚠️  Error checking {filename}: {e}")

print(f"\n[INFO] Saving data to: {filename}")
print(f"Type: '{TARGET_PHRASE}' and hit ENTER.")
print("Run with SUDO if it fails.")

# VARIABLES
records = []
capture = KeystrokeCapture()

try:
    while True:
        # We manually manage the loop here to allow continuous recording
        print("\nReady... Type!")
        current_record = capture.capture_sequence()

        if len(current_record) == REQUIRED_LENGTH:
            records.append(current_record)
            print(f"✅ Recorded Entry #{len(records)} | Length: {len(current_record)}")
        else:
            # Discard immediately
            print(f"⚠️ Discarded Entry (Length {len(current_record)}). No typos allowed!")
            
        # Check if user wants to stop (this part is tricky with just capture_sequence)
        # We'll rely on KeyboardInterrupt for now as per original design logic 
        # but improved to use the class
        
except KeyboardInterrupt:
    print("\nStopping...")

# --- SAVE LOGIC ---
print(f"Saving {len(records)} records to {filename}...")
processed_data = []

for attempt in records:
    row = {}
    for i, stroke in enumerate(attempt):
        # We only save up to REQUIRED_LENGTH, though logic ensures they are equal
        if i < REQUIRED_LENGTH:
            row[f'k{i}_dwell'] = stroke['dwell']
            row[f'k{i}_flight'] = stroke['flight']
    processed_data.append(row)

if processed_data:
    df = pd.DataFrame(processed_data)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    print("Saved successfully.")
else:
    print("No valid data collected.")