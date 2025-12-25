"""
Data collector script for recording keystroke dynamics.
"""
import os
import shutil
from datetime import datetime
import pandas as pd
from config import TARGET_PHRASE, REQUIRED_LENGTH, KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, DATA_DIR
from capture import KeystrokeCapture
from utils import check_columns

def select_profile():
    """Selects which user profile to record data for."""
    print("\n" + "="*30)
    print("   KEYSTROKE DATA COLLECTOR   ")
    print("="*30)
    print("Who is typing right now?")
    print(f"1. Will (The Master) -> {KEYSTROKE_DATA_FILE.name}")
    print(f"2. Stranger (Imposter) -> {IMPOSTER_DATA_FILE.name}")
    print("3. Custom Name (Creates new synced data file)")

    choice = input("\nSelect Option (1-3): ")

    if choice == '1':
        return KEYSTROKE_DATA_FILE
    if choice == '2':
        return IMPOSTER_DATA_FILE
    if choice == '3':
        name = input("Enter your name (e.g. 'alice'): ").strip().lower()
        return DATA_DIR / f"{name}_data.csv"
    
    # Fallback/Default
    print("Invalid choice. Defaulting to keystroke_data.csv")
    return KEYSTROKE_DATA_FILE

# --- MAIN SETUP ---
filename = select_profile()

# --- VALIDATION LOGIC ---
if filename.exists():
    try:
        existing_df = pd.read_csv(filename)
        expected_cols = REQUIRED_LENGTH * 3
        if len(existing_df.columns) != expected_cols:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = filename.with_name(f"{filename.stem}_backup_{timestamp}.csv")
            print(f"\n⚠️  Column mismatch detected! Expected {expected_cols}, "
                  f"found {len(existing_df.columns)}.")
            print(f"   Renaming '{filename.name}' to '{backup_name.name}' and starting fresh.")
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
        print("\nReady... Type!")
        current_record = capture.capture_sequence()

        if len(current_record) == REQUIRED_LENGTH:
            records.append(current_record)
            print(f"✅ Recorded Entry #{len(records)} | Length: {len(current_record)}")
        else:
            # Discard immediately
            print(f"⚠️ Discarded Entry (Length {len(current_record)}). No typos allowed!")
            
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
            row[f'k{i}_hold'] = stroke['hold']
            row[f'k{i}_ud'] = stroke['ud']
            row[f'k{i}_dd'] = stroke['dd']
    processed_data.append(row)

if processed_data:
    df = pd.DataFrame(processed_data)
    if filename.exists():
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    print("Saved successfully.")
else:
    print("No valid data collected.")
