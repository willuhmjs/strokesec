import keyboard
import time
import pandas as pd
import os

# CONFIG
TARGET_PHRASE = "the quick brown fox"
# "the quick brown fox" (19 chars) + Enter (1 char) = 20
REQUIRED_LENGTH = 20 

def select_profile():
    print("\n" + "="*30)
    print("   KEYSTROKE DATA COLLECTOR   ")
    print("="*30)
    print("Who is typing right now?")
    print("1. Will (The Master) -> keystroke_data.csv")
    print("2. Stranger (Imposter) -> imposter_data.csv")
    print("3. Custom Name...")
    
    choice = input("\nSelect Option (1-3): ")
    
    if choice == '1':
        return "keystroke_data.csv"
    elif choice == '2':
        return "imposter_data.csv"
    else:
        name = input("Enter filename (e.g. 'mom'): ")
        return f"{name}_data.csv"

# --- MAIN SETUP ---
filename = select_profile()

# --- VALIDATION LOGIC ---
if os.path.exists(filename):
    try:
        existing_df = pd.read_csv(filename)
        expected_cols = REQUIRED_LENGTH * 2
        if len(existing_df.columns) != expected_cols:
            import shutil
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filename.replace('.csv', '')}_backup_{timestamp}.csv"
            print(f"\n⚠️  Column mismatch detected! Expected {expected_cols}, found {len(existing_df.columns)}.")
            print(f"   Renaming '{filename}' to '{backup_name}' and starting fresh.")
            shutil.move(filename, backup_name)
    except Exception as e:
        print(f"\n⚠️  Error checking {filename}: {e}")

print(f"\n[INFO] Saving data to: {filename}")
print(f"Type: '{TARGET_PHRASE}' and hit ENTER.")
print("Run with SUDO if it fails.")

# VARIABLES
records = []
current_record = []
press_times = {}

def on_event(event):
    global current_record, records, press_times
    
    k = event.name
    now = event.time
    
    if event.event_type == 'down':
        if k not in press_times:
            press_times[k] = now

    elif event.event_type == 'up':
        if k in press_times:
            start_time = press_times.pop(k)
            dwell = now - start_time
            
            flight = 0.0
            if len(current_record) > 0:
                last_release = current_record[-1]['release_ts']
                flight = start_time - last_release

            current_record.append({
                'key': k,
                'dwell': dwell,
                'flight': flight,
                'release_ts': now
            })
            
            if k == 'enter':
                # --- NEW FILTER LOGIC ---
                if len(current_record) == REQUIRED_LENGTH:
                    records.append(current_record)
                    print(f"✅ Recorded Entry #{len(records)} | Length: {len(current_record)}")
                else:
                    # Discard immediately
                    print(f"⚠️ Discarded Entry (Length {len(current_record)}). No typos allowed!")
                
                current_record = []
                press_times = {}

# Hook the global keyboard
keyboard.hook(on_event)

try:
    keyboard.wait() 
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