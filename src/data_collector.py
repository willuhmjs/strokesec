"""
Data collector script for recording keystroke dynamics.
"""
import os
import shutil
import logging
from datetime import datetime
import pandas as pd
from config import TARGET_PHRASE, REQUIRED_LENGTH, KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, DATA_DIR
from capture import KeystrokeCapture
from utils import check_columns, select_user_profile
from logger import logger

# --- MAIN SETUP ---
try:
    filename = select_user_profile(KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, DATA_DIR)
except KeyboardInterrupt:
    logger.info("User cancelled selection.")
    exit(0)

# --- VALIDATION LOGIC ---
if filename.exists():
    try:
        existing_df = pd.read_csv(filename)
        expected_cols = REQUIRED_LENGTH * 3
        if len(existing_df.columns) != expected_cols:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = filename.with_name(f"{filename.stem}_backup_{timestamp}.csv")
            logger.warning(f"Column mismatch detected! Expected {expected_cols}, "
                  f"found {len(existing_df.columns)}.")
            logger.info(f"Renaming '{filename.name}' to '{backup_name.name}' and starting fresh.")
            shutil.move(filename, backup_name)
    except Exception as e:
        logger.error(f"Error checking {filename}: {e}")

logger.info(f"Saving data to: {filename}")
print(f"Type: '{TARGET_PHRASE}' and hit ENTER.")
print("Run with SUDO if it fails to capture keys.")

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
    print("\nStopping collection...")
    logger.info("Collection stopped by user.")

# --- SAVE LOGIC ---
if records:
    logger.info(f"Saving {len(records)} records to {filename}...")
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
        logger.info("Data saved successfully.")
    else:
        logger.warning("No data to save.")
else:
    logger.info("No valid data collected.")
