"""
Training script for the Keystroke Dynamics Authentication model.
"""
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import argparse
from config import KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, MODEL_FILE, SCALER_FILE

# 0. PARSE ARGUMENTS
# ------------------
parser = argparse.ArgumentParser(description="Train the Keystroke Dynamics Authentication model.")
parser.add_argument("input_file", nargs="?", default=KEYSTROKE_DATA_FILE, help="Path to the positive user data CSV (default: from config).")
parser.add_argument("--model", default=MODEL_FILE, help="Path to save the trained model (default: from config).")
parser.add_argument("--scaler", default=SCALER_FILE, help="Path to save the scaler (default: from config).")
args = parser.parse_args()

POSITIVE_DATA_FILE = args.input_file
OUTPUT_MODEL_FILE = args.model
OUTPUT_SCALER_FILE = args.scaler

print(f"Training on: {POSITIVE_DATA_FILE}")
print(f"Saving model to: {OUTPUT_MODEL_FILE}")
print(f"Saving scaler to: {OUTPUT_SCALER_FILE}")

# 1. LOAD POSITIVE DATA (YOU)
# ---------------------------
if not os.path.exists(POSITIVE_DATA_FILE):
    print(f"Error: '{POSITIVE_DATA_FILE}' not found. Record your data first!")
    sys.exit(1)

real_user = pd.read_csv(POSITIVE_DATA_FILE)
real_user['label'] = 1 
print(f"Loaded {len(real_user)} samples from YOU.")

# 2. LOAD NEGATIVE DATA (IMPOSTERS)
# ---------------------------------
# Strategy: Prefer Real Imposters. Fallback to Synthetic Randomness.

if os.path.exists(IMPOSTER_DATA_FILE):
    print("Found REAL imposter data. Using it.")
    fake_user = pd.read_csv(IMPOSTER_DATA_FILE)
    fake_user['label'] = 0
else:
    print("No real imposter data found. Generating 'Random Stranger' data...")
    
    feature_cols = [c for c in real_user.columns if c != 'label']
    n_imposters = len(real_user) # Balance the classes 50/50

    fake_data = []
    for _ in range(n_imposters):
        row = {}
        for col in feature_cols:
            if 'dwell' in col:
                # Random hold: 50ms to 400ms
                row[col] = np.random.uniform(0.05, 0.4) 
            elif 'flight' in col:
                # Random flight: 0ms to 500ms
                row[col] = np.random.uniform(0.0, 0.5)
        row['label'] = 0
        fake_data.append(row)

    fake_user = pd.DataFrame(fake_data)

print(f"Loaded {len(fake_user)} imposter samples.")

# 3. TRAIN
# --------
data = pd.concat([real_user, fake_user], ignore_index=True)
# Ensure we only use feature columns (exclude label)
feature_cols = [c for c in data.columns if c != 'label']
X = data[feature_cols]
y = data['label']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# MLP: 32 neurons -> 16 neurons
# Increased max_iter for convergence
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, learning_rate_init=0.001)

print("\nTraining Neural Network...")
model.fit(X_train, y_train)

# 4. TEST (Validation Split)
# ------------------------
predictions = model.predict(X_test)
print("\n--- INTERNAL VALIDATION RESULTS (Split from Training Data) ---")
print(classification_report(y_test, predictions, target_names=["Imposter", "Real User"]))

# 5. VERIFY AGAINST REAL HUMANS (Comprehensive Human Validation)
# --------------------------------------------------------------
print("\n--- HUMAN VALIDATION REPORT ---")
from config import DATA_DIR
import glob

# Auto-detect all CSV files in data directory to use for validation
validation_datasets = {}
search_pattern = os.path.join(DATA_DIR, "*_data.csv")
files = glob.glob(search_pattern)

# Determine the training user name from the input file
train_file_base = os.path.basename(POSITIVE_DATA_FILE)

for filename in files:
    base_name = os.path.basename(filename)
    
    # Skip the imposter data file used for training
    if base_name == "imposter_data.csv":
        continue
        
    # Skip the main training file itself? 
    # Usually we want to see how well it performs on the training user (should be ~100% approved)
    # The original script had "Will (Owner)" explicitly.
    # We will include everyone found.
    
    # Heuristic for name: remove _data.csv and capitalize
    persona_name = base_name.replace("_data.csv", "").capitalize()
    
    # If this file is the POSITIVE_DATA_FILE, expect 1 (Real User)
    # Otherwise expect 0 (Imposter)
    # Note: POSITIVE_DATA_FILE might be absolute or relative, so compare basenames
    if base_name == train_file_base:
        expected = 1
        display_name = f"{persona_name} (Owner)"
    else:
        expected = 0
        display_name = persona_name

    validation_datasets[display_name] = {"file": filename, "expected": expected}

# Print Table Header
print(f"{'User Name':<20} | {'Total':<8} | {'Approved':<10} | {'Rejected':<10} | {'Pass Rate':<12}")
print("-" * 75)

for name, info in validation_datasets.items():
    csv_file = info["file"]
    expected_label = info["expected"]
    
    if os.path.exists(csv_file):
        try:
            df_val = pd.read_csv(csv_file)
            
            # Filter to feature_cols to match training features exactly
            # (Silently ignore extra cols, fail if missing cols)
            X_val = df_val[feature_cols]
            
            # Scale
            X_val_scaled = scaler.transform(X_val)
            
            # Predict
            preds_val = model.predict(X_val_scaled)
            
            total = len(preds_val)
            approved = np.sum(preds_val == 1)
            rejected = np.sum(preds_val == 0)
            pass_rate = (approved / total) * 100 if total > 0 else 0
            
            # Color/Status indicator could be added, but simple table for now
            print(f"{name:<20} | {total:<8} | {approved:<10} | {rejected:<10} | {pass_rate:6.2f}%")
            
        except Exception as e:
            print(f"{name:<20} | ERROR: {str(e)}")
    else:
        print(f"{name:<20} | FILE NOT FOUND ({csv_file})")

print("-" * 75)


# 6. SAVE
# -------
with open(OUTPUT_MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to '{OUTPUT_MODEL_FILE}'")

with open(OUTPUT_SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to '{OUTPUT_SCALER_FILE}'")