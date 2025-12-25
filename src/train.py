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
import generate_imposters
import visualize

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

# 1. GENERATE IMPOSTER DATA
# ---------------------------
generate_imposters.generate_imposter_data()

# 2. LOAD POSITIVE DATA (YOU)
# ---------------------------
if not os.path.exists(POSITIVE_DATA_FILE):
    print(f"Error: '{POSITIVE_DATA_FILE}' not found. Record your data first!")
    sys.exit(1)

real_user = pd.read_csv(POSITIVE_DATA_FILE)
real_user['label'] = 1 
print(f"Loaded {len(real_user)} samples from YOU.")

# 3. LOAD NEGATIVE DATA (IMPOSTERS)
# ---------------------------------
if os.path.exists(IMPOSTER_DATA_FILE):
    print("Found imposter data. Using it.")
    fake_user = pd.read_csv(IMPOSTER_DATA_FILE)
    fake_user['label'] = 0
else:
    print("Error: Imposter data file missing even after generation attempt.")
    sys.exit(1)

print(f"Loaded {len(fake_user)} imposter samples.")

# 4. SPLIT AND BALANCE (CRITICAL FIX)
# -----------------------------------
# We must split BEFORE oversampling to prevent data leakage (cheating).
# We split real and fake separately, then balance only the training data.

# A. Identify feature columns
feature_cols = [c for c in real_user.columns if c != 'label']

# B. Split Real User Data (80% Train, 20% Test)
real_train, real_test = train_test_split(real_user, test_size=0.2, random_state=42)

# C. Split Imposter Data (80% Train, 20% Test)
fake_train, fake_test = train_test_split(fake_user, test_size=0.2, random_state=42)

# D. Oversample Real User Training Data
# We duplicate real_train rows until they match the count of fake_train
print(f"\n--- BALANCING DATA ---")
print(f"Original Training Split -> Real: {len(real_train)}, Imposter: {len(fake_train)}")

real_train_oversampled = real_train.sample(n=len(fake_train), replace=True, random_state=42)
print(f"Oversampled Training Split -> Real: {len(real_train_oversampled)}, Imposter: {len(fake_train)}")

# E. Combine into final sets
train_data = pd.concat([real_train_oversampled, fake_train], ignore_index=True)
test_data = pd.concat([real_test, fake_test], ignore_index=True)

X_train_raw = train_data[feature_cols]
y_train = train_data['label']

X_test_raw = test_data[feature_cols]
y_test = test_data['label']

# 5. SCALE
# --------
scaler = StandardScaler()
# Fit only on training data
X_train = scaler.fit_transform(X_train_raw)
# Transform test data using the training scaler
X_test = scaler.transform(X_test_raw)

# 6. TRAIN
# --------
# MLP: Simplified architecture to prevent overfitting
# Reduced to single layer of 5 neurons and increased alpha for regularization
model = MLPClassifier(hidden_layer_sizes=(5,), alpha=0.5, max_iter=3000, random_state=42, learning_rate_init=0.001)

print("\nTraining Neural Network...")
model.fit(X_train, y_train)

# 7. TEST (Internal Validation)
# -----------------------------
predictions = model.predict(X_test)
print("\n--- INTERNAL VALIDATION RESULTS (Strict Split) ---")
print(classification_report(y_test, predictions, target_names=["Imposter", "Real User"]))

# 8. VERIFY AGAINST REAL HUMANS (Comprehensive Human Validation)
# --------------------------------------------------------------
print("\n--- HUMAN VALIDATION REPORT ---")
from config import DATA_DIR
import glob

validation_datasets = {}
search_pattern = os.path.join(DATA_DIR, "*_data.csv")
files = glob.glob(search_pattern)

train_file_base = os.path.basename(POSITIVE_DATA_FILE)

for filename in files:
    base_name = os.path.basename(filename)
    if base_name == "imposter_data.csv":
        continue
        
    persona_name = base_name.replace("_data.csv", "").capitalize()
    
    if base_name == train_file_base:
        expected = 1
        display_name = f"{persona_name} (Owner)"
    else:
        expected = 0
        display_name = persona_name

    validation_datasets[display_name] = {"file": filename, "expected": expected}

print(f"{'User Name':<20} | {'Total':<8} | {'Approved':<10} | {'Rejected':<10} | {'Pass Rate':<12}")
print("-" * 75)

for name, info in validation_datasets.items():
    csv_file = info["file"]
    
    if os.path.exists(csv_file):
        try:
            df_val = pd.read_csv(csv_file)
            X_val = df_val[feature_cols]
            X_val_scaled = scaler.transform(X_val)
            preds_val = model.predict(X_val_scaled)
            
            total = len(preds_val)
            approved = np.sum(preds_val == 1)
            rejected = np.sum(preds_val == 0)
            pass_rate = (approved / total) * 100 if total > 0 else 0
            
            print(f"{name:<20} | {total:<8} | {approved:<10} | {rejected:<10} | {pass_rate:6.2f}%")
            
        except Exception as e:
            print(f"{name:<20} | ERROR: {str(e)}")

print("-" * 75)

# 9. SAVE
# -------
with open(OUTPUT_MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to '{OUTPUT_MODEL_FILE}'")

with open(OUTPUT_SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to '{OUTPUT_SCALER_FILE}'")

# 10. VISUALIZE
# ------------
print("\nGenerating visualization...")
visualize.visualize()