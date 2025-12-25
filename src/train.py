import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. LOAD POSITIVE DATA (YOU)
# ---------------------------
if not os.path.exists("keystroke_data.csv"):
    print("Error: 'keystroke_data.csv' not found. Record your data first!")
    exit()

real_user = pd.read_csv("keystroke_data.csv")
real_user['label'] = 1 
print(f"Loaded {len(real_user)} samples from YOU.")

# 2. LOAD NEGATIVE DATA (IMPOSTERS)
# ---------------------------------
# Strategy: Prefer Real Imposters. Fallback to Synthetic Randomness.

if os.path.exists("imposter_data.csv"):
    print("Found REAL imposter data. Using it.")
    fake_user = pd.read_csv("imposter_data.csv")
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP: 32 neurons -> 16 neurons
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)

print("\nTraining Neural Network...")
model.fit(X_train, y_train)

# 4. TEST
# -------
predictions = model.predict(X_test)
print("\n--- RESULTS ---")
print(classification_report(y_test, predictions, target_names=["Imposter", "Real User"]))

# 5. SAVE
# -------
with open("auth_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to 'auth_model.pkl'")