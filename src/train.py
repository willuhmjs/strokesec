"""
Training script for the Keystroke Dynamics Authentication model (Autoencoder).
"""
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
from config import KEYSTROKE_DATA_FILE, MODEL_FILE, SCALER_FILE, THRESHOLD_FILE

# 0. PARSE ARGUMENTS
# ------------------
parser = argparse.ArgumentParser(description="Train the Keystroke Dynamics Autoencoder.")
parser.add_argument("input_file", nargs="?", default=KEYSTROKE_DATA_FILE, help="Path to the positive user data CSV (default: from config).")
parser.add_argument("--model", default=MODEL_FILE, help="Path to save the trained model (default: from config).")
parser.add_argument("--scaler", default=SCALER_FILE, help="Path to save the scaler (default: from config).")
parser.add_argument("--threshold", default=THRESHOLD_FILE, help="Path to save the threshold (default: from config).")
args = parser.parse_args()

if __name__ == "__main__":
    POSITIVE_DATA_FILE = args.input_file
    OUTPUT_MODEL_FILE = args.model
    OUTPUT_SCALER_FILE = args.scaler
    OUTPUT_THRESHOLD_FILE = args.threshold

    print(f"Training on: {POSITIVE_DATA_FILE}")
    print(f"Saving model to: {OUTPUT_MODEL_FILE}")
    print(f"Saving scaler to: {OUTPUT_SCALER_FILE}")
    print(f"Saving threshold to: {OUTPUT_THRESHOLD_FILE}")

    # 1. LOAD POSITIVE DATA
    # ---------------------
    if not os.path.exists(POSITIVE_DATA_FILE):
        print(f"Error: '{POSITIVE_DATA_FILE}' not found. Record your data first!")
        sys.exit(1)

    real_user = pd.read_csv(POSITIVE_DATA_FILE)
    print(f"Loaded {len(real_user)} samples from User.")

    # Identify feature columns (all columns in the new CSV structure are features)
    # Note: The new structure does NOT have a 'label' column in the CSV.
    feature_cols = list(real_user.columns)
    X_real = real_user[feature_cols]

    # 2. SPLIT DATA
    # -------------
    # We split real data into Train (for model learning) and Test (for threshold calculation/validation)
    X_train_raw, X_test_raw = train_test_split(X_real, test_size=0.2, random_state=42)

    # 3. FUZZY AUGMENTATION (Level 3)
    # -------------------------------
    def augment_data(X, num_copies=15, noise_scale=0.05):
        """
        Generates synthetic samples by adding Gaussian noise to real samples.
        """
        augmented_data = []
        # Calculate std dev per feature to scale noise appropriately
        feature_stds = X.std(axis=0)
        # Replace 0 stds with a small value to avoid errors if a feature is constant
        feature_stds[feature_stds == 0] = 1e-6 
        
        for _, row in X.iterrows():
            # Add original
            augmented_data.append(row.values)
            
            # Add copies
            for _ in range(num_copies):
                noise = np.random.normal(0, feature_stds * noise_scale)
                new_row = row.values + noise
                augmented_data.append(new_row)
                
        return pd.DataFrame(augmented_data, columns=X.columns)

    print(f"Augmenting data... (Original: {len(X_train_raw)})")
    X_train_augmented = augment_data(X_train_raw)
    print(f"Augmented Training Set Size: {len(X_train_augmented)}")

    # 4. SCALE
    # --------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_augmented)
    # Scale test data using the SAME scaler
    X_test = scaler.transform(X_test_raw)

    # 5. TRAIN AUTOENCODER (Level 2)
    # ------------------------------
    # Architecture: Input -> Enc -> Bottleneck -> Dec -> Output
    # Input Size = N
    # Hidden Layers = [N/2, N/4, N/2]
    n_features = X_train.shape[1]
    hidden_layers = (int(n_features / 2), int(n_features / 4), int(n_features / 2))

    print(f"Training Autoencoder with architecture: Input({n_features}) -> {hidden_layers} -> Output({n_features})")

    # MLPRegressor aims to predict the INPUT (identity function), minimizing reconstruction error
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=5000,
        random_state=42,
        alpha=0.0001 # L2 regularization
    )

    model.fit(X_train, X_train) # Target is X_train itself

    # 6. CALCULATE THRESHOLD
    # ----------------------
    print("\nCalculating Threshold on Validation Set...")
    # We use the REAL user validation set (X_test) to determine the acceptable error range.
    # We do NOT use augmented data here, as we want the threshold to be tight to the real user.

    X_test_reconstructed = model.predict(X_test)
    # Calculate MSE for each sample individually
    mse_scores = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)

    # Design Spec: Threshold = Mean + 2 * StdDev
    threshold = mean_mse + 2 * std_mse

    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Std MSE:  {std_mse:.6f}")
    print(f"Calculated Threshold: {threshold:.6f}")

    print("\nSimulating Inference on Validation Data:")
    print("-" * 30)
    approvals = 0
    denials = 0
    
    for i, mse in enumerate(mse_scores):
        status = "✅ ACCESS GRANTED" if mse <= threshold else "⛔ ACCESS DENIED"
        if mse <= threshold:
            approvals += 1
        else:
            denials += 1
        print(f"Sample {i+1}: MSE={mse:.6f} -> {status}")
        
    print("-" * 30)
    print(f"Validation Results: {approvals} Approved, {denials} Denied")
    if denials > 0:
        print(f"Note: {denials} legitimate samples were rejected (False Rejection). Consider increasing threshold multiplier if too high.")

    # 7. SAVE ARTIFACTS
    # -----------------
    with open(OUTPUT_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to '{OUTPUT_MODEL_FILE}'")

    with open(OUTPUT_SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to '{OUTPUT_SCALER_FILE}'")

    with open(OUTPUT_THRESHOLD_FILE, "wb") as f:
        pickle.dump(threshold, f)
    print(f"Threshold saved to '{OUTPUT_THRESHOLD_FILE}'")
