# StrokeSec Upgrade Plan: Autoencoder & Advanced Features

This document outlines the technical specifications for upgrading StrokeSec from a binary classifier to an anomaly detection system (Autoencoder) with enhanced feature engineering and data augmentation.

## 1. Feature Engineering (Level 1)
**Goal:** Capture and utilize richer timing data: Hold Time (H), Up-Down Flight (UD), and Down-Down Flight (DD).

### 1.1 Data Structure Updates
*   **Current:** `dwell` (Hold), `flight` (UD).
*   **New:**
    *   `hold`: Time between Press and Release of the same key.
    *   `ud_flight`: Time between Release of Key(N-1) and Press of Key(N).
    *   `dd_flight`: Time between Press of Key(N-1) and Press of Key(N).
*   **Redundancy Note:** `DD = UD + Hold(N-1)`. We will include all three to explicitly model rhythm.

### 1.2 File: `src/capture.py`
*   **Update `KeystrokeCapture.on_key_event`**:
    *   Track `press_ts` explicitly in `current_record`.
    *   Calculate `dd_flight` using `press_ts` of current key and `press_ts` of previous key.
*   **Output Format**: List of dicts: `{'key': k, 'hold': h, 'ud': ud, 'dd': dd, 'press_ts': t}`.

### 1.3 File: `src/data_collector.py`
*   **Update Saving Logic**:
    *   Flatten the new fields into the CSV.
    *   Columns: `k0_hold, k0_ud, k0_dd, k1_hold, ...`
    *   Expected columns per row: `3 * REQUIRED_LENGTH`.

---

## 2. Autoencoder Architecture (Level 2)
**Goal:** Replace Binary Classification (User vs. Imposter) with Anomaly Detection (User vs. Deviation). The model learns to reconstruct the user's typing pattern; high reconstruction error indicates an imposter.

### 2.1 Model Specification
*   **Type:** `sklearn.neural_network.MLPRegressor` (Multi-Layer Perceptron Regressor).
    *   We use a Regressor because the output is continuous (reconstructed features), not a class label.
*   **Architecture (Bottleneck):**
    *   **Input Layer:** Size `N` (3 * Required Length, e.g., 60 features).
    *   **Hidden Layers:** `[N/2, N/4, N/2]` (e.g., `[30, 15, 30]`). This forces the model to learn latent patterns.
    *   **Output Layer:** Size `N` (Same as Input).
*   **Loss Function:** Mean Squared Error (MSE), handled internally by `MLPRegressor`.

### 2.2 Threshold Strategy
*   **Training Phase:**
    1.  Train on User Data (Real + Fuzzy).
    2.  Pass validation set (Real User Data) through the trained model.
    3.  Calculate MSE for each validation sample.
    4.  **Threshold Calculation:** `Threshold = Mean(MSE) + 2 * StdDev(MSE)`.
*   **Storage:** Save `threshold` alongside the model and scaler (e.g., in a pickle file or a config object).

---

## 3. Data Strategy (Level 3)
**Goal:** Solve data scarcity for single-class training using "Fuzzy User" augmentation.

### 3.1 Fuzzy Augmentation (in `src/train.py`)
*   **Logic:** Since we only train on positive data, we need more variance to prevent overfitting to the exact few recorded samples.
*   **Implementation:**
    *   For each real sample:
        *   Generate 10-20 "Fuzzy" copies.
        *   Add Gaussian Noise: `New_Value = Old_Value + Noise`.
        *   `Noise ~ Normal(0, scale)`.
        *   `scale`: 5-10% of the feature's standard deviation across the dataset.
*   **Training Set:** `Real Data + Fuzzy Data`.

---

## 4. Implementation Plan (Files)

### Phase 1: Data Capture (`src/capture.py`, `src/data_collector.py`)
1.  Modify `KeystrokeCapture` to calculate `dd_flight`.
2.  Update `data_collector.py` to save new CSV format.
3.  **Action:** User must re-record data after this update because old data will lack `dd_flight`.

### Phase 2: Training Logic (`src/train.py`)
1.  **Remove** `generate_imposters.py` dependency (or move to validation only).
2.  **Add** `augment_data(X)` function for Fuzzy User generation.
3.  **Replace** `MLPClassifier` with `MLPRegressor`.
4.  **Change** Labels: Target `y` is now `X` (Autoencoder), not `0/1`.
5.  **Compute Threshold**: Calculate and save the reconstruction error threshold.

### Phase 3: Inference (`src/login.py`)
1.  Load Model, Scaler, and Threshold.
2.  Capture input (using updated `capture.py`).
3.  Scale input.
4.  **Predict**: `X_reconstructed = model.predict(X_scaled)`.
5.  **Error**: `MSE = mean_squared_error(X_scaled, X_reconstructed)`.
6.  **Decision**:
    *   `MSE <= Threshold` -> **Access Granted**.
    *   `MSE > Threshold` -> **Access Denied**.
