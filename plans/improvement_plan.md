# StrokeSec Improvement Plan

## Goal
Enhance the robustness of the autoencoder model and improve code quality for better maintainability and error handling.

## 1. Model Robustness Improvements (`src/model.py`, `src/train.py`)

### Current State
- **Architecture:** `Input -> Enc1(0.75) -> Bottleneck(0.5) -> Dec1(0.75) -> Output`.
- **Normalization:** `BatchNorm1d` in Encoder.
- **Activation:** `ReLU`.
- **Dropout:** 0.1 in Encoder.
- **Loss:** `MSELoss`.

### Proposed Changes
1.  **Enhance Architecture:** Add a second encoder/decoder layer to capture more complex non-linear relationships.
    - New Arch: `Input -> Enc1(0.8) -> Enc2(0.6) -> Bottleneck(0.4) -> Dec2(0.6) -> Dec1(0.8) -> Output`.
2.  **Activation Function:** Switch from `ReLU` to `LeakyReLU` (slope 0.1) to prevent "dying ReLU" problem where neurons become inactive.
3.  **Regularization:** Increase `Dropout` slightly (0.2) or apply it in more layers to further prevent overfitting on small datasets.
4.  **Optimizer:** Add `scheduler` (ReduceLROnPlateau) to `src/train.py` to adapt learning rate dynamically if validation loss plateaus.

## 2. Code Cleanup & Quality (`src/*.py`)

### 2.1 Refactoring & Organization
- **`src/capture.py`:** Add docstrings to all methods. Ensure consistent variable naming (`snake_case`).
- **`src/data_collector.py`:**
    - Move "Select Profile" logic to `src/utils.py` or a dedicated UI helper to keep the main script clean.
    - Improve the "Column Mismatch" logic to be more robust (e.g., attempt to migrate data instead of just backing up).
- **`src/config.py`:** Add type hints to constants where applicable.

### 2.2 Error Handling & Logging
- **Global:** Replace `print` statements with Python's `logging` module.
    - Create `src/logger.py` to configure a standard logger (console + file `logs/app.log`).
- **`src/login.py`:**
    - Wrap the main loop in a broader try/except to catch unexpected crashes and log the stack trace.
    - Add specific exception handling for model loading failures.

### 2.3 Dependency Management
- **`requirements.txt`:** Ensure all used libraries (torch, pandas, numpy, etc.) are pinned to stable versions.

## 3. Implementation Steps

1.  **Setup Logging:** Create `src/logger.py` and integrate it into `src/train.py`, `src/login.py`, and `src/data_collector.py`.
2.  **Refactor Model:** Update `KeystrokeAutoencoder` class in `src/model.py` with the deeper architecture and `LeakyReLU`.
3.  **Update Training:** Modify `src/train.py` to include `ReduceLROnPlateau` scheduler.
4.  **Refactor Collector:** Clean up `src/data_collector.py` by extracting UI logic.
5.  **Verify:** Run a full cycle: `data_collector` -> `train` -> `login` to ensure the new model architecture trains and infers correctly.

## 4. Verification Plan
- **Training Test:** Train on existing `will_data.csv`. Ensure Loss decreases and converges.
- **Inference Test:** Run `src/login.py` and verify it can load the new model structure and authenticate successfully.
