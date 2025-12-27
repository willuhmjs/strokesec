# StrokeSec

StrokeSec is a behavioral biometric authentication system. It uses keystroke dynamics—specifically **dwell time** (key press duration) and **flight time** (interval between presses)—to verify user identity.

It utilizes a **Deep Autoencoder** Neural Network for anomaly detection. Instead of training on "User vs. Imposter", it trains *only* on the user's data to learn their unique pattern. Any deviation (high reconstruction error) is flagged as an anomaly.

## Features

*   **Deep Autoencoder Architecture:**
    *   Dynamic layer sizing based on input dimension (Input → 80% → 60% → 40% Bottleneck).
    *   **LeakyReLU** activation for better gradient flow.
    *   **Batch Normalization** for training stability.
    *   **Dropout** layers to prevent overfitting on small datasets.
*   **Robust Training Pipeline:**
    *   **PyTorch** implementation.
    *   **Fuzzy Augmentation:** Automatically generates synthetic variations of typing data.
    *   **Early Stopping** & **Learning Rate Scheduling:** Prevents overfitting and ensures optimal convergence.
    *   **Dynamic Thresholding:** Calculates strict acceptance thresholds based on validation loss statistics (Mean + 1.5 StdDev).

## Setup

```bash
# create/activate venv
python3 -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt
```

## Usage

**Note:** Linux requires `sudo` access to hook into raw keyboard events via `evdev`.

### 1. Data Collection

Record your typing baseline. The system defaults to the phrase *"the quick brown fox"*.

```bash
sudo python3 src/data_collector.py
```

*   Follow the prompts to type the phrase 20+ times.
*   Data is saved to `data/keystroke_data.csv`.

### 2. Train Model

Train the Anomaly Detection Model using the PyTorch pipeline.

```bash
python3 src/train.py
```

*   **Input:** `data/keystroke_data.csv`
*   **Process:** Augments data, scales inputs, trains the Deep Autoencoder, and calculates the anomaly threshold.
*   **Output:** `data/auth_model.pkl` (Model Weights), `data/auth_scaler.pkl`, `data/auth_threshold.pkl`.

### 3. Authenticate

Run the real-time verification loop.

```bash
sudo python3 src/login.py
```

*   The system compares your live typing against the learned model.
*   **Low Error:** Access Granted.
*   **High Error:** Access Denied (Anomaly Detected).

### 4. Visualization (Optional)

Generate a plot comparing your dwell/flight times against a theoretical average.

```bash
python3 src/visualize.py
```

*   Output: `data/typing_pattern.png`

## Key Modules

| File | Description |
| --- | --- |
| `src/capture.py` | Handles low-level keyboard hooking and timestamp extraction. |
| `src/data_collector.py` | CLI tool for building the positive dataset. |
| `src/train.py` | PyTorch training script with Early Stopping, Scheduler, and Fuzzy Augmentation. |
| `src/model.py` | Defines the `KeystrokeAutoencoder` class. |
| `src/login.py` | Loads the model and performs real-time anomaly detection. |
| `src/config.py` | Global constants (target phrase, file paths, model params). |
| `src/utils.py` | Helper functions for data augmentation and processing. |
| `web_collector/` | (Optional) HTML interface for browser-based data collection. |

## Note on Dependencies

This project relies on `evdev` (Linux) or `keyboard` (Windows/Root) for global hooks. Ensure your Python environment has access to input devices.
