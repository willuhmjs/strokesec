# StrokeSec

StrokeSec is a behavioral biometric authentication system. It uses keystroke dynamics—specifically **dwell time** (key press duration) and **flight time** (interval between presses)—to verify user identity.

It uses a Multi-Layer Perceptron (MLP) for binary classification (User vs. Imposter).

## Setup

```bash
# create/activate venv
python3 -m venv venv
source venv/bin/activate

# install deps (scikit-learn, pandas, matplotlib, keyboard/evdev)
pip install -r requirements.txt

```

## Usage

**Note:** Linux requires `sudo` access to hook into raw keyboard events via `evdev`.

### 1. Data Collection

Record your typing baseline. The system defaults to the phrase *"the quick brown fox..."*.

```bash
sudo python3 src/data_collector.py

```

* Follow the prompts to type the phrase 20+ times.
* Data is saved to `data/<username>_data.csv`.

### 2. Generate Negatives

Create synthetic data to train the model on what *isn't* you.

```bash
python3 src/generate_imposters.py

```

* Generates `data/imposter_data.csv` based on statistical averages outside your range.

### 3. Train Model

Train the MLP Classifier.

```bash
python3 src/train.py

```

* Vectorizes the CSV data and trains the scaler/model.
* Artifacts saved: `data/auth_model.pkl` and `data/auth_scaler.pkl`.

### 4. Authenticate

Run the real-time verification loop.

```bash
sudo python3 src/login.py

```

### 5. Visualization (Optional)

Generate a plot comparing your dwell/flight times against the imposter dataset.

```bash
python3 src/visualize.py

```

* Output: `data/typing_pattern.png`

## Key Modules

| File | Description |
| --- | --- |
| `src/capture.py` | Handles low-level keyboard hooking and timestamp extraction. |
| `src/data_collector.py` | CLI tool for building the positive dataset. |
| `src/train.py` | Loads CSVs, scales features, and trains the Scikit-Learn MLP. |
| `src/login.py` | Loads the pickle files and performs inference on live input. |
| `src/config.py` | Global constants (target phrase, file paths, model params). |
| `web_collector/` | (Optional) HTML interface for browser-based data collection. |

### Note on Dependencies

This project relies on `evdev` (Linux) or `keyboard` (Windows/Root) for global hooks. Ensure your Python environment has access to input devices.