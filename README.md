# StrokeSec

StrokeSec is a Keystroke Dynamics Authentication System that uses typing patterns (dwell time and flight time) to distinguish between a real user and an imposter.

## Features
- **Data Collection**: Records typing patterns for the phrase *"the quick brown fox"*.
- **Data Analysis**: Visualizes your typing "fingerprint" (muscle memory vs. rhythm).
- **Machine Learning**: Trains a neural network to authenticate users based on their unique typing biometrics.
- **Login Verification**: Real-time biometric authentication system.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires `sudo` for keyboard hooking on Linux.*

2. **Collect Data** (Train the system)
   ```bash
   sudo python3 src/data_collector.py
   ```
   - Type the target phrase 20-30 times.
   - It will save to `keystroke_data.csv`.

3. **Generate Imposter Data**
   ```bash
   python3 src/generate_imposters.py
   ```
   - Creates synthetic "bad guys" to train the AI against.

4. **Train the Model**
   ```bash
   python3 src/train.py
   ```
   - Trains a Neural Network (MLP) on your data vs. the imposters.
   - Saves the brain to `auth_model.pkl`.

5. **Test Authentication**
   ```bash
   sudo python3 src/login.py
   ```
   - Type the phrase. The system will grant or deny access based on your rhythm.

6. **Visualize Your Pattern**
   ```bash
   python3 src/visualize.py
   ```
   - Generates `typing_pattern.png` showing your dwell and flight time distributions.

## Project Structure
- `src/capture.py`: Shared logic for keyboard hooking and timing.
- `src/config.py`: Central configuration for constants and file paths.
- `src/data_collector.py`: Tool for recording training data.
- `src/generate_imposters.py`: Generates synthetic negative samples.
- `src/login.py`: The actual authentication mechanism.
- `src/train.py`: Model training script using Scikit-Learn.
- `src/visualize.py`: Data visualization tool.
