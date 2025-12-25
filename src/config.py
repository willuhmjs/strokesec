"""
Shared configuration and constants for the Keystroke Dynamics Authentication System.
"""
import os

# Data Configuration
TARGET_PHRASE = "the quick brown fox"
REQUIRED_LENGTH = 20  # 19 chars + 1 enter

# File Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KEYSTROKE_DATA_FILE = os.path.join(DATA_DIR, "keystroke_data.csv")
IMPOSTER_DATA_FILE = os.path.join(DATA_DIR, "imposter_data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "auth_model.pkl")
SCALER_FILE = os.path.join(DATA_DIR, "auth_scaler.pkl")
VISUALIZATION_FILE = os.path.join(DATA_DIR, "typing_pattern.png")
