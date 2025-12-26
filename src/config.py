"""
Shared configuration and constants for the Keystroke Dynamics Authentication System.
"""
from pathlib import Path

# Data Configuration
TARGET_PHRASE = "the quick brown fox"
REQUIRED_LENGTH = len(TARGET_PHRASE) + 1  # Phrase chars + 1 enter

# File Paths
# Use pathlib for more robust cross-platform path handling
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

KEYSTROKE_DATA_FILE = DATA_DIR / "keystroke_data.csv"
IMPOSTER_DATA_FILE = DATA_DIR / "imposter_data.csv"
MODEL_FILE = DATA_DIR / "auth_model.pkl"
SCALER_FILE = DATA_DIR / "auth_scaler.pkl"
THRESHOLD_FILE = DATA_DIR / "auth_threshold.pkl"
VISUALIZATION_FILE = DATA_DIR / "typing_pattern.png"
