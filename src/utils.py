"""
Shared utility functions for the Keystroke Dynamics Authentication System.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from config import REQUIRED_LENGTH

def load_artifacts(model_path, scaler_path, threshold_path):
    """
    Loads the trained authentication model, scaler, and threshold.
    
    Args:
        model_path (str): Path to the pickled model file.
        scaler_path (str): Path to the pickled scaler file.
        threshold_path (str): Path to the pickled threshold file.
        
    Returns:
        tuple: (model, scaler, threshold)
    """
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train.py first!")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found. Run train.py first!")
        sys.exit(1)
    if not os.path.exists(threshold_path):
        print(f"Error: {threshold_path} not found. Run train.py first!")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)
        
    return model, scaler, threshold

def augment_data(X, num_copies=15, noise_scale=0.05):
    """
    Generates synthetic samples by adding Gaussian noise to real samples.
    
    Args:
        X (pd.DataFrame): The original dataset.
        num_copies (int): Number of noisy copies to create per sample.
        noise_scale (float): Standard deviation multiplier for the noise.
        
    Returns:
        pd.DataFrame: Augmented dataset containing original + synthetic samples.
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

def check_columns(df):
    """
    Validates and attempts to fix column headers for keystroke data.
    """
    expected_cols = REQUIRED_LENGTH * 3
    if len(df.columns) != expected_cols:
        print(f"Warning: Expected {expected_cols} columns, found {len(df.columns)}.")
        print("Attempting to verify anyway if columns match feature pattern...")
        
    # Ensure correct column names if they are mismatched but count is correct
    if len(df.columns) == expected_cols:
        expected_headers = []
        for i in range(REQUIRED_LENGTH):
            expected_headers.extend([f"k{i}_hold", f"k{i}_ud", f"k{i}_dd"])
            
        if list(df.columns) != expected_headers:
             print("Adjusting column headers to match model expectation...")
             df.columns = expected_headers
    
    return df