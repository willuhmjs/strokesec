"""
Shared utility functions for the Keystroke Dynamics Authentication System.
"""
import os
import sys
import pickle
import torch
import pandas as pd
import numpy as np
from model import KeystrokeAutoencoder # Import the class definition
from config import REQUIRED_LENGTH

def load_artifacts(model_path, scaler_path, threshold_path):
    """
    Loads PyTorch model, scaler, and threshold.
    Returns: (model, scaler, threshold)
    """
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)
        
    # Load Scaler and Threshold
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(threshold_path, "rb") as f:
        threshold = pickle.load(f)

    # Load PyTorch Model
    try:
        checkpoint = torch.load(model_path)
        input_dim = checkpoint['input_dim']
        
        model = KeystrokeAutoencoder(input_dim)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval() # Set to evaluation mode immediately
    except Exception as e:
        print(f"Failed to load PyTorch model: {e}")
        sys.exit(1)
        
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

def select_user_profile(keystroke_file, imposter_file, data_dir):
    """
    Interactive helper to select which user profile to record data for.
    """
    print("\n" + "="*30)
    print("   KEYSTROKE DATA COLLECTOR   ")
    print("="*30)
    print("Who is typing right now?")
    print(f"1. Primary User -> {keystroke_file.name}")
    print(f"2. Imposter -> {imposter_file.name}")
    print("3. Custom Name (Creates new synced data file)")

    choice = input("\nSelect Option (1-3): ")

    if choice == '1':
        return keystroke_file
    if choice == '2':
        return imposter_file
    if choice == '3':
        name = input("Enter your name (e.g. 'alice'): ").strip().lower()
        # Sanitize filename simple check
        if not name.isalnum():
             print("Invalid name. Using 'custom_user'")
             name = "custom_user"
        return data_dir / f"{name}_data.csv"
    
    # Fallback/Default
    print("Invalid choice. Defaulting to keystroke_data.csv")
    return keystroke_file

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