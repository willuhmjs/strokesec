"""
Batch verification script for external user data against the trained authentication model.
"""
import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from config import MODEL_FILE, SCALER_FILE, REQUIRED_LENGTH

def load_model_and_scaler(model_path, scaler_path):
    """Loads the trained authentication model and scaler."""
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train_model.py first!")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} not found. Run train_model.py first!")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    return model, scaler

def verify_external_data(csv_path, model, scaler):
    """
    Loads external CSV data and verifies each row against the model.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)
        
    print(f"Loading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Validate columns
    # Expected format is k0_dwell, k0_flight, ..., k19_dwell, k19_flight
    # Check if we have the right number of columns
    expected_cols = REQUIRED_LENGTH * 2
    if len(df.columns) != expected_cols:
        print(f"Warning: Expected {expected_cols} columns, found {len(df.columns)}.")
        print("Attempting to verify anyway if columns match feature pattern...")

    # Statistics counters
    total_attempts = len(df)
    approved_count = 0
    rejected_count = 0
    
    print("\n" + "="*60)
    print(f"{'ROW':<5} | {'STATUS':<10} | {'CONFIDENCE':<10} | {'NOTE'}")
    print("-" * 60)

    results = []

    for index, row in df.iterrows():
        # Prepare single sample for prediction
        # The model expects a DataFrame with the same feature columns
        sample = pd.DataFrame([row])
        
        # Scale the input
        sample_scaled = scaler.transform(sample)
        
        # Predict
        # Class 1 = Real User, Class 0 = Imposter
        try:
            proba = model.predict_proba(sample_scaled)[0]
            confidence = proba[1]  # Probability of being the real user
            
            # Threshold from login.py is 0.85
            is_approved = confidence > 0.85
            
            status = "APPROVED" if is_approved else "REJECTED"
            if is_approved:
                approved_count += 1
            else:
                rejected_count += 1
                
            print(f"{index+1:<5} | {status:<10} | {confidence*100:6.2f}%    |")
            
            results.append({
                'row_index': index,
                'status': status,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"{index+1:<5} | ERROR      | N/A        | {str(e)}")

    print("-" * 60)
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Total Attempts: {total_attempts}")
    print(f"Approved:       {approved_count} ({(approved_count/total_attempts)*100:.1f}%)")
    print(f"Rejected:       {rejected_count} ({(rejected_count/total_attempts)*100:.1f}%)")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch verify external keystroke data CSV.")
    parser.add_argument("csv_file", help="Path to the external CSV file to verify")
    parser.add_argument("--model", default=MODEL_FILE, help="Path to the trained model file")
    parser.add_argument("--scaler", default=SCALER_FILE, help="Path to the trained scaler file")
    args = parser.parse_args()

    # Load model
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    
    # Run verification
    verify_external_data(args.csv_file, model, scaler)
