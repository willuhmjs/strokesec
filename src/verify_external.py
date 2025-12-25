"""
Batch verification script for external user data against the trained authentication model.
"""
import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from config import MODEL_FILE, SCALER_FILE, THRESHOLD_FILE, REQUIRED_LENGTH

def load_artifacts(model_path, scaler_path, threshold_path):
    """Loads the trained authentication model, scaler, and threshold."""
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

def verify_external_data(csv_path, model, scaler, threshold):
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
    # Expected format is k0_hold, k0_ud, k0_dd, ..., k19_hold, k19_ud, k19_dd
    expected_cols = REQUIRED_LENGTH * 3
    if len(df.columns) != expected_cols:
        print(f"Warning: Expected {expected_cols} columns, found {len(df.columns)}.")
        print("Attempting to verify anyway if columns match feature pattern...")
        
    # Ensure correct column names if they are mismatched but count is correct
    if len(df.columns) == expected_cols:
         # Generate expected headers
        expected_headers = []
        for i in range(REQUIRED_LENGTH):
            expected_headers.extend([f"k{i}_hold", f"k{i}_ud", f"k{i}_dd"])
            
        # If headers are different (e.g. from web collector might be slightly off or old format), rename them
        if list(df.columns) != expected_headers:
             print("Adjusting column headers to match model expectation...")
             df.columns = expected_headers

    # Statistics counters
    total_attempts = len(df)
    approved_count = 0
    rejected_count = 0
    
    print("\n" + "="*75)
    print(f"{'ROW':<5} | {'STATUS':<15} | {'MSE':<12} | {'THRESHOLD':<12} | {'NOTE'}")
    print("-" * 75)

    results = []

    for index, row in df.iterrows():
        # Prepare single sample for prediction
        # The model expects a DataFrame with the same feature columns
        sample = pd.DataFrame([row])
        
        try:
            # Scale the input
            sample_scaled = scaler.transform(sample)
            
            # Reconstruct (Predict)
            sample_reconstructed = model.predict(sample_scaled)
            
            # Calculate MSE
            mse = np.mean(np.power(sample_scaled - sample_reconstructed, 2))
            
            # Decision
            is_approved = mse <= threshold
            
            status = "APPROVED" if is_approved else "REJECTED"
            if is_approved:
                approved_count += 1
            else:
                rejected_count += 1
            
            # Visual indicator
            status_display = f"{'✅ ' + status if is_approved else '⛔ ' + status}"

            print(f"{index+1:<5} | {status_display:<15} | {mse:<12.6f} | {threshold:<12.6f} |")
            
            results.append({
                'row_index': index,
                'status': status,
                'mse': mse
            })
            
        except Exception as e:
            print(f"{index+1:<5} | ERROR           | N/A          | {threshold:<12.6f} | {str(e)}")

    print("-" * 75)
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
    parser.add_argument("--threshold", default=THRESHOLD_FILE, help="Path to the threshold file")
    args = parser.parse_args()

    # Load model
    model, scaler, threshold = load_artifacts(args.model, args.scaler, args.threshold)
    
    # Run verification
    verify_external_data(args.csv_file, model, scaler, threshold)

