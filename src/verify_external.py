"""
Batch verification script for external user data against the trained authentication model.
"""
import sys
import argparse
import pandas as pd
import numpy as np
from config import MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from utils import load_artifacts, check_columns

def verify_external_data(csv_path, model, scaler, threshold):
    """
    Loads external CSV data and verifies each row against the model.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Use shared utility to validate/fix columns
    df = check_columns(df)

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
    parser.add_argument("--model", default=str(MODEL_FILE), help="Path to the trained model file")
    parser.add_argument("--scaler", default=str(SCALER_FILE), help="Path to the trained scaler file")
    parser.add_argument("--threshold", default=str(THRESHOLD_FILE), help="Path to the threshold file")
    args = parser.parse_args()

    # Load model
    model, scaler, threshold = load_artifacts(args.model, args.scaler, args.threshold)
    
    # Run verification
    verify_external_data(args.csv_file, model, scaler, threshold)
