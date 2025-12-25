"""
Script to generate synthetic imposter data for training.
"""
import os
import pandas as pd
import numpy as np
from config import REQUIRED_LENGTH, IMPOSTER_DATA_FILE

def analyze_user_data(file_path):
    """
    Reads a user's keystroke data CSV and calculates the mean and standard deviation
    for dwell and flight times for each key index.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping.")
        return None

    stats = {}
    for col in df.columns:
        if col == "id": # Skip non-feature columns if any
            continue
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std()
        }
    return stats

def generate_persona_data(persona_name, stats, num_samples=50):
    """
    Generates synthetic keystroke data based on a user's statistical profile.
    """
    data = []
    print(f"Generating {num_samples} samples for persona: {persona_name}")
    
    for _ in range(num_samples):
        row = {}
        # Iterate through the columns expected in the dataset (k0_dwell, k0_flight, etc.)
        # We assume the stats dictionary contains keys for all these features.
        
        # We need to know the structure of features. 
        # Based on previous files, it seems to be k0_dwell, k0_flight, ... k19_flight
        
        for i in range(REQUIRED_LENGTH):
            dwell_key = f'k{i}_dwell'
            flight_key = f'k{i}_flight'
            
            # Retrieve stats for this key, default to global average if missing (safety net)
            # Using defaults similar to original "Average Joe" if specific key data missing
            
            if dwell_key in stats:
                dwell_mean = stats[dwell_key]['mean']
                dwell_std = stats[dwell_key]['std']
            else:
                dwell_mean = 0.12
                dwell_std = 0.03

            if flight_key in stats:
                flight_mean = stats[flight_key]['mean']
                flight_std = stats[flight_key]['std']
            else:
                flight_mean = 0.12
                flight_std = 0.05
            
            # Generate value using normal distribution
            dwell = np.random.normal(dwell_mean, dwell_std)
            flight = np.random.normal(flight_mean, flight_std)
            
            # Ensure logical bounds (dwell > 0, flight can be negative but let's keep it reasonable or as is)
            # The original script had max(0.01, dwell) and max(0.0, flight).
            # Flight time can be negative (rollover typing), so we shouldn't strictly cap it at 0 unless we know for sure.
            # However, checking the user data provided, there are negative flight times.
            # So we will only clamp dwell time to be positive.
            
            row[dwell_key] = max(0.01, dwell)
            row[flight_key] = flight 
            
        data.append(row)
        
    return data

def main():
    """Main execution function."""
    
    # Auto-detect real user files in the data directory
    from config import DATA_DIR
    import glob

    all_data = []
    
    # Find all CSV files ending in _data.csv in the data directory
    # Exclude imposter_data.csv and keystroke_data.csv if they exist there and shouldn't be treated as source profiles
    search_pattern = os.path.join(DATA_DIR, "*_data.csv")
    files = glob.glob(search_pattern)
    
    for filename in files:
        base_name = os.path.basename(filename)
        # Skip output files or the main training file if it shouldn't be used as a persona source
        if base_name in ["imposter_data.csv", "keystroke_data.csv"]:
            continue
            
        persona = base_name.replace("_data.csv", "").capitalize()
        
        print(f"Analyzing data for {persona} from {filename}...")
        stats = analyze_user_data(filename)
        if stats:
            # Generate synthetic data based on this user's stats
            synthetic_data = generate_persona_data(persona, stats, num_samples=50)
            all_data.extend(synthetic_data)

    if not all_data:
        print("No data generated. Please check input files.")
        return

    df = pd.DataFrame(all_data)
    
    output_file = IMPOSTER_DATA_FILE
    
    if os.path.exists(output_file):
        print(f"Overwriting existing {output_file}...")
    
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {len(df)} synthetic imposter records in {output_file}")
    print(f"Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
