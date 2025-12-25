import pandas as pd
import numpy as np
import os

# Configuration matching data_collector.py
TARGET_PHRASE = "the quick brown fox"
REQUIRED_LENGTH = 20 # 19 chars + 1 enter
OUTPUT_FILE = "imposter_data.csv"

def generate_persona_data(persona_name, num_samples=50):
    data = []
    
    print(f"Generating {num_samples} samples for persona: {persona_name}")
    
    for _ in range(num_samples):
        row = {}
        for i in range(REQUIRED_LENGTH):
            # Base stats per persona
            if persona_name == "Turtle":
                # Slow typer: high dwell, long flight
                dwell = np.random.normal(0.2, 0.05)
                flight = np.random.normal(0.3, 0.1)
            
            elif persona_name == "Rabbit":
                # Fast typer: short dwell, short flight
                dwell = np.random.normal(0.08, 0.02)
                flight = np.random.normal(0.05, 0.03)
                
            elif persona_name == "Stumbler":
                # Inconsistent: mixture of fast and slow
                if np.random.random() > 0.8:
                    dwell = np.random.uniform(0.3, 0.6) # Hesitation
                    flight = np.random.uniform(0.2, 0.8)
                else:
                    dwell = np.random.normal(0.12, 0.03)
                    flight = np.random.normal(0.15, 0.05)
            
            else: # "Average Joe"
                dwell = np.random.normal(0.12, 0.03)
                flight = np.random.normal(0.12, 0.05)

            # Ensure non-negative
            row[f'k{i}_dwell'] = max(0.01, dwell)
            row[f'k{i}_flight'] = max(0.0, flight)
            
        data.append(row)
        
    return data

def main():
    personas = ["Turtle", "Rabbit", "Stumbler", "Average Joe"]
    all_data = []
    
    for p in personas:
        all_data.extend(generate_persona_data(p, num_samples=30))
        
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    if os.path.exists(OUTPUT_FILE):
        print(f"Overwriting existing {OUTPUT_FILE}...")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated {len(df)} synthetic imposter records in {OUTPUT_FILE}")
    print(f"Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
