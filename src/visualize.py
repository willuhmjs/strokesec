import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fix for headless servers (SSH)
matplotlib.use('Agg') 

def visualize(filename="keystroke_data.csv", output="typing_pattern.png"):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    print(f"Generating graph from {filename}...")
    df = pd.read_csv(filename)

    plt.figure(figsize=(14, 6))

    # Plot 1: Dwell
    plt.subplot(1, 2, 1)
    dwell_cols = [c for c in df.columns if 'dwell' in c]
    sns.boxplot(data=df[dwell_cols])
    plt.title("Dwell Time Pattern (Muscle Memory)")
    plt.xticks(rotation=45)
    plt.ylabel("Seconds")

    # Plot 2: Flight
    plt.subplot(1, 2, 2)
    flight_cols = [c for c in df.columns if 'flight' in c]
    sns.boxplot(data=df[flight_cols])
    plt.title("Flight Time Pattern (Rhythm)")
    plt.xticks(rotation=45)
    plt.ylabel("Seconds")
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output)
    print(f"Graph saved to {output}. Download to view.")

if __name__ == "__main__":
    visualize()