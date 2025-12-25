import shutil
import os
import pandas as pd
from generate_imposters import generate_persona_data

def migrate():
    if os.path.exists("keystroke_data.csv"):
        # Check if it's already compatible
        df = pd.read_csv("keystroke_data.csv")
        if len(df.columns) == 40: # 20 keys * 2 columns
            print("keystroke_data.csv is already compatible.")
            return

        print("Backing up incompatible keystroke_data.csv to keystroke_data.old.csv...")
        shutil.move("keystroke_data.csv", "keystroke_data.old.csv")

    print("Generating temporary valid keystroke_data.csv (Real User placeholder)...")
    # specific real user behavior usually differs, but this is just to get the pipeline working
    data = generate_persona_data("Average Joe", num_samples=20) 
    df = pd.DataFrame(data)
    df.to_csv("keystroke_data.csv", index=False)
    print("Done. New keystroke_data.csv created with 20 samples.")

if __name__ == "__main__":
    migrate()
