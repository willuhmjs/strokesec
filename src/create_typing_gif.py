import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import glob

# Configuration
DATA_DIR = Path("data")
OUTPUT_GIF = DATA_DIR / "typing_race.gif"
TARGET_PHRASE = "the quick brown fox"
# The phrase + enter = 20 keys usually. 
# We will visualize the characters of the phrase. 
# k0 corresponds to 't', k1 to 'h', etc.

def load_data():
    """
    Loads all user data from csv files in data/ directory.
    Returns a dictionary of {user_name: mean_row_series}
    """
    user_profiles = {}
    
    csv_files = list(DATA_DIR.glob("*_data.csv"))
    
    for fpath in csv_files:
        filename = fpath.name
        
        # Determine User Name
        if filename == "keystroke_data.csv":
            user_name = "Owner"
        elif filename == "imposter_data.csv":
            user_name = "Imposter"
        else:
            user_name = filename.replace("_data.csv", "").capitalize()
            
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            
            # Compute the mean profile (centroid) for this user
            # We only care about numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            mean_profile = df[numeric_cols].mean()
            user_profiles[user_name] = mean_profile
            print(f"Loaded profile for {user_name}")
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            
    return user_profiles

def reconstruct_timeline(profile):
    """
    Reconstructs the press times for each key based on the profile.
    Returns a list of (char, press_time, hold_time)
    """
    timeline = []
    
    # We assume columns are k0_hold, k0_ud, k0_dd, k1_hold...
    # We primarily use DD (Down-Down) time to establish the start of the next key relative to the previous.
    
    current_time = 0.0
    
    # Characters to map to keys. 
    # Note: k19 is usually Enter, which we might visualize as a symbol or ignore.
    chars = list(TARGET_PHRASE) + ["↵"] 
    
    for i in range(20): # 0 to 19
        prefix = f"k{i}"
        
        hold_col = f"{prefix}_hold"
        dd_col = f"{prefix}_dd"
        
        if hold_col not in profile:
            break
            
        hold_time = profile[hold_col]
        dd_time = profile[dd_col] if i > 0 else 0
        
        if i > 0:
            current_time += dd_time
            
        # Check boundaries (sometimes data might have weird neg values if not filtered, but we assume mean is valid)
        if i > 0 and dd_time < 0:
             # Fallback to UD if DD is weird? Or just accept logic.
             # dd = press_current - press_prev
             pass

        char = chars[i] if i < len(chars) else "?"
        
        timeline.append({
            "key_index": i,
            "char": char,
            "press_time": current_time,
            "release_time": current_time + hold_time,
            "hold_time": hold_time
        })
        
    return timeline

def create_animation(user_profiles):
    """
    Creates a matplotlib animation of the typing race.
    """
    users = list(user_profiles.keys())
    timelines = {u: reconstruct_timeline(user_profiles[u]) for u in users}
    
    # Find max duration to set x-axis
    max_time = 0
    for u in users:
        last_event = timelines[u][-1]
        if last_event['release_time'] > max_time:
            max_time = last_event['release_time']
            
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, len(users) * 1.5))
    
    ax.set_xlim(0, max_time + 0.5)
    ax.set_ylim(-0.5, len(users) - 0.5)
    ax.set_yticks(range(len(users)))
    ax.set_yticklabels(users, fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Keystroke Dynamics Race: 'the quick brown fox'", fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Containers for the text objects
    user_texts = {}
    user_cursors = {}
    
    for i, user in enumerate(users):
        # Initial text (empty)
        # We will render the typed text on the line corresponding to the user
        txt = ax.text(0.1, i, "", fontsize=14, fontfamily='monospace', verticalalignment='center')
        user_texts[user] = txt
        
        # Cursor marker (a vertical line or block)
        # cursor = ax.plot([0, 0], [i-0.3, i+0.3], color='red', linewidth=2)[0]
        # user_cursors[user] = cursor

    # Animation function
    def update(frame):
        current_t = frame * 0.05 # 50ms steps
        
        for i, user in enumerate(users):
            timeline = timelines[user]
            
            # Determine which characters have been pressed by current_t
            typed_str = ""
            for event in timeline:
                if current_t >= event['press_time']:
                    # Simple visualization: Character appears when pressed
                    typed_str += event['char']
                else:
                    break
            
            user_texts[user].set_text(typed_str)
            user_texts[user].set_x(0.1) # Keep aligned left
            
            # Optional: Move a cursor or show hold duration visual
            # But simple text appearance is probably what's requested + "style"
            
            # To visualize "style" (holds/flights), maybe we can color the text?
            # Or show a timeline bar below the text?
            
        return user_texts.values()

    # Calculate frames
    # Total time + padding
    total_frames = int((max_time + 1.0) / 0.05)
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    
    print(f"Saving animation to {OUTPUT_GIF}...")
    try:
        ani.save(OUTPUT_GIF, writer='pillow', fps=20)
        print("Done.")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Try installing imageio or ffmpeg if pillow fails.")

if __name__ == "__main__":
    profiles = load_data()
    if profiles:
        create_animation(profiles)
    else:
        print("No profiles found.")
