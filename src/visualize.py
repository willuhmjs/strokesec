"""
Visualization script for analyzing keystroke patterns and model performance.
"""
import os
import glob
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from config import DATA_DIR, KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, MODEL_FILE, VISUALIZATION_FILE, SCALER_FILE

# Fix for headless servers (SSH)
matplotlib.use('Agg')

def load_data():
    """
    Loads all user data (files ending in _data.csv inside data/) 
    and the imposter data.
    """
    # 1. Load all user datasets found in DATA_DIR
    users_data = {}
    
    # We look for all *_data.csv files but exclude imposter_data.csv and keystroke_data.csv
    # wait, keystroke_data.csv is technically Will's data? Or is will_data.csv Will's data?
    # Based on the list_files output:
    # harrison_data.csv, imposter_data.csv, keystroke_data.csv, maximus_data.csv, tristan_data.csv, will_data.csv
    # It seems keystroke_data.csv might be a duplicate or the main training file. 
    # The prompt explicitly asked to analyze: will_data.csv, harrison_data.csv, tristan_data.csv, maximus_data.csv
    
    pattern = os.path.join(DATA_DIR, "*_data.csv")
    all_files = glob.glob(pattern)
    
    # Specific users we want to target based on the prompt
    # If specific files are present, we prefer those names.
    # We will exclude 'imposter_data.csv' from the users list.
    
    for fpath in all_files:
        filename = os.path.basename(fpath)
        if filename == "imposter_data.csv":
            continue
        
        # We can use the filename (minus _data.csv) as the label
        # e.g. "will_data.csv" -> "will"
        if "_data.csv" in filename:
            user_name = filename.replace("_data.csv", "").capitalize()
            # Special case: if keystroke_data.csv exists, treat it as "Main User (Training)" or skip if will_data.csv exists?
            # The prompt says: "Update to match ... new data/ directory ... analyze all available user datasets (will_data.csv, harrison_data.csv, tristan_data.csv, maximus_data.csv)"
            # So let's prioritize the specific names if they exist.
            
            # If both keystroke_data.csv and will_data.csv exist, and they are identical, we might duplicate.
            # Let's trust the named files first.
            
            if filename == "keystroke_data.csv":
                # We can skip this if we have named users, or include it as "Training Data"
                # For now let's just include it if it's there, or maybe rename it to "Default"
                # But looking at the prompt, it lists specific files.
                # Let's include it as "Default" for now to be safe, but the named ones are priority.
                user_name = "Default (keystroke_data)"
            
            df = pd.read_csv(fpath)
            # Add a 'User' column for plotting
            df['User'] = user_name
            # Label for model (we treat all these as "positive" examples for visualization context, 
            # though model training might only use one)
            df['label'] = 1 
            
            users_data[user_name] = df

    # Combine all users into one dataframe for easy plotting with hue
    if users_data:
        all_users_df = pd.concat(users_data.values(), ignore_index=True)
    else:
        print("Warning: No user data found.")
        all_users_df = pd.DataFrame()

    # 2. Load Imposter Data
    if os.path.exists(IMPOSTER_DATA_FILE):
        imposter_df = pd.read_csv(IMPOSTER_DATA_FILE)
        imposter_df['User'] = 'Imposter'
        imposter_df['label'] = 0
    else:
        print("Warning: No imposter data found.")
        imposter_df = pd.DataFrame()

    return all_users_df, imposter_df

def plot_typing_patterns(all_users_df, fig, gs):
    """
    Plots dwell and flight time patterns comparing different users.
    """
    if all_users_df.empty:
        return

    # Filter columns
    dwell_cols = [c for c in all_users_df.columns if 'dwell' in c]
    flight_cols = [c for c in all_users_df.columns if 'flight' in c]

    # We need to melt the dataframe to have "Key" and "Time" columns for seaborn
    # But we also want to preserve "User".
    # Since there are many keys, maybe we aggregate or pick top keys?
    # Or we can plot the distribution of MEAN dwell/flight time per sample?
    # The original script plotted boxplots of ALL keys side-by-side. 
    # With multiple users, side-by-side for every key is too crowded (4 users * 20 keys = 80 boxes).
    
    # Better approach: Plot density of Average Dwell Time and Average Flight Time per sample per user.
    # This gives a "fingerprint" summary.
    
    # Calculate means per row
    all_users_df['Mean Dwell'] = all_users_df[dwell_cols].mean(axis=1)
    all_users_df['Mean Flight'] = all_users_df[flight_cols].mean(axis=1)

    # 1. Dwell Time Distribution (KDE)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(data=all_users_df, x='Mean Dwell', hue='User', fill=True, ax=ax1, palette="viridis", alpha=0.3)
    ax1.set_title("Distribution of Mean Dwell Times", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Seconds")
    ax1.grid(True, alpha=0.3)

    # 2. Flight Time Distribution (KDE)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(data=all_users_df, x='Mean Flight', hue='User', fill=True, ax=ax2, palette="viridis", alpha=0.3)
    ax2.set_title("Distribution of Mean Flight Times", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Seconds")
    ax2.grid(True, alpha=0.3)

def plot_confidence_distribution(all_users_df, imposter_df, fig, gs):
    """
    Plots the distribution of the model's prediction confidence for each user.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return

    # Load model and scaler
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    # Combine data
    # We want to see how the model treats EVERYONE (Users + Imposters)
    combined_df = pd.concat([all_users_df, imposter_df], ignore_index=True)

    # Prepare features
    feature_cols = [c for c in combined_df.columns if c not in ['label', 'User', 'Mean Dwell', 'Mean Flight', 'Confidence Score']]
    
    # Check model features if available
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    else:
        model_features = feature_cols

    # Ensure we have the right columns
    # (Filter down to only what the model needs)
    X = combined_df[model_features]
    
    # Scale features
    X_scaled = scaler.transform(X)

    # Get probabilities (Confidence Scores)
    # The model predicts probability of class 1 (Target User)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    combined_df['Confidence Score'] = y_prob

    # Plot
    # Spanning both columns for clarity
    ax = fig.add_subplot(gs[1, :]) 
    
    sns.histplot(data=combined_df, x='Confidence Score', hue='User', element="step", stat="density", common_norm=False, ax=ax, palette="tab10", alpha=0.3)
    # Also add KDE for smoothness
    sns.kdeplot(data=combined_df, x='Confidence Score', hue='User', fill=False, common_norm=False, ax=ax, palette="tab10", linewidth=2, warn_singular=False)
    
    ax.axvline(0.85, color='red', linestyle='--', linewidth=2, label='Threshold (0.85)')
    ax.set_title("Model Confidence Distribution (User Identity)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Probability of being Target User (0.0 - 1.0)")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_model_performance(all_users_df, imposter_df, fig, gs):
    """
    Plots ROC and Confusion Matrix.
    WARNING: This assumes the model was trained on ONE of these users (likely Will/Default).
    We should probably only use the 'Default' or 'Will' data for the True Positive part of the ROC/CM 
    to honestly evaluate the trained model, unless we retrained it on everyone (which we didn't).
    
    We will assume 'Will' or 'Default' is the target user.
    """
    if imposter_df.empty or not os.path.exists(MODEL_FILE):
        return

    # Identify the target user for the model evaluation
    # We look for "Will" or "Default" in the users
    target_user_keys = [u for u in all_users_df['User'].unique() if 'Will' in u or 'Default' in u]
    if not target_user_keys:
        # Fallback: just take the first one if we can't find Will
        target_user_keys = [all_users_df['User'].unique()[0]]
    
    target_user_df = all_users_df[all_users_df['User'].isin(target_user_keys)]
    
    print(f"Evaluating model performance against: {target_user_keys}")

    # Prepare data for evaluation: Target User (1) vs Imposters (0)
    # We ignore other users here because they are technically "Imposters" to Will's model, 
    # but we don't label them as 0 to avoid confusing the existing Imposter dataset visualization.
    # Actually, treating other users as 0 (imposters) is a GREAT test of the model!
    # Let's add other users to the "Negative" class for the ROC curve?
    # Prompt didn't explicitly ask for cross-user verification, just "Visualization".
    # Let's stick to the standard Real vs Fake (generated) for consistency, 
    # but maybe add a note or line for other users if easy.
    # For simplicity/safety: Use Target User vs Imposter_Data.
    
    eval_data = pd.concat([target_user_df, imposter_df], ignore_index=True)
    
    feature_cols = [c for c in eval_data.columns if c not in ['label', 'User', 'Mean Dwell', 'Mean Flight']]
    # Ensure we only have numeric columns matching the model
    # (The model expects specific columns. The loaded DF might have extra ones now.)
    
    # Load model to check expected features if possible, or just rely on intersection
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    
    # Load scaler again (or pass it, but simpler to reload here to be self-contained)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    # Get features model expects
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    else:
        # Fallback if feature names aren't saved (older sklearn)
        model_features = feature_cols
        
    # Filter X to match model features
    X = eval_data[model_features]
    y = eval_data['label']

    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[2, 0])
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC (Target User vs Generated Imposters)', fontsize=14, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[2, 1])
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax4,
                xticklabels=['Imposter', 'User'], yticklabels=['Imposter', 'User'])
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')

def visualize(output=VISUALIZATION_FILE):
    """Main visualization function."""
    print("Generating enhanced analysis...")
    
    all_users_df, imposter_df = load_data()
    if all_users_df.empty:
        print("No user data available to visualize.")
        return

    # Set up the visualization style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with a grid specification
    # Increased height to accommodate new plot
    fig = plt.figure(figsize=(20, 22)) 
    gs = fig.add_gridspec(3, 2)

    # Plot patterns (Row 0)
    plot_typing_patterns(all_users_df, fig, gs)
    
    # Plot Confidence Distribution (Row 1)
    plot_confidence_distribution(all_users_df, imposter_df, fig, gs)

    # Plot performance (Row 2)
    plot_model_performance(all_users_df, imposter_df, fig, gs)
    
    # Add metadata footer
    user_counts = all_users_df['User'].value_counts().to_string().replace('\n', ', ')
    plt.figtext(0.02, 0.02, f"Samples: {user_counts} | Imposters: {len(imposter_df)}", 
                fontsize=10, wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for footer
    plt.savefig(output, dpi=300)
    print(f"Enhanced visualization saved to {output}. Download to view.")

if __name__ == "__main__":
    visualize()
