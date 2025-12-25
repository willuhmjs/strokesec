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
from config import DATA_DIR, KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, MODEL_FILE, VISUALIZATION_FILE, SCALER_FILE, THRESHOLD_FILE

# Fix for headless servers (SSH)
matplotlib.use('Agg')

def load_data():
    """
    Loads all user data (files ending in _data.csv inside data/) 
    and the imposter data.
    """
    # 1. Load all user datasets found in DATA_DIR
    users_data = {}
    
    pattern = os.path.join(DATA_DIR, "*_data.csv")
    all_files = glob.glob(pattern)
    
    for fpath in all_files:
        filename = os.path.basename(fpath)
        if filename == "imposter_data.csv":
            continue
        
        if "_data.csv" in filename:
            user_name = filename.replace("_data.csv", "").capitalize()
            
            if filename == "keystroke_data.csv":
                user_name = "Default (Training)"
            
            try:
                df = pd.read_csv(fpath)
                # Add a 'User' column for plotting
                df['User'] = user_name
                # Label for model (we treat all these as "positive" examples for visualization context, 
                # though model training might only use one)
                df['label'] = 1 
                
                users_data[user_name] = df
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    # Combine all users into one dataframe for easy plotting with hue
    if users_data:
        all_users_df = pd.concat(users_data.values(), ignore_index=True)
    else:
        print("Warning: No user data found.")
        all_users_df = pd.DataFrame()

    # 2. Load Imposter Data
    # Note: Imposter data is now optional for training but useful for visualization
    if os.path.exists(IMPOSTER_DATA_FILE):
        try:
            imposter_df = pd.read_csv(IMPOSTER_DATA_FILE)
            imposter_df['User'] = 'Imposter'
            imposter_df['label'] = 0
        except Exception as e:
            print(f"Warning: Could not read imposter data: {e}")
            imposter_df = pd.DataFrame()
    else:
        print("Warning: No imposter data found.")
        imposter_df = pd.DataFrame()

    return all_users_df, imposter_df

def plot_typing_patterns(all_users_df, fig, gs):
    """
    Plots dwell, flight, and hold time patterns comparing different users.
    """
    if all_users_df.empty:
        return

    # Filter columns based on new features: hold, ud, dd
    hold_cols = [c for c in all_users_df.columns if '_hold' in c]
    ud_cols = [c for c in all_users_df.columns if '_ud' in c]
    dd_cols = [c for c in all_users_df.columns if '_dd' in c]

    # Calculate means per row
    all_users_df['Mean Hold'] = all_users_df[hold_cols].mean(axis=1)
    all_users_df['Mean UD'] = all_users_df[ud_cols].mean(axis=1)
    all_users_df['Mean DD'] = all_users_df[dd_cols].mean(axis=1)

    # 1. Hold Time Distribution (KDE)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(data=all_users_df, x='Mean Hold', hue='User', fill=True, ax=ax1, palette="viridis", alpha=0.3)
    ax1.set_title("Distribution of Mean Hold Times", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Seconds")
    ax1.grid(True, alpha=0.3)

    # 2. UD (Up-Down) Flight Time Distribution (KDE)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(data=all_users_df, x='Mean UD', hue='User', fill=True, ax=ax2, palette="viridis", alpha=0.3)
    ax2.set_title("Distribution of Mean UD Flight Times", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Seconds")
    ax2.grid(True, alpha=0.3)

def plot_reconstruction_error(all_users_df, imposter_df, fig, gs):
    """
    Plots the distribution of the Autoencoder's reconstruction error (MSE) for each user.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE) or not os.path.exists(THRESHOLD_FILE):
        return

    # Load artifacts
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(THRESHOLD_FILE, "rb") as f:
        threshold = pickle.load(f)

    # Combine data
    combined_df = pd.concat([all_users_df, imposter_df], ignore_index=True)

    # Prepare features
    feature_cols = [c for c in combined_df.columns if c not in ['label', 'User', 'Mean Hold', 'Mean UD', 'Mean DD', 'MSE']]
    
    # Check model features if available
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    else:
        # If feature names aren't saved, try to match by intersection or assume all remaining are features
        # The scaler might know the number of features
        model_features = feature_cols

    # Ensure we have the right columns
    # We need to filter only the columns the model expects.
    # If the model was trained on specific columns, we must use those.
    # Assuming 'feature_cols' contains all the raw feature columns (k0_hold, k0_ud, etc.)
    
    try:
        X = combined_df[model_features]
    except KeyError:
        # Fallback: if names don't match, just take the first N columns where N is scaler's n_features_in_
        if hasattr(scaler, "n_features_in_"):
             X = combined_df.iloc[:, :scaler.n_features_in_]
        else:
             X = combined_df[feature_cols]

    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print(f"Error scaling data for visualization: {e}")
        return

    # Reconstruct and Calculate MSE
    X_reconstructed = model.predict(X_scaled)
    mse_scores = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
    
    combined_df['MSE'] = mse_scores

    # Plot
    ax = fig.add_subplot(gs[1, :]) 
    
    # Histogram + KDE
    sns.histplot(data=combined_df, x='MSE', hue='User', element="step", stat="density", common_norm=False, ax=ax, palette="tab10", alpha=0.3)
    sns.kdeplot(data=combined_df, x='MSE', hue='User', fill=False, common_norm=False, ax=ax, palette="tab10", linewidth=2, warn_singular=False)
    
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    # Annotation for Safe Zone vs Anomaly Zone
    # ax.text(threshold * 0.5, ax.get_ylim()[1]*0.9, "Safe Zone (Low Error)", color='green', ha='center')
    # ax.text(threshold * 1.5, ax.get_ylim()[1]*0.9, "Anomaly Zone (High Error)", color='red', ha='center')

    ax.set_title("Reconstruction Error Distribution (MSE)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean Squared Error (Lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale might be useful if outliers are huge, but let's stick to linear first or limit x
    # ax.set_xscale('log') 

def plot_model_performance(all_users_df, imposter_df, fig, gs):
    """
    Plots ROC and Confusion Matrix based on MSE Thresholding.
    """
    if imposter_df.empty or not os.path.exists(MODEL_FILE) or not os.path.exists(THRESHOLD_FILE):
        return

    # Identify the target user (The one the model was trained on)
    # Usually "Default (Training)" or "Will"
    target_user_keys = [u for u in all_users_df['User'].unique() if 'Training' in u or 'Will' in u]
    if not target_user_keys:
        target_user_keys = [all_users_df['User'].unique()[0]]
    
    target_user_df = all_users_df[all_users_df['User'].isin(target_user_keys)]
    
    print(f"Evaluating model performance against target: {target_user_keys}")

    # Combine Target User (Positive) and Imposters (Negative)
    # We treat all other users as Imposters for this evaluation if we wanted, 
    # but strictly speaking, we want to test Target vs Imposter Dataset first.
    
    # 1 (True) = Target User, 0 (False) = Imposter
    target_user_df = target_user_df.copy()
    target_user_df['label'] = 1
    
    imposter_df = imposter_df.copy()
    imposter_df['label'] = 0
    
    eval_data = pd.concat([target_user_df, imposter_df], ignore_index=True)
    
    # Prepare X and y
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(THRESHOLD_FILE, "rb") as f:
        threshold = pickle.load(f)

    # Feature extraction (same as above)
    feature_cols = [c for c in eval_data.columns if c not in ['label', 'User', 'Mean Hold', 'Mean UD', 'Mean DD', 'MSE']]
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
        X = eval_data[model_features]
    else:
        if hasattr(scaler, "n_features_in_"):
             X = eval_data.iloc[:, :scaler.n_features_in_]
        else:
             X = eval_data[feature_cols]

    y_true = eval_data['label']

    # Predict
    X_scaled = scaler.transform(X)
    X_reconstructed = model.predict(X_scaled)
    mse_scores = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

    # 3. ROC Curve
    # For Autoencoder anomaly detection, "Score" is usually -MSE (higher score = more normal)
    # or we can just flip the logic: 
    # Positive Class (User) should have LOW MSE. Negative Class (Imposter) should have HIGH MSE.
    # To use standard ROC function which expects higher values for Positive class:
    # We can use a similarity score = exp(-MSE) or 1 / (1+MSE)
    # Or just negative MSE.
    
    y_score = -mse_scores # Higher (less negative) is better (lower error)

    ax3 = fig.add_subplot(gs[2, 0])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC (Target vs Imposters)', fontsize=14, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    # Using the fixed Threshold
    # If MSE <= Threshold -> Predicted 1 (User)
    # If MSE > Threshold -> Predicted 0 (Imposter)
    y_pred = (mse_scores <= threshold).astype(int)

    ax4 = fig.add_subplot(gs[2, 1])
    cm = confusion_matrix(y_true, y_pred)
    
    # Label formatting
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Greens', ax=ax4,
                xticklabels=['Imposter', 'User'], yticklabels=['Imposter', 'User'])
    ax4.set_title('Confusion Matrix (Fixed Threshold)', fontsize=14, fontweight='bold')
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
    fig = plt.figure(figsize=(20, 22)) 
    gs = fig.add_gridspec(3, 2)

    # Plot patterns (Row 0)
    plot_typing_patterns(all_users_df, fig, gs)
    
    # Plot Reconstruction Error (Row 1)
    plot_reconstruction_error(all_users_df, imposter_df, fig, gs)

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

