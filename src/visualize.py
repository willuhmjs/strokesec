"""
Visualization script for analyzing keystroke patterns and model performance.
"""
import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix
from config import DATA_DIR, MODEL_FILE, VISUALIZATION_FILE, SCALER_FILE, THRESHOLD_FILE
from utils import check_columns, load_artifacts
from logger import logger

# Fix for headless servers (SSH)
matplotlib.use('Agg')

def load_data():
    """
    Loads all user data (files ending in _data.csv inside data/) 
    """
    all_users_df = pd.DataFrame()
    
    # 1. Find all CSV files in data directory
    csv_files = list(DATA_DIR.glob("*_data.csv"))
    
    if not csv_files:
        logger.warning("No data files found in data/ directory.")
        return pd.DataFrame()

    frames = []
    
    for fpath in csv_files:
        filename = fpath.name
        
        # Determine User Name from filename
        if filename == "keystroke_data.csv":
            user_name = "Owner (Training)"
            label = 1
        elif filename == "imposter_data.csv":
            user_name = "Imposter"
            label = 0
        else:
            # e.g. "tristan_data.csv" -> "Tristan"
            user_name = filename.replace("_data.csv", "").capitalize()
            # We don't know if they are auth or imposter, but for visualization 
            # we usually treat the non-training files as "Tests"
            label = 0 

        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue
                
            # Use utility to ensure columns are correct
            df = check_columns(df)
            
            df['User'] = user_name
            df['label'] = label
            frames.append(df)
            logger.info(f"Loaded {len(df)} samples from {user_name} ({filename})")
            
        except Exception as e:
            logger.error(f"Skipping {filename}: {e}")

    if frames:
        all_users_df = pd.concat(frames, ignore_index=True)
        
    return all_users_df

def plot_typing_patterns(all_users_df, fig, gs):
    """
    Plots dwell, flight, and hold time patterns comparing different users.
    """
    if all_users_df.empty:
        return

    # Filter columns based on features
    hold_cols = [c for c in all_users_df.columns if '_hold' in c]
    ud_cols = [c for c in all_users_df.columns if '_ud' in c]
    
    # Calculate means per row
    all_users_df['Mean Hold'] = all_users_df[hold_cols].mean(axis=1)
    all_users_df['Mean UD'] = all_users_df[ud_cols].mean(axis=1)

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

def plot_reconstruction_error(all_users_df, fig, gs):
    """
    Plots the distribution of the Autoencoder's reconstruction error (MSE) for each user.
    """
    if not MODEL_FILE.exists() or not SCALER_FILE.exists() or not THRESHOLD_FILE.exists():
        logger.warning("Model artifacts not found. Skipping MSE plot.")
        return

    # Load artifacts using shared utility (handles PyTorch loading)
    try:
        model, scaler, threshold = load_artifacts(MODEL_FILE, SCALER_FILE, THRESHOLD_FILE)
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return

    # Prepare features
    # Exclude metadata columns
    meta_cols = ['label', 'User', 'Mean Hold', 'Mean UD', 'Mean DD', 'MSE']
    feature_cols = [c for c in all_users_df.columns if c not in meta_cols]
    
    # In PyTorch, we don't store feature names directly in the model object in the same way,
    # so we rely on the scaler or use all available feature columns that match.
    # The check_columns utility already standardized the columns.
    model_features = feature_cols

    # Filter/Order columns to match model input
    try:
        X = all_users_df[model_features]
    except KeyError as e:
        logger.error(f"Feature mismatch: {e}")
        return

    # Scale features
    try:
        # Transform using values to avoid feature name mismatch warnings if scaler was fitted on numpy
        X_scaled = scaler.transform(X.values)
    except Exception as e:
        logger.error(f"Error scaling data for visualization: {e}")
        return

    # Reconstruct and Calculate MSE (PyTorch)
    input_tensor = torch.FloatTensor(X_scaled)
    model.eval() # Ensure eval mode
    with torch.no_grad():
        reconstructed_tensor = model(input_tensor)
    
    X_reconstructed = reconstructed_tensor.numpy()
    mse_scores = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
    
    all_users_df['MSE'] = mse_scores

    # Plot
    ax = fig.add_subplot(gs[1, :])
    
    # Prepare data for Scatter/Line plot
    # We want X-axis to be just an index, but grouped by user for clarity
    
    # Create a synthetic index for plotting
    all_users_df = all_users_df.sort_values(by=['User'])
    all_users_df['Sample_Index'] = range(len(all_users_df))
    
    # Scatter plot of MSEs
    sns.scatterplot(data=all_users_df, x='Sample_Index', y='MSE', hue='User', style='User', s=100, ax=ax, palette="tab10")
    
    # Plot Threshold Line (Horizontal)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=3, label=f'Model Threshold ({threshold:.4f})')
    
    # Add background zones
    # Green Zone (Safe)
    ax.axhspan(0, threshold, alpha=0.1, color='green', label='Access Granted Zone')
    # Red Zone (Danger)
    # We need a reasonable upper limit for the red zone
    max_mse = all_users_df['MSE'].max()
    ax.axhspan(threshold, max_mse * 1.1, alpha=0.1, color='red', label='Access Denied Zone')
    
    ax.set_title("Authentication Decision Boundary: MSE vs Threshold", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_xlabel("Sample Index")
    
    # Use log scale if the difference is massive (often the case with anomalies)
    # But let's check max value first. If huge, log scale is better.
    if max_mse > threshold * 10:
        ax.set_yscale('log')
        ax.set_ylabel("Mean Squared Error (Log Scale)")
        # Set a realistic lower bound to avoid squashing data due to near-zero MSEs
        ax.set_ylim(bottom=1e-3)
        
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

def visualize(output=VISUALIZATION_FILE):
    """Main visualization function."""
    logger.info("Generating enhanced analysis...")
    
    all_users_df = load_data()
    if all_users_df.empty:
        logger.warning("No user data available to visualize.")
        return

    # Set up the visualization style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with a grid specification
    # Row 0: Pattern Distribution
    # Row 1: Decision Boundary (Scatter)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    # Plot patterns (Row 0)
    plot_typing_patterns(all_users_df, fig, gs)
    
    # Plot Reconstruction Error (Row 1 - Spans both cols)
    plot_reconstruction_error(all_users_df, fig, gs)

    # Add metadata footer
    user_counts = all_users_df['User'].value_counts().to_string().replace('\n', ', ')
    plt.figtext(0.02, 0.02, f"Samples: {user_counts}", fontsize=10, wrap=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for footer
    plt.savefig(output, dpi=300)
    logger.info(f"Enhanced visualization saved to {output}. Download to view.")

if __name__ == "__main__":
    visualize()
