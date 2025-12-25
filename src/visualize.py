"""
Visualization script for analyzing keystroke patterns and model performance.
"""
import os
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from config import KEYSTROKE_DATA_FILE, IMPOSTER_DATA_FILE, MODEL_FILE, VISUALIZATION_FILE

# Fix for headless servers (SSH)
matplotlib.use('Agg')

def load_data():
    """Loads and prepares real and imposter data."""
    if not os.path.exists(KEYSTROKE_DATA_FILE):
        print(f"Error: {KEYSTROKE_DATA_FILE} not found.")
        return None, None

    real_user = pd.read_csv(KEYSTROKE_DATA_FILE)
    real_user['label'] = 1

    if os.path.exists(IMPOSTER_DATA_FILE):
        fake_user = pd.read_csv(IMPOSTER_DATA_FILE)
        fake_user['label'] = 0
    else:
        print("Warning: No imposter data found. Visualization will be limited.")
        fake_user = pd.DataFrame()

    return real_user, fake_user

def plot_typing_patterns(real_user, fig, gs):
    """Plots dwell and flight time patterns."""
    # 1. Dwell Time Analysis (Boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    dwell_cols = [c for c in real_user.columns if 'dwell' in c]
    sns.boxplot(data=real_user[dwell_cols], ax=ax1, palette="Blues")
    ax1.set_title("User Dwell Time Pattern (Muscle Memory)", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylabel("Seconds")
    ax1.grid(True, alpha=0.3)

    # 2. Flight Time Analysis (Boxplot)
    ax2 = fig.add_subplot(gs[0, 1])
    flight_cols = [c for c in real_user.columns if 'flight' in c]
    sns.boxplot(data=real_user[flight_cols], ax=ax2, palette="Reds")
    ax2.set_title("User Flight Time Pattern (Rhythm)", fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylabel("Seconds")
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

def plot_model_performance(real_user, fake_user, fig, gs):
    """Plots ROC curve and Confusion Matrix."""
    if fake_user.empty or not os.path.exists(MODEL_FILE):
        return

    # Prepare data
    data = pd.concat([real_user, fake_user], ignore_index=True)
    feature_cols = [c for c in data.columns if c != 'label']
    X = data[feature_cols]
    y = data['label']

    # Load model
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    # Get predictions (using probability for ROC)
    # We use the whole dataset here just for visualization purposes 
    # (in a real scenario, use test set)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[1, 0])
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax4,
                xticklabels=['Imposter', 'User'], yticklabels=['Imposter', 'User'])
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')

def visualize(output=VISUALIZATION_FILE):
    """Main visualization function."""
    print("Generating enhanced analysis...")
    
    real_user, fake_user = load_data()
    if real_user is None:
        return

    # Set up the visualization style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with a grid specification
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)

    # Plot patterns
    plot_typing_patterns(real_user, fig, gs)

    # Plot performance (if imposter data and model exist)
    plot_model_performance(real_user, fake_user, fig, gs)
    
    # Add metadata
    plt.figtext(0.02, 0.02, f"User Samples: {len(real_user)}\nImposter Samples: {len(fake_user)}", 
                fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Enhanced visualization saved to {output}. Download to view.")

if __name__ == "__main__":
    visualize()