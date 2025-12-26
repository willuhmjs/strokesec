"""
Professional training script using PyTorch with Early Stopping and Checkpointing.
"""
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import KEYSTROKE_DATA_FILE, MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from utils import augment_data
from model import KeystrokeAutoencoder
from logger import logger

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 300
PATIENCE = 30  # Increased patience for deeper model

def train_model(X_train, X_val, input_dim):
    # Prepare DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeystrokeAutoencoder(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Scheduler: Reduce LR if validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Device: {device} | Input Dim: {input_dim}")
    print("-" * 60)

    for epoch in range(MAX_EPOCHS):
        # --- Training Step ---
        model.train()
        train_loss = 0.0
        for batch_features, _ in train_loader:
            batch_features = batch_features.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, _ in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_features)
                val_loss += loss.item() * batch_features.size(0)
        
        val_loss /= len(val_loader.dataset)

        # Step Scheduler
        scheduler.step(val_loss)

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best weights
    model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default=str(KEYSTROKE_DATA_FILE))
    parser.add_argument("--model", default=str(MODEL_FILE))
    parser.add_argument("--scaler", default=str(SCALER_FILE))
    parser.add_argument("--threshold", default=str(THRESHOLD_FILE))
    args = parser.parse_args()

    # 1. Load Data
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} records from {args.input_file}")
    except FileNotFoundError:
        logger.error("Data file not found.")
        sys.exit(1)
        
    X_real = df.values
    
    # 2. Split
    X_train_raw, X_test_raw = train_test_split(X_real, test_size=0.2, random_state=42)
    
    # 3. Augment (Only Train Data)
    logger.info(f"Augmenting data... (Original: {len(X_train_raw)})")
    # Note: augment_data returns a dataframe, we convert to numpy
    X_train_aug_df = augment_data(pd.DataFrame(X_train_raw, columns=df.columns))
    X_train_aug = X_train_aug_df.values
    
    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 5. Train
    input_dim = X_train_scaled.shape[1]
    model = train_model(X_train_scaled, X_test_scaled, input_dim)
    
    # 6. Calculate Threshold (using clean validation data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_test_scaled).to(device)
        reconstructed = model(X_val_tensor).cpu().numpy()
        
    mse_scores = np.mean(np.power(X_test_scaled - reconstructed, 2), axis=1)
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    # Strict threshold: Mean + 1.5 StdDev (Tweak multiplier as needed)
    threshold = mean_mse + 1.5 * std_mse
    
    logger.info(f"Final Threshold: {threshold:.6f} (Mean: {mean_mse:.6f}, Std: {std_mse:.6f})")

    # 7. Save Artifacts
    # Save State Dict for PyTorch
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim  # Save dim to recreate model structure later
    }, args.model)
    
    with open(args.scaler, "wb") as f:
        pickle.dump(scaler, f)
    with open(args.threshold, "wb") as f:
        pickle.dump(threshold, f)

    logger.info("Artifacts saved successfully.")
