import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure we can import from src regardless of where script is run
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import KEYSTROKE_DATA_FILE, MODEL_FILE, SCALER_FILE, THRESHOLD_FILE, REQUIRED_LENGTH
from utils import load_artifacts
from logger import logger

# --- LSTM-GAN Components ---

class LSTMGenerator(nn.Module):
    """
    LSTM-based Generator for Sequential Keystroke Data.
    Generates data one key event at a time to capture rhythm and dependencies.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        # z shape: (batch_size, seq_len, latent_dim)
        lstm_out, _ = self.lstm(z)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        out = self.linear(lstm_out)
        # out shape: (batch_size, seq_len, output_dim)
        return out

class LSTMDiscriminator(nn.Module):
    """
    LSTM-based Discriminator.
    Determines if a sequence of keystrokes is real or fake.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # LSTM output: (batch, seq, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We use the final hidden state of the last layer to classify the whole sequence
        # h_n shape: (num_layers, batch, hidden). Take last layer.
        final_state = h_n[-1] 
        
        validity = self.linear(final_state)
        return validity

def train_gan(X_train, seq_len, feature_dim, latent_dim=64, epochs=300, batch_size=32, device='cpu'):
    """
    Trains a Sequential GAN (LSTM-GAN).
    """
    # Adjust batch size for small datasets
    num_samples = len(X_train)
    if num_samples < batch_size:
        new_batch_size = max(2, num_samples)
        logger.warning(f"Dataset size ({num_samples}) smaller than batch_size ({batch_size}). Adjusting to {new_batch_size}.")
        batch_size = new_batch_size

    # Initialize Models
    generator = LSTMGenerator(latent_dim, hidden_dim=128, output_dim=feature_dim).to(device)
    discriminator = LSTMDiscriminator(input_dim=feature_dim, hidden_dim=128).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()

    # Data Loader
    dataset = TensorDataset(torch.FloatTensor(X_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    logger.info(f"Training LSTM-GAN for {epochs} epochs...")
    
    d_loss = torch.tensor(0.0)
    g_loss = torch.tensor(0.0)

    for epoch in range(epochs):
        for i, (real_imgs,) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones(current_batch_size, 1).to(device)
            fake = torch.zeros(current_batch_size, 1).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            # Noise is now a sequence too: (Batch, Seq_Len, Latent)
            z = torch.randn(current_batch_size, seq_len, latent_dim).to(device)

            # Generate a batch of sequences
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    return generator

def crack_model(input_file=KEYSTROKE_DATA_FILE):
    # 1. Load the Target Auth Model and Scaler
    logger.info("Loading target authentication model and artifacts...")
    try:
        auth_model, scaler, threshold = load_artifacts(
            MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
        )
    except SystemExit:
        logger.error("Failed to load artifacts. Make sure the model is trained first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auth_model.to(device)
    auth_model.eval()

    # 2. Load and Prepare Data
    logger.info(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Data file {input_file} not found.")
        return

    # Scale Data
    X_real = df.values
    X_scaled = scaler.transform(X_real)
    
    # 3. Reshape for LSTM (Batch, Seq_Len, Features)
    # Original shape: (N, 60) -> 3 features per key * 20 keys
    features_per_key = 3
    seq_len = REQUIRED_LENGTH # 20
    
    if X_scaled.shape[1] != seq_len * features_per_key:
         logger.error(f"Data dim {X_scaled.shape[1]} does not match expected {seq_len}*{features_per_key}={seq_len*features_per_key}")
         return

    X_train_seq = X_scaled.reshape(-1, seq_len, features_per_key)
    
    input_dim = features_per_key
    latent_dim = 16 # Latent vector size per time step
    
    # 3. Train the Cracking LSTM-GAN
    logger.info("\n--- Phase 1: Training LSTM-GAN ---")
    generator = train_gan(X_train_seq, seq_len, input_dim, latent_dim=latent_dim, epochs=400, device=device)
    
    # 4. Generate Attack Samples
    logger.info("\n--- Phase 2: Launching Sequence Attack ---")
    num_attacks = 1000
    print(f"Generating {num_attacks} synthetic keystroke patterns...")
    
    with torch.no_grad():
        z = torch.randn(num_attacks, seq_len, latent_dim).to(device)
        fake_samples_seq = generator(z) # (N, 20, 3)
        
        # Flatten back to (N, 60) for Auth Model
        fake_samples_flat = fake_samples_seq.reshape(num_attacks, -1)

        # Save forged data for external verification
        logger.info("Saving forged data...")
        fake_samples_unscaled = scaler.inverse_transform(fake_samples_flat.cpu().numpy())
        forged_df = pd.DataFrame(fake_samples_unscaled, columns=df.columns)
        
        input_dir = os.path.dirname(input_file)
        input_name = os.path.basename(input_file)
        output_path = os.path.join(input_dir, f"forged_{input_name}")
        
        forged_df.to_csv(output_path, index=False)
        print(f"Forged data saved to: {output_path}")
        
        # 5. Test against Auth Model
        reconstructed = auth_model(fake_samples_flat)
        
        # Calculate MSE
        mse_scores = torch.mean(torch.pow(fake_samples_flat - reconstructed, 2), dim=1)
        successful_cracks = (mse_scores < threshold).sum().item()
        
    success_rate = (successful_cracks / num_attacks) * 100
    
    print("-" * 50)
    print(f"Attack Results (LSTM Generator):")
    print(f"Threshold: {threshold:.6f}")
    print(f"Average MSE of Fakes: {mse_scores.mean().item():.6f}")
    print(f"Successful Cracks: {successful_cracks} / {num_attacks}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("-" * 50)
    
    if success_rate > 50:
        logger.critical("CRITICAL: The model is vulnerable to sequential attacks.")
    elif success_rate > 10:
        logger.warning("WARNING: The model shows some vulnerability.")
    else:
        logger.info("PASS: The model is robust against sequential attacks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM-GAN to crack the authentication model.")
    parser.add_argument("input_file", nargs="?", default=str(KEYSTROKE_DATA_FILE), help="Path to the keystroke data CSV file to use for training the GAN.")
    args = parser.parse_args()

    crack_model(input_file=args.input_file)