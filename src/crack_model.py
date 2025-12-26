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

from config import KEYSTROKE_DATA_FILE, MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
from utils import load_artifacts
from logger import logger

# GAN Components
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1), # Matched LeakyReLU usage
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1), # Matched LeakyReLU usage
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)
            # No activation on output because we are generating standardized values (can be negative)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(X_train, input_dim, latent_dim=64, epochs=300, batch_size=32, device='cpu'):
    """
    Trains a GAN to replicate the input data distribution.
    """
    # Adjust batch size for small datasets
    num_samples = len(X_train)
    if num_samples < batch_size:
        # Ensure at least 2 samples for BatchNorm
        new_batch_size = max(2, num_samples)
        logger.warning(f"Dataset size ({num_samples}) smaller than batch_size ({batch_size}). Adjusting to {new_batch_size}.")
        batch_size = new_batch_size

    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    logger.info(f"Training Generative Adversarial Network (GAN) for {epochs} epochs...")
    
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
            z = torch.randn(current_batch_size, latent_dim).to(device)

            # Generate a batch of images
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
        # Note: load_artifacts expects paths as strings or Path objects
        auth_model, scaler, threshold = load_artifacts(
            MODEL_FILE, SCALER_FILE, THRESHOLD_FILE
        )
    except SystemExit:
        logger.error("Failed to load artifacts. Make sure the model is trained first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auth_model.to(device)
    auth_model.eval()

    # 2. Load and Prepare Data for GAN Training
    # We use the same dataset the model was trained on to train our "Cracker" GAN
    logger.info(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Data file {input_file} not found.")
        return

    # Use the target model's scaler to transform data into the expected distribution
    # The GAN will learn to generate data in this scaled space
    X_real = df.values
    X_scaled = scaler.transform(X_real)
    
    input_dim = X_scaled.shape[1]
    latent_dim = 64
    
    # 3. Train the Cracking GAN
    logger.info("\n--- Phase 1: Training Generative Model ---")
    generator = train_gan(X_scaled, input_dim, latent_dim=latent_dim, epochs=300, device=device)
    
    # 4. Generate Attack Samples
    logger.info("\n--- Phase 2: Launching Attack ---")
    num_attacks = 1000
    print(f"Generating {num_attacks} synthetic keystroke patterns...")
    
    with torch.no_grad():
        z = torch.randn(num_attacks, latent_dim).to(device)
        fake_samples = generator(z)

        # Save forged data for external verification
        logger.info("Saving forged data...")
        fake_samples_unscaled = scaler.inverse_transform(fake_samples.cpu().numpy())
        forged_df = pd.DataFrame(fake_samples_unscaled, columns=df.columns)
        
        # Create output filename (e.g., data/forged_keystroke_data.csv)
        input_dir = os.path.dirname(input_file)
        input_name = os.path.basename(input_file)
        output_path = os.path.join(input_dir, f"forged_{input_name}")
        
        forged_df.to_csv(output_path, index=False)
        print(f"Forged data saved to: {output_path}")
        
        # 5. Test against Auth Model
        # The auth model calculates reconstruction error (MSE)
        reconstructed = auth_model(fake_samples)
        
        # Calculate MSE for each sample
        # shape: (n_samples, input_dim)
        mse_scores = torch.mean(torch.pow(fake_samples - reconstructed, 2), dim=1)
        
        # Check how many pass the threshold
        # If MSE < Threshold -> Authenticated (Attack Successful)
        successful_cracks = (mse_scores < threshold).sum().item()
        
    success_rate = (successful_cracks / num_attacks) * 100
    
    print("-" * 50)
    print(f"Attack Results:")
    print(f"Threshold: {threshold:.6f}")
    print(f"Average MSE of Generated Fakes: {mse_scores.mean().item():.6f}")
    print(f"Successful Cracks: {successful_cracks} / {num_attacks}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("-" * 50)
    
    if success_rate > 50:
        logger.critical("CRITICAL: The model is highly vulnerable to generative attacks.")
    elif success_rate > 10:
        logger.warning("WARNING: The model shows some vulnerability to generative attacks.")
    else:
        logger.info("PASS: The model is relatively robust against this simple generative attack.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN to crack the authentication model.")
    parser.add_argument("input_file", nargs="?", default=str(KEYSTROKE_DATA_FILE), help="Path to the keystroke data CSV file to use for training the GAN.")
    args = parser.parse_args()

    crack_model(input_file=args.input_file)