import torch
import torch.nn as nn

class KeystrokeAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(KeystrokeAutoencoder, self).__init__()
        
        # Dynamic Architecture (Input -> Enc1 -> Enc2 -> Bottleneck -> Dec2 -> Dec1 -> Output)
        enc_1 = int(input_dim * 0.8)
        enc_2 = int(input_dim * 0.6)
        bottleneck = int(input_dim * 0.4)
        
        # Minimum size safety
        bottleneck = max(bottleneck, 8)
        enc_2 = max(enc_2, 16)
        enc_1 = max(enc_1, 32)
        
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, enc_1),
            nn.BatchNorm1d(enc_1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Layer 2
            nn.Linear(enc_1, enc_2),
            nn.BatchNorm1d(enc_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Bottleneck
            nn.Linear(enc_2, bottleneck),
            nn.LeakyReLU(0.1)
        )
        
        self.decoder = nn.Sequential(
            # Layer 2 (Reverse)
            nn.Linear(bottleneck, enc_2),
            nn.BatchNorm1d(enc_2),
            nn.LeakyReLU(0.1),
            
            # Layer 1 (Reverse)
            nn.Linear(enc_2, enc_1),
            nn.BatchNorm1d(enc_1),
            nn.LeakyReLU(0.1),
            
            # Output
            nn.Linear(enc_1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded