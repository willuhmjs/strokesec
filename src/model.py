import torch
import torch.nn as nn

class KeystrokeAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(KeystrokeAutoencoder, self).__init__()
        
        # Calculate hidden layer sizes based on input (Dynamic Architecture)
        # Architecture: Input -> Enc1 -> Bottleneck -> Dec1 -> Output
        enc_1 = int(input_dim * 0.75)
        bottleneck = int(input_dim * 0.5)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, enc_1),
            nn.BatchNorm1d(enc_1),  # Batch Norm for training stability
            nn.ReLU(),
            nn.Dropout(0.1),        # Dropout to prevent overfitting
            nn.Linear(enc_1, bottleneck),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, enc_1),
            nn.ReLU(),
            nn.Linear(enc_1, input_dim)
            # No activation on output for regression (reconstructing values)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded