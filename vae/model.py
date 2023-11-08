# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function definitions
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.enc = Encoder(input_size, hidden_size, latent_size)
        self.dec = Decoder(input_size, hidden_size, latent_size)

    def forward(self, x, znoise):
        zmean, zstd = self.enc(x)
        z = zmean + zstd * znoise
        y = self.dec(z)
        return y, zmean, zstd

    def sample(self, znoise):
        return self.dec(znoise)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)
        self.latent_size = latent_size

    def forward(self, x):
        x = F.relu(self.fc(x))
        zmean = self.mean_layer(x)
        zstd = torch.sqrt(torch.exp(self.logvar_layer(x)))
        return zmean, zstd

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        z = F.relu(self.fc1(z))

        # We use a sigmoid activation as our pixels are values between 0 and 1.
        y = F.sigmoid(self.out(z))
        return y

if __name__ == "__main__":
    pass
