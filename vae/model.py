import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int, latent_dim: int) -> None:
        """
        Initializes a variational encoder with the given hyperparameters.

        Args:
            input_channels (int): Number of channels in the input images.
            hidden_size (int): Number of filters used in intermediate convolutional layers.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 2 * hidden_size, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),  # Image grid to single feature vector
        )
        self.mean_layer = nn.Linear(2 * hidden_size * 4 * 4, latent_dim)
        self.log_std_layer = nn.Linear(2 * hidden_size * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Encodes an input batch into a latent mean and standard deviation.

        Args:
            x (Tensor[B, C, H, W]): Batch of input images.

        Returns:
            zmean (Tensor[B, D]): One latent mean per input image, where D == latent_dim.
            zstd (Tensor[B, D]): One latent standard deviation per input image, where D == latent_dim.
        """
        x = self.conv(x)
        zmean = self.mean_layer(x)
        zstd = torch.exp(self.log_std_layer(x))
        return zmean, zstd


class Decoder(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int, latent_dim: int) -> None:
        """
        Initializes a standard decoder with the given hyperparameters.

        Args:
            input_channels (int): Number of channels in the input images.
            hidden_size (int): Number of filters used in intermediate convolutional layers.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * hidden_size),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_size, hidden_size, kernel_size=3, padding=1, stride=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_size, input_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a sample from the latent distribution to a sample in data space.

        Args:
            z (torch.Tensor): Sample in latent space.

        Returns:
            y (torch.Tensor): Sample in data space.
        """
        y = self.linear(z)
        y = y.reshape(y.shape[0], -1, 4, 4)
        y = self.net(y)
        return y


if __name__ == "__main__":
    pass
