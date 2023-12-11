import torch
import torchvision
import torch.nn.functional as F
from dataset import mnist

from model import Encoder, Decoder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse


class VAE(pl.LightningModule):
    def __init__(self, input_channels: int, hidden_size: int, latent_dim: int, lr: float):
        """
        Initializes a VAE model with the specified hyperparameters.

        Args:
            input_channels (int): Number of input filterst to the CNN.
            hidden_size (int): Number of output filters from the CNN.
            latent_dim (int): Dimensionality of the latent space.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()

        self.enc = Encoder(input_channels, hidden_size, latent_dim)
        self.dec = Decoder(input_channels, hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """
        Calculates the reconstruction loss as the MSE between the input and the output of the decoder. Calculates
        the regularization loss as the KL divergence between the output of the encoder and an isotropic Gaussian.

        Args:
            x (Tensor[B, C, H, W]): Input images.

        Returns:
            L_reconstruction (Tensor[]): Reconstruction loss.
            L_regularization (Tensor[]): Regularization loss.
        """
        zmean, zstd = self.enc(x)
        z = zmean + zstd * torch.randn(zmean.shape, device=zmean.device)
        y = self.dec(z)
        L_reconstruction = F.mse_loss(y, x, reduction='sum')
        L_regularization = -0.5 * torch.sum(1 + torch.log(torch.square(zstd)) - torch.square(zmean) - torch.square(zstd))
        return L_reconstruction, L_regularization

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Samples from the model by decoding a sample from an isotropic Gaussian.

        Args:
            batch_size (int): Batch size.

        Returns:
            y (Tensor[B, C, H, W]): Samples from the model.
        """
        y = self.dec(torch.randn(batch_size, self.hparams.latent_dim, device=self.device))
        return y

    def training_step(self, batch: int, batch_idx: int) -> torch.Tensor:
        """
        Calculates the Evidence Lower Bound of the training set batch as the sum of the reconstruction and
        regularization losses. Writes to log file, and returns for backpropagation.

        Args:
            batch (Tensor[B, C, H, W]): Training batch.
            batch_idx (int): Batch index.

        Returns:
            ELBO (Tensor[]): Evidence lower bound.
        """
        x, _ = batch
        L_reconstruction, L_regularization = self.forward(x)
        ELBO = L_reconstruction + L_regularization
        self.log("train_L_reconstruction", L_reconstruction, on_step=False, on_epoch=True)
        self.log("train_L_regularization", L_regularization, on_step=False, on_epoch=True)
        self.log("train_ELBO", ELBO, on_step=False, on_epoch=True)
        return ELBO

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Calculates the Evidence Lower Bound of the validation set batch as the sum of the reconstruction and
        regularization losses, and writes it to log file.

        Args:
            batch (Tensor[B, C, H, W]): Validation batch.
            batch_idx (int): Batch index.
        """
        x, _ = batch
        L_reconstruction, L_regularization = self.forward(x)
        ELBO = L_reconstruction + L_regularization
        self.log("val_L_reconstruction", L_reconstruction, on_step=False, on_epoch=True)
        self.log("val_L_regularization", L_regularization, on_step=False, on_epoch=True)
        self.log("val_ELBO", ELBO, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx) -> None:
        """
        Calculates the Evidence Lower Bound of the test set batch as the sum of the reconstruction and
        regularization losses, and writes it to log file.

        Args:
            batch (Tensor[B, C, H, W]): Test batch.
            batch_idx (int): Batch index.
        """
        x, _ = batch
        L_reconstruction, L_regularization = self.forward(x)
        ELBO = L_reconstruction + L_regularization
        self.log("test_ELBO", ELBO, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Configures Adam with the model parameters and the specified learning rate.

        Returns:
            optimizer (torch.optim): Adam optimizer with specified learning rate.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size: int, every_n_epochs: int, save_to_disk: bool) -> None:
        """
        Callback that generates samples from the model every n epochs and optionally saves them to disk.

        Args:
            batch_size (int): Batch size.
            every_n_epochs (int): Generate samples every n epochs.
            save_to_disk (bool): Whether to save samples to disk.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called on every train epoch. We use it to call the sample_and_save function every n epochs.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer.
            pl_module (pl.LightningModule): PyTorch Lightning module.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch: int) -> None:
        """
        Generates samples from the model, puts them in the Tensorboard logger, and saves them to disk if specified.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer.
            pl_module (pl.LightningModule): PyTorch Lightning module.
            epoch (int): Current epoch.
        """
        samples = pl_module.sample(self.batch_size)
        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True, value_range=(0, 1), pad_value=0.5)
        grid = grid.detach().cpu()
        trainer.logger.experiment.add_image("Samples", grid, global_step=epoch)
        if self.save_to_disk:
            torchvision.utils.save_image(grid,
                        os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_samples.png"))


def train(args: argparse.Namespace) -> [dict]:
    """
    Trains and tests a VAE model with the specified hyperparameters.

    Args:
        args (argparse.Namespace): Namespace object containing the hyperparameters.

    Returns:
        test_result ([dict]): Evidence Lower Bound on test set.

    """
    os.makedirs(args.log_dir, exist_ok=True)
    trainloader, valloader, testloader = mnist(args.batch_size)

    save = ModelCheckpoint(save_weights_only=True, monitor="val_ELBO")
    generate = GenerateCallback(batch_size=64, every_n_epochs=5, save_to_disk=True)
    trainer = pl.Trainer(default_root_dir=args.log_dir, accelerator="auto",
                         max_epochs=args.epochs, callbacks=[save, generate])

    pl.seed_everything(args.seed)
    model = VAE(input_channels=1, hidden_size=args.hidden_size, latent_dim=args.latent_dim, lr=args.lr)

    # Train model
    trainer.fit(model, trainloader, valloader)

    # Test model
    model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, testloader)

    return test_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--latent_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_size', default=32, type=int,
                        help='Number of filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data.')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.')
    parser.add_argument('--log_dir', default='VAE_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')

    args = parser.parse_args()

    train(args)
