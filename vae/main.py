# Import statements
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
    def __init__(self, input_channels, hidden_size, latent_dim, lr):
        super().__init__()
        self.save_hyperparameters()

        self.enc = Encoder(input_channels, hidden_size, latent_dim)
        self.dec = Decoder(input_channels, hidden_size, latent_dim)

    def forward(self, x):
        zmean, zstd = self.enc(x)
        z = zmean + zstd * torch.randn(zmean.shape, device=zmean.device)
        y = self.dec(z)
        L_reconstruction = F.mse_loss(y, x, reduction='sum')
        L_regularization = -0.5 * torch.sum(1 + torch.log(torch.square(zstd)) - torch.square(zmean) - torch.square(zstd))
        return L_reconstruction, L_regularization

    @torch.no_grad()
    def sample(self, batch_size):
        y = self.dec(torch.randn(batch_size, self.hparams.latent_dim, device=self.device))
        return y

    def training_step(self, batch, batch_idx):
        x, _ = batch
        L_reconstruction, L_regularization = self.forward(x)
        ELBO = L_reconstruction + L_regularization
        self.log("train_L_reconstruction", L_reconstruction, on_step=False, on_epoch=True)
        self.log("train_L_regularization", L_regularization, on_step=False, on_epoch=True)
        self.log("train_ELBO", ELBO, on_step=False, on_epoch=True)
        return ELBO

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        L_reconstruction, L_regularization = self.forward(x)
        ELBO = L_reconstruction + L_regularization
        self.log("val_L_reconstruction", L_reconstruction, on_step=False, on_epoch=True)
        self.log("val_L_regularization", L_regularization, on_step=False, on_epoch=True)
        self.log("val_ELBO", ELBO, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        L_reconstruction, L_regularization = self.forward(x)
        ELBO = L_reconstruction + L_regularization
        self.log("test_ELBO", ELBO, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size, every_n_epochs, save_to_disk):
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        samples = pl_module.sample(self.batch_size)
        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True, value_range=(0, 1), pad_value=0.5)
        grid = grid.detach().cpu()
        trainer.logger.experiment.add_image("Samples", grid, global_step=epoch)
        if self.save_to_disk:
            torchvision.utils.save_image(grid,
                        os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_samples.png"))


def train(args):
    os.makedirs(args.log_dir, exist_ok=True)
    trainloader, valloader, testloader = mnist(args.batch_size)

    save = ModelCheckpoint(save_weights_only=True, monitor="val_ELBO")
    generate = GenerateCallback(batch_size=64, every_n_epochs=5, save_to_disk=True)
    trainer = pl.Trainer(default_root_dir=args.log_dir, accelerator="auto",
                         max_epochs=args.epochs, callbacks=[save, generate])

    pl.seed_everything(args.seed)
    model = VAE(input_channels=1, hidden_size=args.hidden_size, latent_dim = args.latent_dim, lr=args.lr)

    # Train model
    trainer.fit(model, trainloader, valloader)

    # Test model
    model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, testloader)

    return test_result


if __name__ == "__main__":
    # Feel free to add more argument parameters
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
