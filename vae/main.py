# Import statements
import torch
import torchvision
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
import json
from model import VAE
import torch.profiler
import time

# Function definitions
def train(vae, loader, opt, sched, savedir, device, epochs, latent_size):
    zdist = MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))
    for epoch in range(epochs):
        start = time.time()
        for step, (x, _) in enumerate(loader):
            x = x.to(device)
            znoise = zdist.sample_n(x.shape[0]).to(device)
            x = x.view(x.shape[0], -1)
            y, zmean, zstd = vae(x, znoise)
            L1 = -0.5 * torch.sum(1 + torch.log(torch.square(zstd)) - torch.square(zmean) - torch.square(zstd))
            L2 = F.mse_loss(y, x, reduction='sum')
            L = (L1 + L2)
            L.backward()
            opt.step()
            opt.zero_grad()
        print(f'L1: {round(L1.item(), 2)}\tL2: {round(L2.item(), 2)}')
        end = time.time()
        print(end - start)
        sched.step()
        plt.imshow(x[1].cpu().view(28, 28))
        plt.savefig(f'./{savedir}/groundtruth_{epoch}.png')
        plt.imshow(y[1].detach().cpu().view(28, 28))
        plt.savefig(f'./{savedir}/reconstructed_{epoch}.png')

        # To generate a random sample, we sample z ~ N(0, I), and pass it through the decoder.
        znoise = zdist.sample_n(1).to(device)
        y = vae.sample(znoise)
        plt.imshow(y.detach().cpu().view(28, 28))
        plt.savefig(f'./{savedir}/generated_{epoch}.png')

        with open(f'./{savedir}/log.txt', 'a') as fopen:
            fopen.write(f'Epoch: {epoch}\nLR: {sched.get_last_lr()}\nL1: {-L1}\nL2: {-L2}\n')
    torch.save(vae.state_dict(), f'./{savedir}/vae.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='results', type=str, help='Path to the save directory')
    parser.add_argument('--hidden_size', default=500, type=int, help='Number of hidden neurons')
    parser.add_argument('--latent_size', default=10, type=int, help='Dimensionality of the latent space')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    args = parser.parse_args()
    if not os.path.exists(f'./{args.save_dir}/'):
        os.mkdir(f'./{args.save_dir}/')
    with open(f'./{args.save_dir}/args.txt', 'a') as fopen:
        json.dump(args.__dict__, fopen, indent=2)

    trainset = torchvision.datasets.MNIST(root='./data/', download=True, train=True, transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae = VAE(784, args.hidden_size, args.latent_size).to(device)
    print(f'Training on {device}')
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    train(vae, trainloader, optimizer, scheduler, args.save_dir, device, args.epochs, args.latent_size)

if __name__ == "__main__":
    main()
