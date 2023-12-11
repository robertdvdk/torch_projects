
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


def mnist(batch_size: int, data_root: str = './data/') -> [DataLoader, DataLoader, DataLoader]:
    """
    Splits the official MNIST training set into a training set and a validation set, and returns the training,
    validation, and test dataloaders.

    Args:
        batch_size (int): Batch size.
        data_root (str): Path to the directory containing the MNIST folder.

    Returns:
        trainloader (torch.utils.data.DataLoader): Training set dataloader.
        valloader (torch.utils.data.DataLoader): Validation set dataloader.
        testloader (torch.utils.data.DataLoader): Test set dataloader.
    """
    train_val = MNIST(root=data_root, download=True, train=True, transform=ToTensor())
    trainset, valset = random_split(train_val, [54000, 6000])
    testset = MNIST(root=data_root, download=True, train=False, transform=ToTensor())
    trainloader = DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4)
    return trainloader, valloader, testloader


if __name__ == "__main__":
    pass
