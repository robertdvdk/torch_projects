import torchvision
import torch.utils.data


def mnist(batch_size: int):
    """
    Splits the official MNIST training set into a training set and a validation set, and returns the training,
    validation, and test dataloaders.

    Args:
        batch_size (int): Batch size.

    Returns:
        trainloader (torch.utils.data.DataLoader): Training set dataloader.
        valloader (torch.utils.data.DataLoader): Validation set dataloader.
        testloader (torch.utils.data.DataLoader): Test set dataloader.
    """
    train_val = torchvision.datasets.MNIST(root='./data/', download=True, train=True,
                                           transform=torchvision.transforms.ToTensor())
    trainset, valset = torch.utils.data.random_split(train_val, [54000, 6000])
    testset = torchvision.datasets.MNIST(root='./data/', download=True, train=False,
                                         transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=True,
                                              num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, drop_last=False, shuffle=False,
                                            num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False,
                                            num_workers=4)
    return trainloader, valloader, testloader


if __name__ == "__main__":
    pass
