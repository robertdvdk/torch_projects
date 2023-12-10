# Import statements
import torchvision
import torch.utils.data


# Function definitions
def mnist(batch_size):
    train_val = torchvision.datasets.MNIST(root='./data/', download=True, train=True,
                                           transform=torchvision.transforms.ToTensor())
    trainset, valset = torch.utils.data.random_split(train_val, [54000, 6000])
    testset = torchvision.datasets.MNIST(root='./data/', download=True, train=False,
                                         transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=True,
                                              num_workers=4)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=False,
                                            num_workers=4)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=False,
                                            num_workers=4)
    return trainloader, valloader, testloader


if __name__ == "__main__":
    pass
