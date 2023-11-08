# Import statements
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torch.utils.data
import model
import time, gc
import os
import numpy as np
from tqdm import tqdm
import argparse
import torchvision

# Function definitions
def val(net, loader, device):
    accs = []
    for x, t in loader:
        x, t = x.to(device), t.to(device)
        preds = torch.argmax(net(x), dim=-1)
        accs.append((preds == t).float().mean().cpu().numpy())
    return np.mean(accs)

def train_batch(X, y, net, loss_fn, optimizer, device, scaler=None):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    if not scaler:
        preds = net(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
    else:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            preds = net(X)
            loss = loss_fn(preds, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return (torch.argmax(preds, dim=-1) == y).float().mean().cpu().numpy()

def train(net, trainloader, valloader, device, optimizer, num_epochs, scaler, model_name):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        accs = []
        start = time.time()
        for i, (X, y) in enumerate(tqdm(trainloader)):
            correct = train_batch(X, y, net, loss_fn, optimizer, device, scaler)
            accs.append(correct)
        train_acc = np.mean(accs)
        with torch.no_grad():
            val_acc = val(net, valloader, device)
        time_taken = time.time() - start
        with open(f'./results/{model_name}.txt', 'a') as fopen:
            fopen.write(f'EPOCH {epoch}\nTime: {time_taken}\nTrain accuracy: {train_acc}\nValidation accuracy: {val_acc}\n')
        torch.save(net.state_dict(), f'./{model_name}.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()

    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.ResNet(3, 224, 1000)
    net = net.to(device)
    train_transforms = transforms.Compose([
                                            transforms.RandomResize(256, 480),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(224),
                                            transforms.ColorJitter(0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.ImageNet('../../datasets/imagenet/', transform=train_transforms, split='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=8, pin_memory=False, shuffle=True)
    valset = torchvision.datasets.ImageNet('../../datasets/imagenet/', transform=test_transforms, split='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, num_workers=4)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    train(net, trainloader, valloader, device, optimizer, 100, scaler, args.model_name)

if __name__ == "__main__":
    main()
