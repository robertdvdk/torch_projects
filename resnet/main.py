# Import statements
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import datasets
import model
import time, gc
import os
from tqdm import tqdm
import argparse

# Function definitions
def mean(lst):
    return round(sum(lst) / len(lst), 3)

def val(X, y, net, device, dataset):
    X, y = X.to(device), y.to(device)
    preds = net(X)
    return calculate_accuracy(preds, y, 'top1' if dataset == 'cifar10' else 'top5')

def calculate_accuracy(preds, y, metric='top1'):
    if metric == 'top1':
        return (torch.argmax(preds, dim=-1) == y).float().mean().cpu().numpy()
    top5 = torch.sort(preds, dim=-1, descending=True)[1][:, :5]
    y_stack = torch.stack((y, y, y, y, y), dim=-1)
    return torch.where(top5 == y_stack, 1., 0.).sum(dim=-1).mean().cpu().numpy()


def train_batch(X, y, net, loss_fn, optimizer, device, dataset, scaler=None):
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
    return calculate_accuracy(preds, y, 'top1' if dataset == 'cifar10' else 'top5'), loss.item()

def train(net, trainloader, valloader, device, optimizer, num_epochs, scheduler,
          model_name, dataset, writer, scaler=None):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_accs, val_accs, losses = [], [], []
        start = time.time()
        for X, y in tqdm(trainloader):
            train_acc, loss = train_batch(X, y, net, loss_fn, optimizer, device, dataset, scaler)
            train_accs.append(train_acc)
            losses.append(loss)

        with torch.no_grad():
            for X, y in tqdm(valloader):
                val_acc = val(X, y, net, device, dataset)
                val_accs.append(val_acc)

        torch.save(net.state_dict(), f'./{model_name}/trained_model.pt')
        scheduler.step()

        epoch_train_acc, epoch_loss, epoch_val_acc = mean(train_accs), mean(losses), mean(val_accs)
        epoch_time = round(time.time() - start, 3)
        writer.add_scalar('Training accuracy', epoch_train_acc, epoch)
        writer.add_scalar('Training loss', epoch_loss, epoch)
        writer.add_scalar('Validation accuracy', epoch_val_acc, epoch)
        writer.add_scalar('Epoch time', epoch_time, epoch)
        writer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    args = parser.parse_args()

    if not os.path.exists(f'./{args.model_name}'):
        os.mkdir(f'./{args.model_name}')
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=args.dataset)

    if args.dataset.lower() == 'cifar10':
        net = model.ResNet(3, 32, 10)
        trainset = datasets.CIFAR10(root='../../datasets/cifar10/', train=True)
        valset = datasets.CIFAR10(root='../../datasets/cifar10/', train=False)
    elif args.dataset.lower() == 'imagenet':
        net = model.ResNet(3, 224, 1000)
        trainset = datasets.ImageNet(root='../../datasets/imagenet/', split='train')
        valset = datasets.ImageNet(root='../../datasets/imagenet/', split='val')
    else:
        raise RuntimeError("Choose either cifar10 or imagenet as dataset")

    writer.add_text('Dataset', args.dataset.lower())
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Optimizer', args.optimizer)
    writer.add_text('Weight decay', str(args.weight_decay))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.num_epochs))

    net = net.to(device)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=8,
                                              pin_memory=False, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, num_workers=4)

    optimizer = torch.optim.SGD(params=net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train(net=net, trainloader=trainloader, valloader=valloader, device=device, optimizer=optimizer,
          num_epochs=args.num_epochs, scheduler=scheduler, model_name=args.model_name, dataset=args.dataset,
          writer=writer, scaler=scaler)
    writer.close()
if __name__ == "__main__":
    main()
