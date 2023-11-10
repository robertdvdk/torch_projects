"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import torch.utils.data
import torchvision.transforms.v2 as transforms
import json
import torchvision

# Function definitions
class INaturalist(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split='train', category='fine'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if category == 'fine':
            self.category = 2
        elif category == 'coarse':
            self.category = 1
            self.supercategories = {'Amphibians': 0, 'Birds': 1, 'Fungi': 2, 'Insects': 3, 'Plants': 4, 'Reptiles': 5}

        if split not in ['train', 'val', 'test']:
            raise ValueError('Choose one of the following splits: train, val, test')

        with open(root_dir + f'{split}2019.json', 'r') as fopen:
            self.annotations = json.load(fopen)['images']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.root_dir + self.annotations[idx]['file_name']
        img = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        img = self.transform(img)
        if self.split == 'test':
            return img
        else:
            if self.category == 2:
                label = int(self.annotations[idx]['file_name'].split('/')[2])
            else:
                label = self.supercategories[self.annotations[idx]['file_name'].split('/')[2]]
            return img, label

class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, **kwargs):
        kwargs['transform'] = self.get_transforms()[0] if kwargs['split'] == 'train' else self.get_transforms()[1]
        super().__init__(**kwargs)

    @staticmethod
    def get_transforms():
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
        return train_transforms, test_transforms

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, **kwargs):
        kwargs['transform'] = self.get_transforms()[0] if kwargs['train'] else self.get_transforms()[1]
        print(kwargs)
        super().__init__(**kwargs)

    @staticmethod
    def get_transforms():
        train_transforms = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        return train_transforms, test_transforms

def main():
    pass

if __name__ == "__main__":
    main()
