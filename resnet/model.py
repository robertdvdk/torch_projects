# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function definitions
class Block(nn.Module):
    def __init__(self, in_features, out_features, residual=True):
        super().__init__()

        # Downsample if the output map has more features
        stride = 1 if in_features == out_features else 2

        # If bias is immediately followed by BatchNorm, then the bias is superfluous.
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.batchnorm = nn.BatchNorm2d(out_features)

    def forward(self, x):
        out = self.relu(self.batchnorm(self.conv1(x)))
        out = self.batchnorm(self.conv2(out))

        if self.in_features == self.out_features:
            out = self.relu(out + x)
        else:
            map_size = out.shape[-1]

            # If the dimensionality of the input and output maps differ, we fill the non-matching dimensions with zeros.
            # Also, we use a stride of 2 in the spatial dimension.
            out = self.relu(out + F.pad(x, (0, 0, 0, 0, 0, self.out_features - self.in_features))[:, :, torch.arange(0, map_size*2, 2)][:, :, :, torch.arange(0, map_size*2, 2)])
        return out

class BlockSequence(nn.Module):
    def __init__(self, in_features, out_features, n_blocks, residual=True):
        super().__init__()
        if in_features == out_features:
            self.blocks = nn.ModuleList([])
        else:
            self.blocks = nn.ModuleList([Block(in_features, out_features, residual)])
        self.blocks.extend([Block(out_features, out_features, residual) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_features, image_size, classes, residual=True):
        super().__init__()

        self.first_layer = nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        feature_dims = [64, 64, 128, 256, 512]
        self.res_blockseqs = nn.ModuleList([BlockSequence(in_dim, out_dim, 2, residual) for in_dim, out_dim
                      in zip(feature_dims[:-1], feature_dims[1:])])

        self.globalpool = nn.AvgPool2d(kernel_size=image_size//32)
        self.dense = nn.Linear(512, classes)

    def forward(self, x):
        x = self.batchnorm(self.first_layer(x))
        x = self.relu(self.maxpool(x))
        for blockseq in self.res_blockseqs:
            x = blockseq(x)
        x = self.globalpool(x).squeeze(-1).squeeze(-1)
        x = self.dense(x)
        return x

def main():
    pass

if __name__ == "__main__":
    main()