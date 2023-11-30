
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchscripts.utils import *





class TaxoNNsub(nn.Module):
    def __init__(self, input_size):
        super(TaxoNNsub, self).__init__()
        self.input_size = input_size
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=self.padding, stride=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.25)
        output = self.forward(torch.zeros(1, 1, self.input_size))
        self.flat_size = output.size(-1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  
        return x
class TaxoNN(nn.Module):
    def __init__(self, encoders, num_classes):
        super(TaxoNN, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.num_encoders = len(encoders)
        fc_input_size = sum(encoder.flat_size for encoder in self.encoders.values())

        self.fc = nn.Linear(in_features=fc_input_size, out_features=100)
        self.out = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        encoder_outputs = [self.encoders[group](x[group]) for group in self.encoders.keys()]
        x = torch.cat(encoder_outputs, dim=1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.softmax(x)


        return x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #padding = (kernel_size - 1) / 2
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ResidualBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.2)
        self.sa = SpatialAttention()  

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sa(out) * out
        out += residual
        out = self.lrelu(out)
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.dropout = nn.Dropout(p=0.5)  
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.dropout(self.fc1(self.avg_pool(x)))))
        max_out = self.fc2(self.relu1(self.dropout(self.fc1(self.max_pool(x)))))
        out = self.sigmoid(avg_out + max_out)
        return self.gamma * out + x
class ResClass(nn.Module):
    def __init__(self, input_size, resblocks):
        super(ResClass, self).__init__()

        self.resblocks = nn.ModuleList()
        self.filters = []
        output_size = input_size

        for i, (fs, ks, str) in enumerate(resblocks):
            in_channels = 1 if i == 0 else self.filters[i-1]
            self.resblocks.append(ResidualBlock(input_size, in_channels, fs, ks, str))
            self.filters.append(fs)
            # Calculate output size for each block
            output_size = (output_size - ks + 2 * ((ks - 1) // 2)) // str + 1
        self.flat_size = output_size * output_size * self.filters[-1]
        #print(self.flat_size)
        self.fc_flat = nn.Linear(self.flat_size, 2048)
        self.lrelu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)
    def reparametrize_gaussian(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        #latent_z = []
        for block in self.resblocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc_flat(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
