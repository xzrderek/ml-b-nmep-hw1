"""ResNet implementation taken from kuangliu on github
link: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    expansion: int = 1
    def __init__(self,in_channels, out_channels, stride = 1, norm_layer= None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        self.shortcut = nn.Sequential()

    def conv3x3(in_channels, out_channels, stride = 1):
      return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self):
      # Read the following, and uncomment it when you understand it, no need to add more code
      num_classes = 10
      super(ResNet18, self).__init__()
      self.in_channels = 64
      self.conv1 = nn.Conv2d(in_channels=3, 
                              out_channels=64, 
                              kernel_size=3,
                              stride=1, 
                              padding=1, 
                              bias=False)
      self.bn1 = nn.BatchNorm2d(64)
      self.layer1 = self.make_block(out_channels=64, stride=1)
      self.layer2 = self.make_block(out_channels=128, stride=2)
      self.layer3 = self.make_block(out_channels=256, stride=2)
      self.layer4 = self.make_block(out_channels=512, stride=2)
      self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, stride):
        # Read the following, and uncomment it when you understand it, no need to add more code
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Read the following, and uncomment it when you understand it, no need to add more code
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
