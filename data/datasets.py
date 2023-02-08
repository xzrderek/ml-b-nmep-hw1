from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import Compose, Normalize, PILToTensor, Resize


class MediumImagenetDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset = ImageFolder("/data/medium-imagenet/data")

        transforms = []
        transforms.append(PILToTensor())
        transforms.append(lambda x: x.to("cuda"))
        normalization = torch.tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]).to("cuda")
        transforms.append(Normalize(normalization[0], normalization[1]))
        transforms.append(Resize(config.DATA.IMG_SIZE))
        self.transforms = Compose(transforms)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


class CIFAR10Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset = CIFAR10(".")

        transforms = []
        transforms.append(PILToTensor())
        transforms.append(lambda x: x.to("cuda"))
        transforms.append(Resize(config.DATA.IMG_SIZE))
        self.transforms = Compose(transforms)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
