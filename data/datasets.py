from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as transforms
# from torchvision.transforms import Compose, Normalize, PILToTensor, Resize


class MediumImagenetDataset(Dataset):
    def __init__(self, input_size, mode="train"):
        self.dataset = ImageFolder("/data/medium-imagenet/data")
        self.input_size = input_size

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
    
    def _get_transforms(self):
        transform = []
        if self.mode == "train":
            transform.append([
                transforms.PILToTensor(),
                ])
            normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
            transforms.append(transforms.Normalize(normalization[0], normalization[1]))
            transforms.append(transforms.Resize(self.input_size))
            self.transform = transforms.Compose(transforms)

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.train = True

        self.transform = self._get_transforms()
        self.dataset = CIFAR10(root="/data/cifar10", train=self.train, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
    
    def _get_transforms(self):
        if self.train:
            transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        else:
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        return transforms.Compose(transform)
