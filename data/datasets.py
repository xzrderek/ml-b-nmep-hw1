from __future__ import annotations

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
import h5py
ImageFile.LOAD_TRUNCATED_IMAGES = True


# from torchvision.transforms import Compose, Normalize, PILToTensor, Resize


class MediumImagenetDataset(Dataset):
    def __init__(self, config, train=True, augment=True):
        self.config = config
        self.train = train
        self.input_size = config.DATA.IMG_SIZE
        self.transform = self._get_transforms()
        self.augment=True
        ds = ImageFolder("/data/medium-imagenet/data")
        if train:
            self.dataset = Subset(ds, range(0, len(ds) * 9 // 10))
        else:
            self.dataset = Subset(ds, range(len(ds) * 9 // 10, len(ds)))

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self):
        transform = []
        transform.append(transforms.PILToTensor())
        transform.append(lambda x: x.to(torch.float))
        normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform.append(transforms.Normalize(normalization[0], normalization[1]))
        if self.train and self.augment:
            transform.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ]
            )
        transform.append(transforms.Resize([self.input_size] * 2))
        return transforms.Compose(transform)


class MediumImagenetHDF5Dataset(Dataset):
    def __init__(self, config, train=True, augment=True):
        self.config = config
        self.train = train
        self.input_size = config.DATA.IMG_SIZE
        self.transform = self._get_transforms()


    def __getitem__(self, index):
        f = h5py.File("/data/medium-imagenet/data.hdf5", "r")
        image = f["images-" + ("train" if self.train else "val")][index]
        label = f["labels-" + ("train" if self.train else "val")][index]
        image = self.transform(image)
        label = torch.tensor(label).float()
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self):
        transform = []
        transform.append(lambda x: torch.tensor(x/256).to(torch.float))
        normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform.append(transforms.Normalize(normalization[0], normalization[1]))
        if self.train:
            transform.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ]
            )
        return transforms.Compose(transform)


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.train = train

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
