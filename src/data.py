import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_cifar10_loaders(data_dir, mean, std, batch_size, aug_type="none", device="cpu"):
    if aug_type == "none":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif aug_type == "basic":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(f"Unknown aug_type: {aug_type}")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    pin_memory = device != "cpu"

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory, persistent_workers=True)

    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory, persistent_workers=True)

    return train_loader, test_loader


class CIFAR10C(Dataset):
    def __init__(self, data_dir, corruption, severity, transform=None):
        # severity: 1~5
        # data_dir 아래에 {corruption}.npy, labels.npy 존재
        data = np.load(os.path.join(data_dir, f"{corruption}.npy"))
        labels = np.load(os.path.join(data_dir, "labels.npy"))

        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = data[start:end]
        self.labels = labels[start:end]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_cifar10c_loader(data_dir, corruption, severity, mean, std, batch_size, device="cpu"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    pin_memory = device != "cpu"

    dataset = CIFAR10C(data_dir, corruption, severity, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    return loader