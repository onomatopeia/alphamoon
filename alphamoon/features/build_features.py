import torch
import numpy as np
import os
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 10
num_workers = 0
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

phases = ['train', 'valid', 'test']


def get_data_loader(directory, phase, input_size, batch_size=10, num_workers=0):
    if phase == 'train':
        image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(0.05),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            normalize
        ])
        shuffle = True
    else:
        image_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
        shuffle = False

    return torch.utils.data.DataLoader(
        datasets.ImageFolder(directory, image_transforms),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True)

