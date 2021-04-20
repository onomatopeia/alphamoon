import random
from enum import Enum

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset


class Phase(Enum):
    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'


class TripletDataset(Dataset):
    """
    For each anchor, a positive and a negative samples are chosen at random
    """

    def __init__(self, X, y):
        self.X = X
        self.anchors = []
        self.positives = []
        self.negatives = []

        count_arr = np.bincount(y[:, 0])
        all_classes = {i: np.where(y == i)[0] for i in range(len(count_arr))}

        for i, count in enumerate(count_arr):
            if count == 0:
                continue
            anchor = list(all_classes[i])
            random.shuffle(anchor)
            self.anchors.extend(anchor)

            positive = list(all_classes[i])
            random.shuffle(positive)
            self.positives.extend(positive)

            population = [x for x, x_count in enumerate(count_arr) if
                          x != i and x_count > 0]
            negative_classes = random.choices(population, k=count)
            for j, neg_count in enumerate(np.bincount(negative_classes)):
                self.negatives.extend(
                    random.choices(all_classes[j], k=neg_count))

        triplets = list(zip(self.anchors, self.positives, self.negatives))
        random.shuffle(triplets)
        self.anchors, self.positives, self.negatives = zip(*triplets)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        return (
                   self.X[self.anchors[index]],
                   self.X[self.positives[index]],
                   self.X[self.negatives[index]]
                ), []

    def __len__(self):
        return len(self.anchors)


def get_data_loaders(X_train, y_train, batch_size: int = 10,
                     shuffle: bool = True,
                     num_workers: int = 0, pin_memory: bool = True,
                     train_fraction=0.8):
    data_loaders = dict()
    params = dict(batch_size=batch_size, shuffle=shuffle,
                  num_workers=num_workers, pin_memory=pin_memory)
    train_dataset = TripletDataset(X_train, y_train)
    total = X_train.shape[0]
    train_size = int(total * train_fraction)
    validation_size = total - train_size
    train_set, validation_set = torch.utils.data.random_split(train_dataset,
                                                              [train_size,
                                                               validation_size])
    data_loaders[Phase.TRAIN] = torch.utils.data.DataLoader(train_set, **params)
    data_loaders[Phase.VALIDATION] = torch.utils.data.DataLoader(validation_set,
                                                                 **params)
    return data_loaders


def get_transformation_matrix(img_w_h):
    import random

    if random.random() > 0.5:
        theta = np.deg2rad(random.randint(-10, 10))
        ry = rx = img_w_h / 2

        r00 = np.cos(theta)
        r01 = -np.sin(theta)
        r10 = np.sin(theta)
        r11 = np.cos(theta)
        r02 = rx - r00 * rx - r01 * ry
        r12 = ry - r10 * rx - r11 * ry

        rotation = np.array([[r00, r01, r02], [r10, r11, r12], [0, 0, 1]])
    else:
        rotation = np.identity(3)

    Sx = Sy = 1.0
    Tx = Ty = 0
    if random.random() > 0.5:
        Sx = random.uniform(0.88, 1.12)
    if random.random() > 0.5:
        Sy = random.uniform(0.88, 1.12)
    if random.random() > 0.5:
        Tx = random.randint(-7, 7)
    if random.random() > 0.5:
        Ty = random.randint(-7, 7)

    affine2 = np.array([[Sy, 0, Tx], [0, Sx, Ty], [0, 0, 1]])
    trans = np.matmul(rotation, affine2)
    return trans
