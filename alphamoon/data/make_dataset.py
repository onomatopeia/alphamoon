import numpy as np
from torch.utils.data import Dataset
import random
from enum import Enum
import torch
import torch.utils.data


class Phase(Enum):
    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'


class TripletDataset(Dataset):
    """
    Training: for each anchor, a positive and a negative samples are chosen at random
    Testing: fixed triplets ?
    """

    def __init__(self, X_train, y_train, X_test, y_test, phase):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.phase = phase
        self.train_classes = self.split_to_classes(y_train)
        self.test_classes = self.split_to_classes(y_test)

    @staticmethod
    def split_to_classes(labels):
        outcome = dict()
        count_arr = np.bincount(labels[:, 0])
        for i in count_arr:
            outcome[i] = np.where(labels == i)[0]
        return outcome

    def __getitem__(self, index):
        if self.phase == Phase.TRAIN:
            return self.get_triplet(index, self.X_train, self.y_train), []
        else:
            return self.get_triplet(index, self.X_test, self.y_test), []

    @staticmethod
    def get_triplet(index, X, y):
        x_anchor = X[index]
        y_anchor = y[index, 0]
        indices_for_pos = np.where(y == y_anchor)[0]
        indices_for_neg = np.where(y != y_anchor)[0]

        idx_pos = random.choice(indices_for_pos)
        idx_neg = random.choice(indices_for_neg)

        return x_anchor, X[idx_pos], X[idx_neg]

    def __len__(self):
        if self.phase == Phase.TRAIN:
            return len(self.y_train)
        else:
            return len(self.y_test)


def get_data_loader(phase: Phase, X_train, y_train, X_test, y_test, batch_size: int = 10, shuffle: bool = True,
                    num_workers: int = 0, pin_memory: bool = True):
    return torch.utils.data.DataLoader(TripletDataset(X_train, y_train, X_test, y_test, phase), batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
