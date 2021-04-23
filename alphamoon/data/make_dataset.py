import pickle
import random
from enum import Enum
from typing import Tuple, Any, Iterable, List, Dict

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split

from alphamoon.constants import INPUT_DATA_PATH

Triplet = Tuple[Iterable, Iterable, Iterable]


class Phase(Enum):
    """Enum describing the phases of model training and evaluation.
    """
    TRAIN = 'train'
    VALIDATION = 'valid'
    TEST = 'test'


class TripletDataset(Dataset):
    """Dataset of triplets in which for each anchor image, a positive and
    a negative sample are chosen at random.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initializes an instance of the ``TripletDataset`` class.

        :param X: feature matrix of size n_examples x n_features
        :param y: label matrix or column vector of size n_examples x 1
        """
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

    def __getitem__(self, index: Any) -> Tuple[Triplet, List]:
        """Returns a tuple that as the first element contains a triplet of \
        images, and as the second contains an empty list.

        :param index: index
        :return: item(s) at index
        """
        if torch.is_tensor(index):
            index = index.tolist()

        return (
                   self.X[self.anchors[index]],
                   self.X[self.positives[index]],
                   self.X[self.negatives[index]]
               ), []

    def __len__(self) -> int:
        """Returns the length of all available examples.
        """
        return len(self.anchors)


def get_data_loader(X: np.ndarray, y: np.ndarray,
                    batch_size: int = 10, shuffle: bool = True,
                    num_workers: int = 0,
                    pin_memory: bool = True) -> DataLoader:
    """Creates a dictionary in which keys correspond to the phase of training \
    / evaluating the model whereas values are data loaders for respective \
    phases.

    :param X: feature matrix of size n_examples x n_features
    :param y: label matrix or column vector of size n_examples x 1
    :param batch_size: size of a batch
    :param shuffle: a flag whether to shuffle the data
    :param num_workers: number of workers, 0 by default corresponding to the \
        auto setup
    :param pin_memory: whether to use page-locked ("pinned") memory
    :return: dictionary of Phase-bound data loaders
    """
    train_set = TripletDataset(X, y)
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_transformation_matrix(img_w_h: int) -> np.ndarray:
    """Randomly samples planar image transformations from which a \
    transformation matrix is created and returned. \
    Each transformation is included in the image with probability 0.5. \
    Parameters of each drawn transformation are selected at random.

    Possible transformations are:

        - rotation by an angle drawn from [-10, 10] about the centre of an image
        - horizontal scaling by a factor drawn from [0.88, 1.12]
        - vertical scaling by a factor drawn from [0.88, 1.12]
        - horizontal translation by the number of pixels drawn from [-7,7]
        - vertical translation by the number of pixels drawn from [-7,7]

    :param img_w_h: image width and height (images are assumed to be square)
    :return: transformation matrix
    """
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


def fix_duplicate_classes(X: np.ndarray, y: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Fixes duplicate classes by reassigning examples in class 30 to class
    14.

    :param X: examples
    :param y: labels
    :return: modified examples, modified labels
    """
    one_shot_example = np.where(y[:, 0] == 30)[0][0]

    X_augmented = X.copy()
    y_augmented = y.copy()
    y_augmented[one_shot_example] = 14
    return X_augmented, y_augmented


def get_data(file_path=INPUT_DATA_PATH,
             fix_duplicate_classes_on: bool = True) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Returns the corrected data.

    :param file_path: path to a file with data
    :param fix_duplicate_classes_on: whether to fix duplicate classes
    :return: a tuple of samples and labels numpy arrays
    """
    with file_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)

    if fix_duplicate_classes_on:
        X, y = fix_duplicate_classes(X, y)
    return X, y
