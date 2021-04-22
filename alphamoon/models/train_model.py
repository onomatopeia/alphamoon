import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.cuda
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphamoon.constants import (MODELS_DIR, RAW_DATA_DIR,
                                 MODELS_INVESTIGATION_DIR,
                                 EMBEDDING_MODEL_FILENAME)
from alphamoon.data.make_dataset import get_data_loaders, Phase
from alphamoon.features.build_features import EmbeddingNet, TripletNet
from alphamoon.models.classifier import KNearestEmbedding


class EmbeddingNetSupervisor:
    """This class implements the supervision of the learning process of the
    embedding neural network.
    """

    def __init__(self) -> None:
        """Initializes the :class:`EmbeddingNetSupervisor` class.
        """
        self.model_path: Path
        self.valid_loss_min = np.Inf

    def train_embedding(self, n_epochs: int,
                        loaders: Dict[Phase, DataLoader],
                        model: TripletNet,
                        optimizer: Optimizer,
                        criterion: Module,
                        use_cuda: bool,
                        directory: Path) -> 'EmbeddingNetSupervisor':
        """Train an embedding by means of a Triplet Net ``model``.

        :param n_epochs: number of epochs
        :param loaders: dictionary containing dataset loaders for train and \
            validation sets
        :param model: triplet network model instance
        :param optimizer: optimizer instance
        :param criterion: criterion (loss function) instance
        :param use_cuda: a flag whether to use CUDA
        :param directory: output directory path where the created models \
            should be saved
        :return: self
        """
        for epoch in range(1, n_epochs + 1):
            print(f'Epoch {epoch}')
            train_loss = self.train(model, loaders[Phase.TRAIN], use_cuda,
                                    criterion, optimizer)
            valid_loss = self.validate(model, loaders[Phase.VALIDATION],
                                       use_cuda, criterion)

            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} '
                  f'\tValidation Loss: {valid_loss:.6f}')
            self.save_model(valid_loss, model, directory, epoch)

        return self

    def save_model(self, valid_loss: float, model: TripletNet, directory: Path,
                   epoch: int) -> 'EmbeddingNetSupervisor':
        """Save the model if validation loss has decreased.

        :param valid_loss: validation loss
        :param model: model to evaluate
        :param directory: output directory to save the model
        :param epoch: epoch
        :return: self
        """
        if valid_loss <= self.valid_loss_min:
            print(
                f'Validation loss decreased ({self.valid_loss_min:.6f} '
                f'--> {valid_loss:.6f}).  Saving model ...')
            try:
                self.model_path = self.save_embedding_model(model,
                                                            directory,
                                                            epoch=epoch)
                self.valid_loss_min = valid_loss
            except Exception as e:
                print(f'Could not save the model due to {str(e)}')
        return self

    @staticmethod
    def move_to_gpu(data: Iterable[torch.Tensor], use_cuda: bool) \
            -> Iterable[torch.Tensor]:
        if use_cuda:
            return tuple(d.to("cuda") for d in data)
        return data

    def train(self, model: Module, data_loader: DataLoader, use_cuda: bool,
              criterion: Module, optimizer: Optimizer) -> float:
        """Performs one epoch of training of the model and returns the obtained
        training loss.

        :param model: model to be trained
        :param data_loader: training data loader
        :param use_cuda: a flag whether CUDA can be used
        :param criterion: loss function
        :param optimizer: optimizer
        :return: training loss
        """
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data = self.move_to_gpu(data, use_cuda)
            optimizer.zero_grad()
            output = model(*data)
            loss = criterion(*output)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        return train_loss

    def validate(self, model: Module, data_loader: DataLoader, use_cuda: bool,
                 criterion: Module) -> float:
        """Performs one epoch of validating of the model and returns the
        obtained validation loss.

        :param model: model to be validated
        :param data_loader: validation data loader
        :param use_cuda: a flag whether CUDA can be used
        :param criterion: loss function
        :return: validation loss
        """
        valid_loss = 0.0
        model.eval()
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data = self.move_to_gpu(data, use_cuda)
            output = model.forward(*data)
            loss = criterion(*output)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        return valid_loss

    @staticmethod
    def save_embedding_model(model: TripletNet,
                             directory: Path = MODELS_DIR,
                             epoch: int = -1) -> Path:
        """Pickles an embedding model to a file.

        :param model: model
        :param directory: output directory
        :param epoch: epoch number
        :return: the path to the output file
        """
        embedding_model_path = directory / EMBEDDING_MODEL_FILENAME
        state_dict_model_path = directory / 'state_dict_model.pt'
        if epoch > 0:
            torch.save(model.embedding_net.state_dict(),
                       directory / f'embedding_model_{epoch}.pt')
            torch.save(model.state_dict(),
                       directory / f'state_dict_model_{epoch}.pt')

        torch.save(model.embedding_net.state_dict(), embedding_model_path)
        torch.save(model.state_dict(), state_dict_model_path)
        print(
            f'Model saved to {state_dict_model_path}, '
            f'embedding model saved to {embedding_model_path}')
        return embedding_model_path


class Executor:
    """This class supervises the learning process of the
    :class:`KNearestEmbedding` classifier.
    """

    def __init__(self, input_data_path: Path = RAW_DATA_DIR / 'train.pkl',
                 embedding_length: int = 64,
                 output_length: Optional[int] = None, random_state: int = 0,
                 test_size: float = 0.33, n_epochs: int = 10,
                 margin: Union[float, int] = 10) -> None:
        """Initializes an instance of the ``Executor`` class.

        :param input_data_path: path to the input data `pkl` file
        :param embedding_length: length of the embedding layer
        :param output_length: length of the output layer, optional, assumed to \
            be the same as ``embedding_length`` if not provided
        :param random_state: random state
        :param test_size: fraction of test set in the whole dataset
        :param n_epochs: number of epochs, 10 by default
        :param margin: triplet loss margin
        """
        with input_data_path.open('rb') as file_handle:
            self.X, self.y = pickle.load(file_handle)

        self.X, self.y = self.fix_duplicate_classes(self.X, self.y)
        self.input_length = self.X.shape[1]
        self.embedding_length = embedding_length
        self.output_length = output_length if output_length is not None \
            else embedding_length
        self.embeddings_model_path = None
        self.random_state = random_state
        self.test_size = test_size
        self.n_epochs = n_epochs
        self.margin = margin
        self.use_cuda = torch.cuda.is_available()
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=torch.nn.PairwiseDistance(), margin=self.margin)
        self.embedding_model = EmbeddingNet(self.input_length,
                                            self.embedding_length,
                                            self.output_length)
        self.model = TripletNet(self.embedding_model)
        if self.use_cuda:
            self.model.to("cuda")
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the corrected data.

        :return: a tuple of samples and labels numpy arrays
        """
        return self.X, self.y

    @staticmethod
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

    def train_test_split(self) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the dataset into training and testing sets.

        :return: a tuple of X_train, X_test, y_train, y_test
        """
        ss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size,
                                    random_state=self.random_state)

        for train_index, test_index in ss.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            return X_train, X_test, y_train, y_test

        raise Exception('The StratifiedShuffleSplit should have provided '
                        'exactly one split of data to training and testing '
                        'sets')

    def train_embedding(self, X_train: np.ndarray,
                        y_train: np.ndarray, directory: Path) -> 'Executor':
        """Trains the embedding neural network given training samples
        and labels. Once the training phase is completed, the best obtained
        embedding model is reloaded.

        :param X_train: training samples
        :param y_train: training labels
        :param directory: output directory
        :return: self
        """
        loaders = get_data_loaders(X_train, y_train)
        embeddings_model_path = EmbeddingNetSupervisor().train_embedding(
            self.n_epochs, loaders, self.model, self.optimizer,
            self.loss_fn, self.use_cuda, directory).model_path
        self.embedding_model.load_state_dict(torch.load(embeddings_model_path))
        return self

    def train_classifier(self, X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray) -> int:
        """Trains the k-Nearest Neighbors classifier and determines the value
        of k for which model's F1 score is the highest.

        :param X_train: train samples
        :param X_test: test samples
        :param y_train: train labels
        :param y_test: test labels
        :return: the value of best k
        """
        best_classifier_F1 = 0.0
        best_classifier_n = 0
        for n in range(1, 15, 2):
            classifier = KNearestEmbedding(self.embedding_model, n)
            classifier.fit(X_train, y_train)
            F1 = f1_score(classifier.predict(X_test), y_test,
                          average='weighted')

            if F1 > best_classifier_F1:
                best_classifier_F1 = F1
                best_classifier_n = n
            classifier_n_sav_path = MODELS_INVESTIGATION_DIR \
                / f'finalized_model_{n}.sav'
            pickle.dump(classifier, classifier_n_sav_path.open('wb'))
        source_path = MODELS_INVESTIGATION_DIR \
            / f'finalized_model_{best_classifier_n}.sav'
        destination_path = MODELS_DIR / f'finalized_model.sav'
        shutil.copy(source_path, destination_path)
        return best_classifier_n


def determine_best_model() -> int:
    """This function determines the best model by performing the following
    actions:

        - reads the input data
        - splits the data into train and validation sets
        - creates a triplet network model
        - trains the embeddings neural network on training set while watching \
        out for over-fitting on the validation set
        -  trains a series k-NN classifiers on embeddings and returns such k \
        for which the F1 score is maximized.

    :return: k for which the F1 score of a k-NN classifier is maximized.
    """
    executor = Executor()
    X_train, X_test, y_train, y_test = executor.train_test_split()
    executor.train_embedding(X_train, y_train, MODELS_INVESTIGATION_DIR)
    return executor.train_classifier(X_train, X_test, y_train, y_test)


def train_final_model(n_neighbors: int = 9) -> None:
    """This function trains the final model by performing the following actions:

        - reads the input data
        - creates a triplet network model
        - trains the embeddings neural network on a training set while \
        watching out for over-fitting on a validation set
        - the best obtained embedding model is restored from files
        - an instance of :class:`KNearestEmbedding` is created and trained on \
        the whole dataset
        - the trained ``KNearestEmbedding`` model is pickled to a file

    :param n_neighbors: number of neighbors, 9 by default
    :return: None
    """
    executor = Executor()
    X, y = executor.get_data()
    executor.train_embedding(X, y, MODELS_DIR)
    classifier = KNearestEmbedding(executor.embedding_model, n_neighbors) \
        .fit(X, y)
    pickle.dump(classifier.classifier,
                (MODELS_DIR / f'finalized_model.sav').open('wb'))


if __name__ == '__main__':
    train_final_model(determine_best_model())
