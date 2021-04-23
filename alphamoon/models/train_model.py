import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union, Type, Any, List

import numpy as np
import torch
import torch.cuda
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphamoon.constants import (MODELS_DIR, MODELS_INVESTIGATION_DIR)
from alphamoon.data.make_dataset import (get_data_loader, Phase, get_data,
                                         DataPreProcessing,
                                         augment_training_set)
from alphamoon.features.build_features import EmbeddingNet, TripletNet
from alphamoon.models.classifier import KNearestEmbedding


class EmbeddingNetSupervisor:
    """This class implements the supervision of the learning process of the
    embedding neural network.
    """

    def __init__(self) -> None:
        """Initializes the :class:`EmbeddingNetSupervisor` class.
        """
        self.model_path: Optional[Path] = None
        self.valid_loss_min = np.Inf

    def train_embedding(self, n_epochs: int,
                        loaders: Dict[Phase, DataLoader],
                        model: TripletNet,
                        optimizer: Optimizer,
                        criterion: Module,
                        use_cuda: bool,
                        directory: Path) -> Dict[Phase, List[str]]:
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
        :return: dictionary of formatted losses per iteration per phase
        """
        losses = defaultdict(list)

        for epoch in range(1, n_epochs + 1):
            print(f'Epoch {epoch}')
            train_loss = self.train(model, loaders[Phase.TRAIN], use_cuda,
                                    criterion, optimizer)
            valid_loss = self.validate(model, loaders[Phase.VALIDATION],
                                       use_cuda, criterion)

            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} '
                  f'\tValidation Loss: {valid_loss:.6f}')
            losses[Phase.TRAIN].append(f'{train_loss:.6f}')
            losses[Phase.VALIDATION].append(f'{valid_loss:.6f}')
            self.save_model(valid_loss, model, directory, epoch)

        return losses

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
        embedding_model_path = directory / 'embedding_model.pt'
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


def train_test_split(X, y, test_size, random_state) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits the dataset into training and testing sets.

    :return: a tuple of X_train, X_test, y_train, y_test
    """
    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)

    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test

    raise Exception('The StratifiedShuffleSplit should have provided '
                    'exactly one split of data to training and testing '
                    'sets')


class Executor:
    """This class supervises the learning process of the
    :class:`KNearestEmbedding` classifier.
    """

    def __init__(self, input_length: int,
                 embedding_length: int = 64,
                 output_length: Optional[int] = None, random_state: int = 0,
                 test_size: float = 0.33, n_epochs: int = 10,
                 margin: Union[float, int] = 10) -> None:
        """Initializes an instance of the ``Executor`` class.

        :param input_length: path to the input data `pkl` file
        :param embedding_length: length of the embedding layer
        :param output_length: length of the output layer, optional, assumed to \
            be the same as ``embedding_length`` if not provided
        :param random_state: random state
        :param test_size: fraction of test set in the whole dataset
        :param n_epochs: number of epochs, 10 by default
        :param margin: triplet loss margin
        """
        self.input_length = input_length
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    @staticmethod
    def get_data_loaders(X_train: np.ndarray, X_valid: np.ndarray,
                         y_train: np.ndarray, y_valid: np.ndarray) \
            -> Dict[Phase, DataLoader]:
        """Given training and validation samples and labels this function
        returns a dictionary of phase-keyed data loaders.

        :param X_train: training samples
        :param X_valid: validation samples
        :param y_train: training labels
        :param y_valid: validation labels
        """
        return {
            Phase.TRAIN: get_data_loader(X_train, y_train),
            Phase.VALIDATION: get_data_loader(X_valid, y_valid)
        }

    def train_embedding(self, X_train: np.ndarray, X_valid: np.ndarray,
                        y_train: np.ndarray, y_valid: np.ndarray,
                        directory: Path) -> Dict[Phase, List[str]]:
        """Trains the embedding neural network given training samples
        and labels. Once the training phase is completed, the best obtained
        embedding model is reloaded.

        :param X_train: training samples
        :param X_valid: validation samples
        :param y_train: training labels
        :param y_valid: validation labels
        :param directory: output directory
        :return: dictionary of formatted losses per iteration per phase
        """
        loaders = self.get_data_loaders(X_train, X_valid, y_train, y_valid)
        embedding_net_supervisor = EmbeddingNetSupervisor()
        losses = embedding_net_supervisor.train_embedding(
            self.n_epochs, loaders, self.model, self.optimizer,
            self.loss_fn, self.use_cuda, directory)
        embeddings_model_path = embedding_net_supervisor.model_path
        self.embedding_model.load_state_dict(torch.load(embeddings_model_path))
        return losses

    def train_classifier(self, X_train: np.ndarray, X_valid: np.ndarray,
                         y_train: np.ndarray, y_valid: np.ndarray,
                         classifier_class: Type[Any],
                         classifier_kwargs: List[Dict[str, Any]]) \
            -> Tuple[KNearestEmbedding, Dict[str, Any], float]:
        """Trains an instance of ``classifier_class`` with each set of
        hyperparameters in ``classifier_kwargs`` and determines those
        hyperparameters for which the F1 score obtained by the classifier
         is the highest.

        :param X_train: train samples
        :param X_valid: test samples
        :param y_train: train labels
        :param y_valid: test labels
        :param classifier_class: classifier class
        :param classifier_kwargs: a list of dictionaries of classifier's
            hyperparameters
        :return: the best model, set of hyperparameters and the best obtained F1
        """
        best_classifier_f1 = 0.0
        best_params_idx = 0
        best_classifier = None
        for i, hyperparameters in enumerate(classifier_kwargs):
            classifier = KNearestEmbedding(self.embedding_model,
                                           classifier_class,
                                           **hyperparameters)
            classifier.fit(X_train, y_train)
            F1 = f1_score(y_valid, classifier.predict(X_valid),
                          average='weighted')
            print(hyperparameters)
            classifier.evaluate(X_valid, y_valid)
            if F1 > best_classifier_f1:
                best_classifier_f1 = F1
                best_params_idx = i
                best_classifier = classifier
                classifier_n_sav_path = MODELS_INVESTIGATION_DIR \
                                        / f'finalized_model_{i}.sav'
                pickle.dump(classifier, classifier_n_sav_path.open('wb'))

        if best_classifier is None:
            if len(classifier_kwargs) == 0:
                raise ValueError('No classifier kwargs provided')
            raise ValueError('No classifier attained positive F1-score')

        source_path = MODELS_INVESTIGATION_DIR \
                      / f'finalized_model_{best_params_idx}.sav'
        destination_path = MODELS_DIR / f'finalized_model.sav'
        shutil.copy(source_path, destination_path)

        return (best_classifier, classifier_kwargs[best_params_idx],
                best_classifier_f1)


def determine_best_model(embedding_length: int,
                         margin: Union[float, int],
                         knn_params: List[Dict[str, Any]],
                         data_pre_processing: DataPreProcessing
                         = DataPreProcessing.FIX_DUPLICATE_CLASSES,
                         evaluate_on_test_dataset: bool = False) \
        -> Dict[str, Any]:
    """This function determines the best model by performing the following
    actions:

        - reads the input data
        - splits the data into train and validation sets
        - creates a triplet network model
        - trains the embeddings neural network on training set while watching \
        out for over-fitting on the validation set
        -  trains a series k-NN classifiers on embeddings and returns such k \
        for which the F1 score is maximized.

    :param embedding_length: length of embedding
    :param margin: value of the margin in the triplet loss function
    :param knn_params: list of classifier's parameters dictionaries
    :param data_pre_processing: an action that shall be done to the data in \
        the preprocessing stage
    :param evaluate_on_test_dataset: a flag whether to evaluate the final \
        model on the test dataset, False by default
    :return: dictionary of classifier's parameters
    """

    random_state = 0
    X, y = get_data(data_pre_processing=data_pre_processing)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.2,
                                                          random_state=random_state)

    if data_pre_processing == DataPreProcessing.DUPLICATE_TRAINING:
        X_train, y_train = augment_training_set(X_train, y_train)

    executor = Executor(X.shape[1], margin=margin,
                        embedding_length=embedding_length)
    losses = executor.train_embedding(X_train, X_valid, y_train, y_valid,
                                      MODELS_INVESTIGATION_DIR)

    print(f'embedding_length={embedding_length}, margin={margin}')
    print(losses)
    classifier, params, f1score = executor \
        .train_classifier(X_train, X_valid, y_train, y_valid,
                          KNeighborsClassifier, knn_params)
    print(params, f1score)
    if evaluate_on_test_dataset:
        print('Evaluation on the test dataset')
        classifier.evaluate(X_test, y_test)
    return params


def train_final_model(classifier_class: Type[Any] = KNeighborsClassifier,
                      **classifier_params: Any) -> None:
    """This function trains the final model by performing the following actions:

        - reads the input data
        - creates a triplet network model
        - trains the embeddings neural network on a training set while \
        watching out for over-fitting on a validation set
        - the best obtained embedding model is restored from files
        - an instance of :class:`KNearestEmbedding` is created and trained on \
        the whole dataset
        - the trained ``KNearestEmbedding`` model is pickled to a file

    :param classifier_class: classifier class, KNeighborsClassifier by default
    :param classifier_params: classifier parameters as keyword arguments
    :return: None
    """
    random_state = 0
    X, y = get_data()
    executor = Executor(X.shape[1])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.2,
                                                          random_state=random_state)
    executor.train_embedding(X_train, X_valid, y_train, y_valid,
                                      MODELS_DIR)
    classifier = KNearestEmbedding(executor.embedding_model,
                                   classifier_class, **classifier_params) \
        .fit(X, y)
    pickle.dump(classifier.classifier,
                (MODELS_DIR / f'finalized_model.sav').open('wb'))


if __name__ == '__main__':

    params = [dict(n_neighbors=n) for n in range(1, 15, 2)]
    best_params = determine_best_model(64, 10, params,
                                       evaluate_on_test_dataset=True)
    train_final_model(**best_params)

