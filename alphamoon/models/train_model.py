import pickle
import shutil
from typing import Dict

import numpy as np
import torch.cuda
import torch.nn
import torch.optim
from sklearn.metrics import (f1_score, accuracy_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphamoon.constants import *
from alphamoon.data.make_dataset import get_data_loaders, Phase
from alphamoon.features.build_features import EmbeddingNet, TripletNet


class EmbeddingNetSupervisor:

    def __init__(self):
        self.model_path = None
        self.valid_loss_min = np.Inf

    def train_embedding(self, n_epochs: int,
                        loaders: Dict[Phase, DataLoader],
                        model: TripletNet,
                        optimizer: torch.optim.Optimizer,
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
        :return: path to the pickle of the trained model
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

    def save_model(self, valid_loss, model, directory, epoch):
        """Save the model if validation loss has decreased.

        :param valid_loss: validation loss
        :param model: model to evaluate
        :param directory: output directory to save the model
        :param epoch: epoch
        :return: tuple consisting of the path to the model file and of the \
        current minimal validation loss
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
    def move_to_gpu(data, use_cuda):
        if use_cuda:
            return tuple(d.to("cuda") for d in data)
        return data

    def train(self, model, data_loader, use_cuda, criterion, optimizer):
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

    def validate(self, model, data_loader, use_cuda, criterion):
        valid_loss = 0.0
        model.eval()
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data = self.move_to_gpu(data, use_cuda)
            output = model.forward(*data)
            loss = criterion(*output)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        return valid_loss

    @staticmethod
    def save_embedding_model(model, directory=MODELS_DIR, epoch=-1):
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


class ClassifierSupervisor(object):
    def __init__(self, embedding_model, n_neighbours):
        self.embedding_model = embedding_model
        self.use_cuda = torch.cuda.is_available()
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbours)

    def get_embedding(self, X):
        X = torch.from_numpy(X)
        if self.use_cuda:
            X = X.to("cuda")

        return self.embedding_model.forward(X.float()).detach().to("cpu").numpy()

    def fit(self, X, y):
        X_train_embedded = self.get_embedding(X)
        y_train_embedded = y.ravel()
        self.classifier = self.classifier.fit(X_train_embedded, y_train_embedded)
        return self

    def predict(self, X):
        X_test_embedded = self.get_embedding(X)
        return self.classifier.predict(X_test_embedded)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_flat = y.ravel()
        return accuracy_score(y_flat, y_pred)


"""weighted_f1 = f1_score(y_flat, y_pred, average='weighted')
print('F1 =', weighted_f1)
print('Precision =',
      precision_score(y_test_flat, y_pred, average='weighted'))
print('Recall =', recall_score(y_test_flat, y_pred, average='weighted'))
print('Accuracy =', )
return weighted_f1"""


class Executor:
    def __init__(self, input_data_path=RAW_DATA_DIR / 'train.pkl', embedding_length=64, output_length=None, random_state=0, test_size=0.33):
        with input_data_path.open('rb') as file_handle:
            self.X, self.y = pickle.load(file_handle)

        self.X, self.y = self.fix_duplicate_classes(self.X, self.y)
        self.input_length = self.X.shape[1]
        self.embedding_length = embedding_length
        self.output_length = output_length if output_length is not None else embedding_length
        self.embedding_model = None
        self.model = None
        self.random_state = random_state
        self.test_size = test_size
        self.n_epochs = 10
        self.margin = 10
        self.use_cuda = torch.cuda.is_available()
        self.optimizer = None
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=torch.nn.PairwiseDistance(), margin=self.margin)

    def run(self):
        self._train_test_split()
        self.create_model()
        embeddings_model_path = self.train_embeddings()
        self.train_classifier(embeddings_model_path)

    def create_model(self):
        self.embedding_model = EmbeddingNet(self.input_length, self.embedding_length, self.output_length)
        self.model = TripletNet(self.embedding_model)
        if self.use_cuda:
            self.model.to("cuda")
        self.optimizer = torch.optim.Adam(self.model.parameters())
        return self

    @staticmethod
    def fix_duplicate_classes(X, y):
        one_shot_example = np.where(y[:, 0] == 30)[0][0]

        X_augmented = X.copy()
        y_augmented = y.copy()
        y_augmented[one_shot_example] = 14
        return X_augmented, y_augmented

    def _train_test_split(self):
        ss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)

        for train_index, test_index in ss.split(self.X, self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            return X_train, X_test, y_train, y_test

        raise Exception('This should never happen')

    def train_embeddings(self, X_train, y_train):
        loaders = get_data_loaders(X_train, y_train)
        embeddings_model = EmbeddingNetSupervisor().train_embedding(
            self.n_epochs, loaders, self.model, self.optimizer,
            self.loss_fn, self.use_cuda,
            MODELS_INVESTIGATION_DIR)
        return embeddings_model.model_path

    def train_classifier(self, embeddings_model_path):
        # train the classifier
        self.embedding_model.load_state_dict(torch.load(embeddings_model_path))
        best_classifier_F1 = 0.0
        best_classifier_n = 0
        for n in range(1, 15, 2):
            classifier = ClassifierSupervisor(self.embedding_model, n)
            classifier.fit(self.X_train, self.y_train)
            F1 = f1_score(classifier.predict(self.X_test), self.y_test, average='weighted')

            if F1 > best_classifier_F1:
                best_classifier_F1 = F1
                best_classifier_n = n
            classifier_n_sav_path = MODELS_INVESTIGATION_DIR / f'finalized_model_{n}.sav'
            pickle.dump(classifier, classifier_n_sav_path.open('wb'))
        source_path = MODELS_INVESTIGATION_DIR / f'finalized_model_{best_classifier_n}.sav'
        destination_path = MODELS_DIR / f'finalized_model.sav'
        shutil.copy(source_path, destination_path)


def determine_best_model():
    Executor().run()


def train_final_model():
    executor = Executor()
    X, y = executor.X, executor.y
    embeddings_model_path = executor.create_model().train_embeddings(X, y)
    embedding_model = executor.embedding_model
    embedding_model.load_state_dict(torch.load(embeddings_model_path))
    classifier = ClassifierSupervisor(embedding_model, 9)
    classifier.fit(executor.X, executor.y)
    pickle.dump(classifier.classifier, (MODELS_DIR / f'finalized_model.sav').open('wb'))


if __name__ == '__main__':
    # determine_best_model()
    train_final_model()
