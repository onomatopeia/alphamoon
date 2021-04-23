from typing import Type, Any

import numpy as np
import torch
import torch.cuda
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from alphamoon.features.build_features import EmbeddingNet


class KNearestEmbedding:
    """Class to supervise the learning process of the K-Nearest Neighbors
    Classifier of the alphanumeric handwritten characters embeddings.
    """

    def __init__(self, embedding_model: EmbeddingNet,
                 classifier_class: Type[Any],
                 **kwargs: Any) -> None:
        """Initializes an instance of the :class:`ClassifierSupervisor` class.

        :param embedding_model: embedding model
        :param classifier_class: class of the classifier
        :param kwargs: keyword arguments specific for the ``classfier_class``,
            for instance:

            - ``n_neighbours``: number of neighbors in KNN algorithm
        """
        self.embedding_model = embedding_model
        self.use_cuda = torch.cuda.is_available()
        self.classifier = classifier_class(**kwargs)

    def get_embedding(self, X: np.ndarray) -> np.ndarray:
        """Given an array of samples this function returns an array of samples'
        embeddings.

        :param X: array of samples
        :return: array of embeddings
        """
        X = torch.from_numpy(X)
        if self.use_cuda:
            X = X.to("cuda")

        return self.embedding_model.forward(X.float()).detach().to(
            "cpu").numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestEmbedding':
        """Fits the model to the data.

        :param X: samples
        :param y: labels
        :return: self (fitted model)
        """
        X_train_embedded = self.get_embedding(X)
        y_train_embedded = y.ravel()
        self.classifier = self.classifier.fit(X_train_embedded,
                                              y_train_embedded)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given an array of test samples employs the trained model to predict
        associated labels and returns them in the form of an array.

        :param X: test samples array
        :return: predicted classes array
        """
        X_test_embedded = self.get_embedding(X)
        return self.classifier.predict(X_test_embedded)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Given an array of samples and an array of true labels, this function
        returns accuracy score of the trained classifier on the provided set
        of samples.

        :param X: array of samples
        :param y: array of labels
        :return: accuracy score
        """
        y_pred = self.predict(X)
        y_flat = y.ravel()
        return accuracy_score(y_flat, y_pred)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Given an array of samples and an array of true labels, this function
        calculates and prints out performance metrics.

        :param X: array of samples
        :param y: array of labels
        :return: None
        """
        y_pred = self.predict(X)
        y_flat = y.ravel()
        print(self.classifier.__class__.__name__)
        print('Accuracy,Precision,Recall,F1-score')
        print(f'- {accuracy_score(y_flat, y_pred):.6f}')
        print(f'- {precision_score(y_flat, y_pred, average="weighted"):.6f}')
        print(f'- {recall_score(y_flat, y_pred, average="weighted"):.6f}')
        print(f'- {f1_score(y_flat, y_pred, average="weighted"):6f}')
