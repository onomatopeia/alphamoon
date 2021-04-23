import pickle
from pathlib import Path

import numpy as np
import torch
import torch.cuda

from sklearn.neighbors import KNeighborsClassifier

from alphamoon.constants import (MODELS_DIR, EMBEDDING_MODEL_FILENAME,
                                 CLASSIFICATION_MODEL_FILENAME)
from alphamoon.features.build_features import EmbeddingNet


class Classifier:
    """Multiclass classifier of handwritten digits and letters.
    """

    def __init__(self, folder: Path = MODELS_DIR,
                 use_cuda: bool = True) -> None:
        """Initializes an instance of the Classifier class.

        :param folder: existing folder where trained embedding model and \
        trained classifier are kept
        :param use_cuda: a flag whether cuda shall be used
        :return: None
        """
        self.embedding_model = EmbeddingNet(3136, 64, 64)
        self.embedding_model.load_state_dict(
            torch.load(folder / EMBEDDING_MODEL_FILENAME))
        pickled_model_path = folder / CLASSIFICATION_MODEL_FILENAME
        with pickled_model_path.open('rb') as pickled_model:
            self.classifier: KNeighborsClassifier = pickle.load(pickled_model)
        self.cuda = use_cuda and torch.cuda.is_available()
        if self.cuda:
            self.embedding_model.cuda()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the class labels for the provided data.

        :param X_test: Test samples
        :return: Class labels for each data sample.
        """
        X_test_data = torch.from_numpy(X_test)
        if self.cuda:
            X_test_data = X_test_data.cuda()

        X_test_embedded = self.embedding_model.forward(X_test_data.float()) \
            .detach().cpu().numpy()
        return self.classifier.predict(X_test_embedded)[..., np.newaxis]
