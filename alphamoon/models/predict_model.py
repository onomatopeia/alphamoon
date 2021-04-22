import pickle
import time
import numpy as np
import torch
import torch.cuda
from pathlib import Path
from alphamoon.constants import (MODELS_DIR, EMBEDDING_MODEL_FILENAME,
                                 CLASSIFICATION_MODEL_FILENAME, RAW_DATA_DIR)
from alphamoon.features.build_features import EmbeddingNet
from alphamoon.models.train_model import ClassifierSupervisor  # PEP8


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
            self.classifier = pickle.load(pickled_model)
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

        X_test_embedded = self.embedding_model.forward(X_test_data.float())
        return self.classifier.predict(X_test_embedded.detach().cpu().numpy())


if __name__ == '__main__':
    input_data_path = RAW_DATA_DIR / 'train.pkl'

    with input_data_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)
    start = time.time()
    Classifier().predict(X)
    end = time.time() - start
    print(end)
    start = time.time()
    Classifier(use_cuda=False).predict(X)
    end = time.time() - start
    print(end)
