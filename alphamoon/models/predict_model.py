import torch
import torch.cuda
import pickle
import time
from alphamoon.constants import *
from alphamoon.features.build_features import EmbeddingNet


class Classifier:

    def __init__(self, folder=MODELS_DIR, use_cuda=True):
        self.embedding_model = EmbeddingNet(3136, 64, 64)
        self.embedding_model.load_state_dict(torch.load(folder / EMBEDDING_MODEL_FILENAME))
        with (folder / CLASSIFICATION_MODEL_FILENAME).open('rb') as pickled_model:
            self.classifier = pickle.load(pickled_model)
        self.cuda = use_cuda and torch.cuda.is_available()
        if self.cuda:
            self.embedding_model.cuda()
        print(self.cuda)

    def predict(self, X_test):
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

