import pickle

import numpy as np
import torch.cuda
import torch.cuda
import torch.nn
import torch.nn
import torch.optim
import torch.optim
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import shutil
from alphamoon.constants import *
from alphamoon.data.make_dataset import get_data_loaders, Phase
from alphamoon.features.build_features import EmbeddingNet, TripletNet


def train_embedding(n_epochs, loaders, model, optimizer, criterion, use_cuda, directory, valid_loss_min=np.Inf):
    """returns trained model"""
    embeddings_model_path = None

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}')
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(loaders[Phase.TRAIN])):
            # move to GPU
            if use_cuda:
                data = tuple(d.cuda() for d in data)

            optimizer.zero_grad()
            output = model(*data)
            loss = criterion(*output)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(tqdm(loaders[Phase.VALIDATION])):
            # move to GPU
            if use_cuda:
                data = tuple(d.cuda() for d in data)
            output = model.forward(*data)
            loss = criterion(*output)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
            try:
                embeddings_model_path = save_embedding_model(model, directory, epoch=epoch)
            except Exception as e:
                print(f'Could not save the model due to {str(e)}')
            valid_loss_min = valid_loss

    # return the path to the trained model
    return embeddings_model_path


def save_embedding_model(model, directory=MODELS_DIR, epoch=-1):
    embedding_model_path = directory / 'embedding_model.pt'
    state_dict_model_path = directory / 'state_dict_model.pt'
    if epoch > 0:
        torch.save(model.embedding_net.state_dict(), directory / f'embedding_model_{epoch}.pt')
        torch.save(model.state_dict(), directory / f'state_dict_model_{epoch}.pt')

    torch.save(model.embedding_net.state_dict(), embedding_model_path)
    torch.save(model.state_dict(), state_dict_model_path)
    print(f'Model saved to {state_dict_model_path}, embedding model saved to {embedding_model_path}')
    return embedding_model_path


def train_classifier(embedding_model, X_train, y_train, use_cuda, n_neighbours):
    from sklearn.neighbors import KNeighborsClassifier

    X_train_data = torch.from_numpy(X_train)
    if use_cuda:
        X_train_data = X_train_data.cuda()

    X_train_embedded = embedding_model.forward(X_train_data.float())

    classifier = KNeighborsClassifier(n_neighbors=n_neighbours)
    classifier = classifier.fit(X_train_embedded.detach().cpu().numpy(), y_train.ravel())
    return classifier


def test_classifier(classifier, embedding_model, X_test, y_test, use_cuda, n_neighbours):
    from sklearn.metrics import f1_score, precision_score, recall_score

    X_test_data = torch.from_numpy(X_test)
    if use_cuda:
        X_test_data = X_test_data.cuda()

    X_test_embedded = embedding_model.forward(X_test_data.float())
    y_pred = classifier.predict(X_test_embedded.detach().cpu().numpy())

    print('n =', n_neighbours)
    weighted_f1 = f1_score(y_test.ravel(), y_pred, average='weighted')
    print('F1 =', weighted_f1)
    print('Precision =', precision_score(y_test.ravel(), y_pred, average='weighted'))
    print('Recall =', recall_score(y_test.ravel(), y_pred, average='weighted'))
    return weighted_f1


def determine_best_model():
    # read data
    input_data_path = RAW_DATA_DIR / 'train.pkl'
    with input_data_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)

    # fix the data i.e. change the class 30 to 14
    one_shot_example = np.where(y[:, 0] == 30)[0][0]

    X_augmented = X.copy()
    y_augmented = y.copy()
    y_augmented[one_shot_example] = 14

    # prepare train and test
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
    for train_index, test_index in ss.split(X_augmented, y_augmented):
        X_train, X_test = X_augmented[train_index], X_augmented[test_index]
        y_train, y_test = y_augmented[train_index], y_augmented[test_index]

    # define the embedding model
    embedding_length = 64
    embedding_model = EmbeddingNet(X_train.shape[1], embedding_length, embedding_length)
    model = TripletNet(embedding_model)

    # set up everything
    n_epochs = 10
    margin = 10
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(), margin=margin)
    optimizer = torch.optim.Adam(model.parameters())

    # data loaders
    loaders = get_data_loaders(X_train, y_train)

    # train embeddings
    embeddings_model_path = train_embedding(n_epochs, loaders, model, optimizer, loss_fn, use_cuda,
                                            MODELS_INVESTIGATION_DIR)

    # train the classifier
    embedding_model.load_state_dict(torch.load(embeddings_model_path))
    best_classifier_F1 = 0.0
    best_classifier_n = 0
    for n in range(1, 15, 2):
        classifier = train_classifier(embedding_model, X_train, y_train, use_cuda, n)
        F1 = test_classifier(classifier, embedding_model, X_test, y_test, use_cuda, n)

        if F1 > best_classifier_F1:
            best_classifier_F1 = F1
            best_classifier_n = n
        pickle.dump(classifier, (MODELS_INVESTIGATION_DIR / f'finalized_model_{n}.sav').open('wb'))
    shutil.copy(MODELS_INVESTIGATION_DIR / f'finalized_model_{best_classifier_n}.sav',
                (MODELS_DIR / f'finalized_model.sav'))


def main():
    # read data
    input_data_path = RAW_DATA_DIR / 'train.pkl'
    with input_data_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)

    # fix the data i.e. change the class 30 to 14
    one_shot_example = np.where(y[:, 0] == 30)[0][0]

    X_augmented = X.copy()
    y_augmented = y.copy()
    y_augmented[one_shot_example] = 14

    # define the embedding model
    embedding_length = 64
    embedding_model = EmbeddingNet(X.shape[1], embedding_length, embedding_length)
    model = TripletNet(embedding_model)

    # set up everything
    n_epochs = 10
    margin = 10
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(), margin=margin)
    optimizer = torch.optim.Adam(model.parameters())

    # data loaders
    loaders = get_data_loaders(X_augmented, y_augmented)

    # train embeddings
    embeddings_model_path = train_embedding(n_epochs, loaders, model, optimizer, loss_fn, use_cuda, MODELS_DIR)

    # train the classifier
    embedding_model.load_state_dict(torch.load(embeddings_model_path))
    classifier = train_classifier(embedding_model, X_augmented, y_augmented, use_cuda, n_neighbours=9)
    pickle.dump(classifier, (MODELS_DIR / f'finalized_model.sav').open('wb'))


if __name__ == '__main__':
    determine_best_model()
    # main()
