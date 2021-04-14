import pickle

import numpy as np
import torch.cuda
import torch.nn
import torch.optim
from scipy.ndimage import affine_transform
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from alphamoon.constants import *
from alphamoon.data.make_dataset import get_data_loaders, Phase
from alphamoon.features.build_features import EmbeddingNet, TripletNet


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path='state_dict_model.pt',
          from_epoch=0, valid_loss_min=np.Inf):
    """returns trained model"""

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
            epoch + from_epoch,
            train_loss,
            valid_loss
        ))

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
            try:
                save_model(model, save_path)
            except Exception as e:
                print(f'Could not save the model due to {str(e)}')
            valid_loss_min = valid_loss

    # return trained model
    return model


def save_model(model, file_name='state_dict_model.pt', epoch=-1):
    if epoch > 0:
        full_path = Path(__file__).resolve().parents[2] / 'models' / file_name
        file_stem = full_path.stem
        file_ext = full_path.suffix
        torch.save(model.state_dict(), f'{file_stem}_{epoch}{file_ext}')
    torch.save(model.state_dict(), file_name)


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders[Phase.TEST]):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


if __name__ == '__main__':

    def get_transformation_matrix(img_w_h):
        theta = np.deg2rad(-7)
        ry = rx = img_w_h / 2

        r00 = np.cos(theta)
        r01 = -np.sin(theta)
        r10 = np.sin(theta)
        r11 = np.cos(theta)
        r02 = rx - r00 * rx - r01 * ry
        r12 = ry - r10 * rx - r11 * ry

        rotation = np.array([[r00, r01, r02], [r10, r11, r12], [0, 0, 1]])

        Sx = 0.9
        Sy = 1.1
        Tx = -5
        Ty = 2

        affine2 = np.array([[Sy, 0, Tx], [0, Sx, Ty], [0, 0, 1]])
        trans = np.matmul(rotation, affine2)
        return trans

    input_data_path = RAW_DATA_DIR / 'train.pkl'
    with input_data_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)

    one_shot_example = np.where(y[:, 0] == 30)[0][0]

    img_w_h = int(np.sqrt(X.shape[1]))
    trans = get_transformation_matrix(img_w_h)
    x2 = affine_transform(np.reshape(X[one_shot_example], (img_w_h, img_w_h)), trans)
    x2_ravel = np.ravel(x2)
    X_augmented = np.vstack((X, x2_ravel))
    y_augmented = np.vstack((y, [30]))

    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
    for train_index, test_index in ss.split(X_augmented, y_augmented):
        X_train, X_test = X_augmented[train_index], X_augmented[test_index]
        y_train, y_test = y_augmented[train_index], y_augmented[test_index]

    embedding_model = EmbeddingNet(X_train.shape[1], 64, 64)
    model = TripletNet(embedding_model)

    """model_transfer = models.inception_v3(pretrained=True)
    model_transfer = freeze_weights(model_transfer)
    model_transfer.fc = create_fc_layer(model_transfer.fc, no_classes)
    model_transfer.AuxLogits.fc = create_fc_layer(model_transfer.AuxLogits.fc, no_classes)

    assert model_transfer.fc.out_features == 133, \
        f"Expected 133 classes but got {model_transfer.fc.out_features} (dog_classes = {no_classes})"
    assert model_transfer.AuxLogits.fc.out_features == 133, \
        f"Expected 133 classes in AuxLogits but got {model_transfer.AuxLogits.fc.out_features} (dog_classes = {no_classes})"

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_transfer = model_transfer.cuda()

    criterion_transfer = nn.CrossEntropyLoss()

    lr = 0.01
    opt_class = opt_SGD

    trainable_params = get_trainable_parameters([model_transfer.fc, model_transfer.AuxLogits.fc])
    print(len(trainable_params))
    optimizer_transfer = get_optimizer(opt_class, trainable_params, lr)
    """

    # train the model
    n_epochs = 10
    margin = 1
    use_cuda = True  # torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=torch.nn.PairwiseDistance(), margin=margin)
    optimizer = torch.optim.Adam(model.parameters())

    loaders = get_data_loaders(X_train, y_train, X_test, y_test)
    train(n_epochs, loaders, model, optimizer, loss_fn, use_cuda)

    # load the model that got the best validation accuracy (uncomment the line below)
    # model_transfer.load_state_dict(torch.load(f'model_transfer_{suffix}.pt'))
    # test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
