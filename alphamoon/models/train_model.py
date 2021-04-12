import os

import numpy as np
import torch.cuda
from alphamoon.data.make_dataset import Phase


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, from_epoch=0, valid_loss_min=np.Inf):
    """returns trained model"""

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders[Phase.TRAIN]):
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
        """model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model.forward(data)
            loss = criterion(output, target)
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
            valid_loss_min = valid_loss"""

    # return trained model
    return model


def save_model(model, save_path, epoch=-1):
    if epoch > 0:
        file_name, file_ext = os.path.splitext(save_path)
        torch.save(model.state_dict(), f'{file_name}_{epoch}{file_ext}')
    torch.save(model.state_dict(), save_path)


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


"""if __name__ == '__main__':
    model = TripletNet()
    model_transfer = models.inception_v3(pretrained=True)
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

    opt_SGD = 'SGD'
    opt_Adam = 'Adam'
    opt_RMSprop = 'RMSprop'

    criterion_transfer = nn.CrossEntropyLoss()

    lr = 0.01
    opt_class = opt_SGD

    trainable_params = get_trainable_parameters([model_transfer.fc, model_transfer.AuxLogits.fc])
    print(len(trainable_params))
    optimizer_transfer = get_optimizer(opt_class, trainable_params, lr)

    # get data loaders

    dog_images_dir = os.path.join('data', 'dogImages')
    train_dir = os.path.join(dog_images_dir, 'train')
    valid_dir = os.path.join(dog_images_dir, 'valid')
    test_dir = os.path.join(dog_images_dir, 'test')

    loaders_transfer = dict(train=get_data_loader(train_dir, 'train', 224),
                           valid=get_data_loader(valid_dir, 'valid', 224),
                           test=get_data_loader(test_dir, 'test', 224))

    # train the model

    n_epochs = 100
    suffix = f'_{opt_class}_{n_epochs}_{lr}'
    model_transfer = train(n_epochs, loaders_transfer, model_transfer, optimizer_transfer,
                           criterion_transfer, use_cuda, f'model_transfer_{suffix}.pt')

    # load the model that got the best validation accuracy (uncomment the line below)
    model_transfer.load_state_dict(torch.load(f'model_transfer_{suffix}.pt'))
    test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
"""
