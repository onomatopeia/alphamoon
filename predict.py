import numpy as np

from alphamoon.models import predict_model


def predict(input_data: np.ndarray) -> np.ndarray:
    """Given input data numpy array, this function returns a single-column
    numpy array with predicted labels for each sample in the input data set.

    :param input_data: input data numpy array of shape n_samples x n_features
    :return: response numpy array of shape n_samples x 1
    """
    return predict_model.Classifier().predict(input_data)


def test_predict() -> None:
    """Unit test asserting that the shape of the array returned by the
    :meth:`predict` method is as expected.

    :return: None
    """
    from alphamoon.constants import RAW_DATA_DIR
    import pickle
    input_data_path = RAW_DATA_DIR / 'train.pkl'

    with input_data_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)
    result = predict(X)
    assert y.shape == result.shape
