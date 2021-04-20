from alphamoon.models import predict_model


def predict(input_data):
    return predict_model.Classifier().predict(input_data)


def test_predict():
    from alphamoon.constants import RAW_DATA_DIR
    import pickle
    input_data_path = RAW_DATA_DIR / 'train.pkl'

    with input_data_path.open('rb') as file_handle:
        X, y = pickle.load(file_handle)
    result = predict(X)
    assert y.shape == result.shape
