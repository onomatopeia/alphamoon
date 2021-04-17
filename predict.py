from alphamoon.models import predict_model


def predict(input_data):
    return predict_model.Classifier().predict(input_data)
