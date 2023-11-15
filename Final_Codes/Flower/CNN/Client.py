import flwr as fl
import tensorflow as tf

from Main import x_train, y_train, x_test, y_test
from Model import create_cnn_model

model = create_cnn_model()

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}
    
def start_client():
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())