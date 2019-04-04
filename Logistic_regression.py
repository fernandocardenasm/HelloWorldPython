import numpy as np
import tensorflow as tf


class DataService:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist

    def load_mnist_dataset(self) -> tuple:
        # tuple: (x_train, y_train), (x_test, y_test)
        # each variable of type numpy.ndarray
        return self.mnist.load_data()


class LogisticRegression:

    def main(self):
        data_service = DataService()
        (x_train, y_train), (x_test, y_test) = data_service.load_mnist_dataset()
        x_train = tf.keras.utils.normalize(x_train, axis=-1, order=2)
        x_test = tf.keras.utils.normalize(x_test, axis=-1, order=2)

        # reshape: (n_row * n_column, m)
        X = np.reshape(x_train, (x_train.shape[0], -1)).T

        # Includes the number of elements for x as part of the layer_sizes
        layer_sizes = [30, 40, 50]
        layer_sizes.insert(0, X.shape[0])
        parameters = self.init_parameters(layer_sizes=layer_sizes)

    def init_parameters(self, layer_sizes: [int]) -> dict:
        num_layers = len(layer_sizes)
        if num_layers < 2:
            print("The number of layers must be greater than 2.")
            return

        parameters = {}

        for layer_index in range(1, num_layers):
            parameters["W" + str(layer_index)] = np.random.randn(layer_sizes[layer_index], layer_sizes[layer_index - 1])
            parameters["b" + str(layer_index)] = np.zeros((layer_sizes[layer_index], 1))

        # Take the last element of layer_sizes
        parameters["W" + str(layer_index)] = np.random.randn(1, layer_sizes[num_layers - 1])
        parameters["b" + str(layer_index)] = np.zeros((1, 1))

        return parameters

    def linear_forward(self, W: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(W, A) + b

    def sigmoid(self, Z: np.ndarray):
        return 1/(1 + np.exp(-Z))


if __name__ == '__main__':
    LogisticRegression().main()
