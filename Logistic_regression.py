import numpy as np
import tensorflow as tf


class DataService:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
        print(type(self.mnist))

    def load_mnist_dataset(self) -> tuple:
        # tuple: (x_train, y_train), (x_test, y_test)
        # each variable of type numpy.ndarray
        return self.mnist.load_data()


class LogisticRegression:
    def foward_propagation(self):
        print("Something")

    def main(self):
        data_service = DataService()
        (x_train, y_train), (x_test, y_test) = data_service.load_mnist_dataset()


if __name__ == '__main__':

    LogisticRegression().main()
