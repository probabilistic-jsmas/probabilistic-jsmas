"""
LeNet5 architecture for MNIST
"""

from cleverhans.dataset import MNIST
from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Softmax, MLP

from models.cleverhans_utils import MaxPooling2D
from models.model_utils import model_training, model_testing


FILE_NAME = "mnist.joblib"


def model_train(file_name=FILE_NAME):
    """
    Creates the joblib file of LeNet-5 trained over the MNIST dataset.

    Parameters
    ----------
    file_name: str, optional
        The name of the joblib file.
    """

    layers = [
        Conv2D(20, (5, 5), (1, 1), "VALID"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Conv2D(50, (5, 5), (1, 1), "VALID"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Flatten(),
        Linear(500),
        ReLU(),
        Linear(10),
        Softmax()
    ]

    model = MLP(layers, (None, 28, 28, 1))

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    model_training(model, file_name, x_train, y_train, x_test, y_test, nb_epochs=20, batch_size=128,
                   learning_rate=0.001)


def model_test(file_name=FILE_NAME):
    """
    Evaluates the performances of the model over the MNIST dataset.

    Parameters
    ----------
    file_name: str, optional
        The name of the joblib file.
    """

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    model_testing(file_name, x_train, y_train, x_test, y_test)
