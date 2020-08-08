"""
LeNet5 architecture from MNIST trained with the augmented dataset
"""

from cleverhans.dataset import MNIST
from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Softmax, MLP

from models.cleverhans_utils import MaxPooling2D
from models.model_utils import model_training, model_testing

import numpy as np

TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
AUGMENT_SIZE = 20000
NB_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def model_train(attack):
    """
    Creates the joblib file of LeNet-5 trained over the augmented MNIST dataset.

    Parameters
    ----------
    attack: str
        The augmented dataset used (either "jsma", "wjsma" or "tjsma").
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

    mnist = MNIST(train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    x_add = np.load("defense/augmented/" + attack + "_x.npy")[:AUGMENT_SIZE]
    y_add = np.load("defense/augmented/" + attack + "_y.npy")[:AUGMENT_SIZE]

    x_train = np.concatenate((x_train, x_add.reshape(x_add.shape + (1,))), axis=0).astype(np.float32)
    y_train = np.concatenate((y_train, y_add), axis=0).astype(np.float32)

    model_training(model, "mnist_defense_" + attack + ".joblib", x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS,
                   batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)


def model_test(attack):
    """
    Evaluates the performances of the model over the original MNIST test set and the augmented set.

    Parameters
    ----------
    attack: str
        The augmented dataset used (either "jsma", "wjsma" or "tjsma").
    """

    mnist = MNIST(train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    print("ORIGINAL MNIST TEST")

    model_testing("mnist_defense_" + attack + ".joblib", x_train, y_train, x_test, y_test)

    x_add = np.load("defense/augmented/" + attack + "_x.npy")[:AUGMENT_SIZE]
    y_add = np.load("defense/augmented/" + attack + "_y.npy")[:AUGMENT_SIZE]

    x_train = np.concatenate((x_train, x_add.reshape(x_add.shape + (1,))), axis=0).astype(np.float32)
    y_train = np.concatenate((y_train, y_add), axis=0).astype(np.float32)

    print("====================")
    print("AUGMENTED MNIST TEST")

    model_testing("mnist_defense_" + attack + ".joblib", x_train, y_train, x_test, y_test)
