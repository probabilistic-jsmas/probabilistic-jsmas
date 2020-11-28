from joblib import load, dump

import numpy as np


def _get_mnist_dataset():
    from cleverhans.dataset import MNIST

    dataset = MNIST()
    x_train, y_train = dataset.get_set("train")
    x_test, y_test = dataset.get_set("test")

    return x_train, y_train, x_test, y_test


def _get_cifar10_dataset():
    from cleverhans.dataset import CIFAR10

    dataset = CIFAR10()
    x_train, y_train = dataset.get_set("train")
    x_test, y_test = dataset.get_set("test")

    y_train = y_train.reshape((-1, 10))
    y_test = y_test.reshape((-1, 10))

    return x_train, y_train, x_test, y_test


def load_dataset(dataset_path):
    """
    Loads and returns a dataset.

    Parameters
    ----------
    dataset_path: str
        The name of the dataset.

    Returns
    -------
    x_train, y_train, x_test, y_test: numpy.ndarray
        The dataset arrays.
    """

    if dataset_path == "mnist":
        return _get_mnist_dataset()
    elif dataset_path == "cifar10":
        return _get_cifar10_dataset()

    x_train, y_train, x_test, y_test = load(f"datasets/joblibs/{dataset_path}.joblib")

    return x_train.astype(np.float32), y_train.astype(np.float32), x_test.astype(np.float32), y_test.astype(np.float32)


def save_dataset(dataset_path, x_train, y_train, x_test, y_test):
    """
    Saves a dataset.

    Parameters
    ----------
    dataset_path: str
        The name of the dataset.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    """

    dump((x_train, y_train, x_test, y_test), f"datasets/joblibs/{dataset_path}.joblib")
