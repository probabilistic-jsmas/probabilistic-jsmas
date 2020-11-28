from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Dropout, GlobalAveragePool, Softmax, MLP

from .cleverhans_utils import MaxPooling2D
from .model_utils import model_training, model_testing, substitute_training


EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
LABEL_SMOOTHING = .1

EPOCHS_JBDA = 6
BATCH_SIZE_JBDA = 32
LAMBDA = .1


def train_mnist(model_path, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING):
    """
    Trains a LeNet-5 model on the MNIST dataset.

    Parameters
    ----------
    model_path: str
        The name of the joblib file.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    epochs: int, optional
        The number of epochs.
    batch_size: int, optional
        The batch size.
    learning_rate: float, optional
        The learning rate.
    label_smoothing: float, optional
        The amount of label smoothing used.
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

    model_training(
        model=MLP(layers, (None, 28, 28, 1)), file_name=model_path, x_train=x_train, y_train=y_train, x_test=x_test, 
        y_test=y_test, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        label_smoothing=label_smoothing
    )


def train_cifar10(model_path, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING):
    """
    Trains an AllConvolutional model on the CIFAR10 dataset.

    Parameters
    ----------
    model_path: str
        The name of the joblib file.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    epochs: int, optional
        The number of epochs.
    batch_size: int, optional
        The batch size.
    learning_rate: float, optional
        The learning rate.
    label_smoothing: float, optional
        The amount of label smoothing used.
    """

    layers = [
        Conv2D(64, (3, 3), (1, 1), "SAME"),
        ReLU(),
        Conv2D(128, (3, 3), (1, 1), "SAME"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Conv2D(128, (3, 3), (1, 1), "SAME"),
        ReLU(),
        Conv2D(256, (3, 3), (1, 1), "SAME"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Conv2D(256, (3, 3), (1, 1), "SAME"),
        ReLU(),
        Conv2D(512, (3, 3), (1, 1), "SAME"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Conv2D(10, (3, 3), (1, 1), "SAME"),
        GlobalAveragePool(),
        Softmax()
    ]

    model_training(
        model=MLP(layers, (None, 32, 32, 3)), file_name=model_path, x_train=x_train, y_train=y_train, x_test=x_test,
        y_test=y_test, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        label_smoothing=label_smoothing
    )


def train_gtsrb(model_path, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING):
    """
    Trains a model on the gtsrb dataset.

    Parameters
    ----------
    model_path: str
        The name of the joblib file.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    epochs: int, optional
        The number of epochs.
    batch_size: int, optional
        The batch size.
    learning_rate: float, optional
        The learning rate.
    label_smoothing: float, optional
        The amount of label smoothing used.
    """

    layers = [
        Conv2D(64, (5, 5), (1, 1), "SAME"),
        ReLU(),
        MaxPooling2D((3, 3), (2, 2), "VALID"),
        Conv2D(64, (5, 5), (1, 1), "SAME"),
        ReLU(),
        MaxPooling2D((3, 3), (2, 2), "VALID"),
        Flatten(),
        Linear(384),
        ReLU(),
        Dropout(0.8),
        Linear(192),
        ReLU(),
        Dropout(0.8),
        Linear(43),
        Softmax()
    ]

    model_training(
        model=MLP(layers, (None, 32, 32, 3)), file_name=model_path, x_train=x_train, y_train=y_train, x_test=x_test,
        y_test=y_test, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        label_smoothing=label_smoothing
    )
    
    
def train_gtsrb_substitute(model_path, oracle_path, x_train, y_train, x_test, y_test, epochs=EPOCHS,
                           batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING,
                           epochs_jbda=EPOCHS_JBDA, batch_size_jbda=BATCH_SIZE_JBDA, lamb=LAMBDA):
    """
    Trains a substitute model to mimic an oracle trained on gtsrb.

    Parameters
    ----------
    model_path: str
        The name of the joblib file.
    oracle_path: str
        The name of the oracle.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    epochs: int, optional
        The number of epochs.
    batch_size: int, optional
        The batch size.
    learning_rate: float, optional
        The learning rate.
    label_smoothing: float, optional
        The amount of label smoothing used.
    epochs_jbda: int
        The number of JBDA epochs.
    batch_size_jbda: int
        The size of JBDA batches.
    lamb: float
        The lambda parameter of the JBDA.
    """

    layers = [
        Conv2D(16, (3, 3), (1, 1), 'SAME'),
        ReLU(0.2),
        MaxPooling2D((2, 2), (2, 2), 'VALID'),
        Conv2D(32, (3, 3), (1, 1), 'SAME'),
        ReLU(0.2),
        MaxPooling2D((2, 2), (2, 2), 'VALID'),
        Conv2D(64, (3, 3), (1, 1), 'SAME'),
        ReLU(0.2),
        Flatten(),
        Linear(43),
        Softmax()
    ]

    substitute_training(
        model=MLP(layers, (None, 32, 32, 3)), file_name=model_path, oracle_path=oracle_path, x_train=x_train,
        y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, label_smoothing=label_smoothing, epochs_jbda=epochs_jbda,
        batch_size_jbda=batch_size_jbda, lamb=lamb
    )
    

def test_model(model_path, x_train, y_train, x_test, y_test):
    model_testing(model_path, x_train, y_train, x_test, y_test)