"""
Useful function to test and train models adapted from the cleverhans library
"""
from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

from attacks import get_batch_indices

import tensorflow as tf
import numpy as np

import joblib


EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
LABEL_SMOOTHING = .1

INITIAL_SUB_SIZE = 1000
EPOCHS_JBDA = 6
BATCH_SIZE_JBDA = 32
LAMBDA = .1


def _do_eval(session, x, y, predictions, x_set, y_set, params, accuracy_type):
    """
    Evaluates a model and prints out the results.

    Parameters
    ----------
    session: tf.Session
        The tf Session used.
    x: tf.Tensor
        The input placeholder.
    y: tf.Tensor
        The output placeholder.
    predictions: tf.Tensor
        The symbolic logits output of the model.
    x_set: numpy.ndarray
        The input array.
    y_set: numpy.ndarray
        The output array.
    params: dict
        The evaluation parameters.
    accuracy_type: str
        The type of set used for the evaluation (either "Train" or "Test")
    """

    print(f"{accuracy_type} accuracy: {model_eval(session, x, y, predictions, x_set, y_set, args=params):0.4f}")


def evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, params):
    """
    Evaluates a model and prints out the results.

    Parameters
    ----------
    session: tf.Session
        The tf Session used.
    x: tf.Tensor
        The input placeholder.
    y: tf.Tensor
        The output placeholder.
    predictions: tf.Tensor
        The symbolic logits output of the model.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    params: dict
        The evaluation parameters.
    """

    _do_eval(session, x, y, predictions, x_train, y_train, params, "Train")
    _do_eval(session, x, y, predictions, x_test, y_test, params, "Test")


def model_training(model, file_name, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING):
    """
    Trains the model with the specified parameters.

    Parameters
    ----------
    model: cleverhans.model.Model
        The cleverhans picklable model
    file_name: str
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
    
    session = tf.Session()

    img_rows, img_cols, channels = x_train.shape[1:4]
    classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, classes))

    train_params = {
        "nb_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    eval_params = {"batch_size": batch_size}

    predictions = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)
    
    def train_evaluation():
        """
        Prints the performances of the models after each epoch.
        """
        
        evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, eval_params)

    train(session, loss, x_train, y_train, evaluate=train_evaluation, args=train_params, var_list=model.get_params())

    with session.as_default():
        joblib.dump(model, f"models/joblibs/{file_name}.joblib")


def model_testing(file_name, x_train, y_train, x_test, y_test):
    """
    Runs the evaluation and prints out the results.

    Parameters
    ----------
    file_name: str
        The name of the joblib file.
    x_train: numpy.ndarray
        The input array of the train dataset.
    y_train: numpy.ndarray
        The output array of the train dataset.
    x_test: numpy.ndarray
        The input array of the test dataset.
    y_test: numpy.ndarray
        The output array of the test dataset.
    """

    session = tf.Session()

    with session.as_default():
        model = joblib.load(f"models/joblibs/{file_name}.joblib")

    img_rows, img_cols, channels = x_train.shape[1:4]
    classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, classes))

    eval_params = {"batch_size": 128}

    predictions = model.get_logits(x)

    evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, eval_params)


def substitute_training(model, file_name, oracle_path, x_train, y_train, x_test, y_test,
                        initial_sub_size=INITIAL_SUB_SIZE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE, label_smoothing=LABEL_SMOOTHING, epochs_jbda=EPOCHS_JBDA,
                        batch_size_jbda=BATCH_SIZE_JBDA, lamb=LAMBDA):
    """
    Trains a substitute model using the Jacobian-Based Dataset Augmentation (see https://arxiv.org/pdf/1602.02697.pdf).

    Parameters
    ----------
    model: cleverhans.model.Model
        The cleverhans picklable model
    file_name: str
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
    initial_sub_size: int
        The initial size of the subset.
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

    session = tf.Session()

    with session.as_default():
        model_oracle = joblib.load(f"models/joblibs/{oracle_path}.joblib")

    width, height, depth = x_train.shape[1:]
    classes = y_train.shape[1]

    x_in = tf.placeholder(tf.float32, shape=(None, width, height, depth))
    y_in = tf.placeholder(tf.float32, shape=(None, classes))

    logits = model.get_logits(x_in)

    y_out_substitute = tf.nn.softmax(logits)
    y_out_oracle = model_oracle(x_in)

    target_class = tf.reshape(tf.one_hot(tf.argmax(y_in, axis=1), depth=classes), shape=(-1, classes, 1, 1, 1))

    derivatives = tf.reshape(batch_jacobian(y_out_substitute, x_in), shape=(-1, classes, width, height, depth))
    derivatives = tf.reduce_sum(derivatives * target_class, axis=1)

    x_out = tf.maximum(0., tf.minimum(1., x_in + lamb * tf.sign(derivatives)))

    loss = CrossEntropy(model, smoothing=label_smoothing)
    eval_params = {"batch_size": batch_size}

    x_sub, x_test, y_test = x_test[:initial_sub_size], x_test[initial_sub_size:], y_test[initial_sub_size:]

    train_params = {
        "nb_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    for i in range(epochs_jbda):
        sub_size = x_sub.shape[0]

        y_sub = np.concatenate([
            session.run(y_out_oracle, feed_dict={x_in: x_sub[batch]})
            for batch in get_batch_indices(0, sub_size, batch_size)
        ])

        def train_evaluation():
            evaluate(session, x_in, y_in, logits, x_train, y_train, x_test, y_test, eval_params)

        train(session, loss, x_sub, y_sub, evaluate=train_evaluation, args=train_params, var_list=model.get_params())

        if i != epochs_jbda - 1:
            extra_samples = [x_sub]

            for batch in get_batch_indices(0, sub_size, batch_size_jbda):
                extra_samples.append(session.run(x_out, feed_dict={y_in: y_sub[batch], x_in: x_sub[batch]}))

            x_sub = np.concatenate(extra_samples)
            np.random.shuffle(x_sub)

    with session.as_default():
        joblib.dump(model, f"models/joblibs/{file_name}.joblib")
