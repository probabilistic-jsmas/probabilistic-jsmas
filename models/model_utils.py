"""
Useful function to test and train models adapted from the cleverhans library
"""
from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy
from cleverhans.serial import save, load

import tensorflow as tf


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001


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

    accuracy = model_eval(session, x, y, predictions, x_set, y_set, args=params)

    print("%s accuracy: %0.4f" % (accuracy_type, accuracy))


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


def model_training(model, file_name, x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, num_threads=None, label_smoothing=0.1):
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
    nb_epochs: int, optional
        The number of epochs.
    batch_size: int, optional
        The batch size.
    learning_rate: float, optional
        The learning rate.
    num_threads: int, optional
        The number of threads used.
    label_smoothing: float, optional
        The amount of label smooting used.
    """

    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    
    session = tf.Session(config=tf.ConfigProto(**config_args))

    img_rows, img_cols, channels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    train_params = {
        "nb_epochs": nb_epochs,
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
        save("models/joblibs/" + file_name, model)


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
        model = load("models/joblibs/" + file_name)

    img_rows, img_cols, channels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    eval_params = {"batch_size": 128}

    predictions = model.get_logits(x)

    evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, eval_params)


def get_labels(file_name, x_set):
    """
    Returns the predicted labels of the input array.

    Parameters
    ----------
    file_name: str
        The name of the joblib.
    x_set: numpy.ndarray
        The input array.
    """

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

    with tf.Session() as sess:
        model = load("models/joblibs/" + file_name + ".joblib")
        last = model(x)

        z = sess.run(last, feed_dict={x: x_set})

        sess.close()

    return z
