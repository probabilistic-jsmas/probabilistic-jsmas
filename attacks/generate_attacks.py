from joblib import load

import tensorflow as tf
import pandas as pd
import numpy as np

import os


FIRST_INDEX = 0
LAST_INDEX = 1000
BATCH_SIZE = 1
THETA = 1.
CLIP_MIN = 0.
CLIP_MAX = 1.
MAX_ITER = 57


def get_batch_indices(first_index, last_index, batch_size):
    """
    Returns the batch indices.

    Parameters
    ----------
    first_index: int
        The first index of the batches.
    last_index: int
        The last index of the batches.
    batch_size: int
        The size of the batches.

    Returns
    -------
    batch_indices: list
        The list of indices of each batches.
    """

    indices = range(first_index, last_index)

    batch_indices = [
        indices[x * batch_size:batch_size * (x + 1)] for x in
        range(len(indices) // batch_size + (len(indices) % batch_size != 0))
    ]

    return batch_indices


def keep_well_predicted(model, session, x_set, y_set, batch_size=128):
    """
    Sanitizes the set by keeping only the well-predicted samples.

    Parameters
    ----------
    model: cleverhans.model.Model
        The cleverhans model.
    session: tensorflow.Session
        The tensorflow session.
    x_set: numpy.ndarray
        The input array of the dataset.
    y_set: numpy.ndarray
        The output array of the dataset.
    batch_size: int
        The size of the batches used to predict.

    Returns
    -------
    x_set: numpy.ndarray
        The input array of the reduced dataset.
    y_set: numpy.ndarray
        The output array  of the reduced dataset.
    """

    x_in = tf.placeholder(dtype=tf.float32, shape=(None,) + x_set.shape[1:])
    y_out = tf.argmax(model(x_in), axis=1)

    predicted_classes = np.concatenate([
        session.run(y_out, feed_dict={x_in: x_set[batch]}) for batch in get_batch_indices(0, x_set.shape[0], batch_size)
    ])

    mask = np.argmax(y_set, axis=1) == predicted_classes

    return x_set[mask], y_set[mask]


def generate_attacks_targeted(attack, model_path, save_path, x_set, y_set, first_index=FIRST_INDEX,
                              last_index=LAST_INDEX, batch_size=BATCH_SIZE, theta=THETA, clip_min=CLIP_MIN,
                              clip_max=CLIP_MAX, max_iter=MAX_ITER, cast=tf.float32):
    """
    Generates and saves adversarial samples for each possible targeted against the specified model.

    Parameters
    ----------
    attack: function
        The attack used. The function returns the symbolic tf.Tensor of the results.
    model_path: str
        The model name.
    save_path: str
        The folder in which the adversarial samples will be saved.
    x_set: numpy.ndarray
        The input array of the dataset.
    y_set: numpy.ndarray
        The output array of the dataset.
    first_index: int
        The index of the first attacked sample.
    last_index: int
        The index of the last attacked samples.
    batch_size: int
        The size of the adversarial batches.
    theta: float
        The amount by which the pixel are modified (can either be positive or negative).
    clip_min: float
        Minimum component value for clipping.
    clip_max: float
        Maximum component value for clipping.
    max_iter: int
        Maximum iteration before the attack stops.
    cast: tf.dtype
        The tensor data type used.
    """

    if not os.path.exists(f"results/{save_path}"):
        os.mkdir(f"results/{save_path}")

    sess = tf.Session()

    x_in = tf.placeholder(dtype=tf.float32, shape=(None,) + x_set.shape[1:])
    y_in = tf.placeholder(dtype=tf.float32, shape=(None,) + y_set.shape[1:])

    with sess.as_default():
        model = load(f"models/joblibs/{model_path}.joblib")

    x_set, y_set = keep_well_predicted(model, sess, x_set, y_set)

    if last_index > x_set.shape[0]:
        last_index = x_set.shape[0]

    x_adv = attack(model, x_in, y_in, theta, clip_min, clip_max, max_iter, cast)

    nb_classes = y_set.shape[1]

    y_set = np.argmax(y_set, axis=1).astype(int)

    indices = range(first_index, last_index)

    x_set_targeted = []
    y_set_targeted = []

    current_class_batch = []
    target_classes_batch = []

    for index in indices:
        sample = x_set[index]
        current_class = y_set[index]

        target_classes = list(range(nb_classes))
        target_classes.remove(current_class)

        current_class_batch.append(current_class)
        target_classes_batch += target_classes

        x_set_targeted.append(np.repeat(sample.reshape((1,) + sample.shape), nb_classes - 1, axis=0))

        y_target = np.zeros((len(target_classes), nb_classes))
        y_target[np.arange(len(target_classes)), target_classes] = 1

        y_set_targeted.append(y_target)

    x_set_targeted = np.concatenate(x_set_targeted)
    y_set_targeted = np.concatenate(y_set_targeted)

    x_set_crafted = []

    file_index = first_index
    batch_index = 0
    sample_count = x_set_targeted.shape[0]

    for batch in get_batch_indices(first_index * (nb_classes - 1), last_index * (nb_classes - 1), batch_size):
        samples = x_set_targeted[batch]
        sample_classes = y_set_targeted[batch]

        adversarial_batch = sess.run(x_adv, feed_dict={x_in: samples, y_in: sample_classes})

        x_set_crafted += list(adversarial_batch)

        while len(x_set_crafted) >= nb_classes - 1:
            results = pd.DataFrame()

            adversarial_samples = x_set_crafted[:nb_classes - 1]
            current_class = current_class_batch[file_index - first_index]

            del x_set_crafted[:nb_classes - 1]

            for adversarial_sample in adversarial_samples:
                target_class = target_classes_batch[batch_index]

                results[f"{file_index}_{current_class}_{target_class}"] = adversarial_sample.reshape(-1)

                batch_index += 1

            sample = x_set[file_index]

            results[f"{file_index}_o"] = sample.reshape(-1)

            results.to_csv(f"results/{save_path}/image_{file_index}.csv", index=False)

            file_index += 1

        print(f"Done: {batch_index + len(x_set_crafted)} / {sample_count}")


def generate_attacks_non_targeted(attack, model_path, save_path, x_set, y_set, first_index=FIRST_INDEX,
                                  last_index=LAST_INDEX, batch_size=BATCH_SIZE, theta=THETA, clip_min=CLIP_MIN,
                                  clip_max=CLIP_MAX, max_iter=MAX_ITER, cast=tf.float32, use_logits=True,
                                  non_stop=False):
    """
    Generates and saves adversarial non-targeted samples against the specified model.

    Parameters
    ----------
    attack: function
        The attack used. The function returns the symbolic tf.Tensor of the results.
    model_path: str
        The model name.
    save_path: str
        The folder in which the adversarial samples will be saved.
    x_set: numpy.ndarray
        The input array of the dataset.
    y_set: numpy.ndarray
        The output array of the dataset.
    first_index: int
        The index of the first attacked sample.
    last_index: int
        The index of the last attacked samples.
    batch_size: int
        The size of the adversarial batches.
    theta: float
        The amount by which the pixel are modified (can either be positive or negative).
        Set this to None for maximal attacks.
    clip_min: float
        Minimum component value for clipping.
    clip_max: float
        Maximum component value for clipping.
    max_iter: int
        Maximum iteration before the attack stops.
    cast: tf.dtype
        The tensor data type used.
    use_logits: bool
        Uses the logits (Z variation) when set to True and the softmax (F variation) values otherwise.
    non_stop: bool
        When set to True, the attacks continue until max_iter is reached (used to attack the substitute for black box).
    """

    if not os.path.exists(f"results/{save_path}"):
        os.mkdir(f"results/{save_path}")

    sess = tf.Session()

    x_in = tf.placeholder(dtype=tf.float32, shape=(None,) + x_set.shape[1:])

    with sess.as_default():
        model = load(f"models/joblibs/{model_path}.joblib")

    x_set, y_set = keep_well_predicted(model, sess, x_set, y_set)

    if last_index > x_set.shape[0]:
        last_index = x_set.shape[0]

    if theta is None:
        x_adv = attack(model, x_in, clip_min, clip_max, max_iter, cast, use_logits, non_stop)
    else:
        x_adv = attack(model, x_in, theta, clip_min, clip_max, max_iter, cast, use_logits, non_stop)

    sample_count = last_index - first_index
    sample_crafted = 0

    y_set = np.argmax(y_set, axis=1)

    for batch in get_batch_indices(first_index, last_index, batch_size):
        samples = x_set[batch]
        classes = y_set[batch]

        adversarial_batch = sess.run(x_adv, feed_dict={x_in: samples})

        for index, file_index in zip(range(len(batch)), batch):
            results = pd.DataFrame()

            results[f"{file_index}_{classes[index]}_u"] = adversarial_batch[index].reshape(-1)
            results[f"{file_index}_o"] = samples[index].reshape(-1)

            results.to_csv(f"results/{save_path}/image_{file_index}.csv", index=False)

        sample_crafted += len(batch)

        print(f"Done: {sample_crafted} / {sample_count}")


def generate_attacks_non_targeted_substitute(attack, model_path, oracle_path, save_path, x_set, y_set,
                                             first_index=FIRST_INDEX, last_index=LAST_INDEX, batch_size=BATCH_SIZE,
                                             theta=THETA, clip_min=CLIP_MIN, clip_max=CLIP_MAX, max_iter=MAX_ITER,
                                             cast=tf.float32, use_logits=False, non_stop=True):
    """
    Generates and saves adversarial non-targeted samples against a substitute model.

    Parameters
    ----------
    attack: function
        The attack used. The function returns the symbolic tf.Tensor of the results.
    model_path: str
        The model name.
    oracle_path: str
        The oracle model name.
    save_path: str
        The folder in which the adversarial samples will be saved.
    x_set: numpy.ndarray
        The input array of the dataset.
    y_set: numpy.ndarray
        The output array of the dataset.
    first_index: int
        The index of the first attacked sample.
    last_index: int
        The index of the last attacked samples.
    batch_size: int
        The size of the adversarial batches.
    theta: float
        The amount by which the pixel are modified (can either be positive or negative).
        Set this to None for maximal attacks.
    clip_min: float
        Minimum component value for clipping.
    clip_max: float
        Maximum component value for clipping.
    max_iter: int
        Maximum iteration before the attack stops.
    cast: tf.dtype
        The tensor data type used.
    use_logits: bool
        Uses the logits (Z variation) when set to True and the softmax (F variation) values otherwise.
    non_stop: bool
        When set to True, the attacks continue until max_iter is reached (used to attack the substitute for black box).
    """

    if not os.path.exists(f"results/{save_path}"):
        os.mkdir(f"results/{save_path}")

    sess = tf.Session()

    x_in = tf.placeholder(dtype=tf.float32, shape=(None,) + x_set.shape[1:])

    with sess.as_default():
        model = load(f"models/joblibs/{model_path}.joblib")
        oracle = load(f"models/joblibs/{oracle_path}.joblib")

    x_set, y_set = keep_well_predicted(oracle, sess, x_set, y_set)

    if last_index > x_set.shape[0]:
        last_index = x_set.shape[0]

    if theta is None:
        x_adv = attack(model, x_in, clip_min, clip_max, max_iter, cast, use_logits, non_stop)
    else:
        x_adv = attack(model, x_in, theta, clip_min, clip_max, max_iter, cast, use_logits, non_stop)

    model_pred = tf.argmax(model(x_in), axis=1)
    oracle_pred = tf.argmax(oracle(x_in), axis=1)

    sample_count = last_index - first_index
    sample_crafted = 0

    y_set = np.argmax(y_set, axis=1)

    for batch in get_batch_indices(first_index, last_index, batch_size):
        samples = x_set[batch]
        classes = y_set[batch]

        adversarial_batch = sess.run(x_adv, feed_dict={x_in: samples})

        sub_classes, oracle_classes = sess.run([model_pred, oracle_pred], feed_dict={x_in: samples})
        adv_sub_classes, adv_oracle_classes = sess.run([model_pred, oracle_pred], feed_dict={x_in: adversarial_batch})

        for index, file_index in zip(range(len(batch)), batch):
            results = pd.DataFrame()

            cls = classes[index]
            sub_class, oracle_class = sub_classes[index], oracle_classes[index]
            adv_sub_class, adv_oracle_class = adv_sub_classes[index], adv_oracle_classes[index]

            results[f"{file_index}_{cls}_{adv_oracle_class}_{adv_sub_class}_u"] = adversarial_batch[index].reshape(-1)
            results[f"{file_index}_{cls}_{oracle_class}_{sub_class}_o"] = samples[index].reshape(-1)

            results.to_csv(f"results/{save_path}/image_{file_index}.csv", index=False)

        sample_crafted += len(batch)

        print(f"Done: {sample_crafted} / {sample_count}")
