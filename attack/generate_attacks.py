"""
Generates the adversarial samples against the specified model and saves each attack in a single CSV file
"""

import tensorflow as tf

import pandas as pd

import numpy as np

from attack.saliency_map_attack import SaliencyMapMethod

from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_argmax
from cleverhans.serial import load

import os


def generate_attacks(save_path, file_path, x_set, y_set, attack, gamma, first_index, last_index, batch_size=1):
    """
    Applies the voting saliency map attack against the specified model in targeted mode.

    Parameters
    ----------
    save_path: str
        The path of the folder in which the crafted adversarial samples will be saved.
    file_path: str
        The path to the joblib file of the model to attack.
    x_set: numpy.ndarray
        The dataset input array.
    y_set: numpy.ndarray
        The dataset output array.
    attack: str
        The type of used attack (either "jsma", "wjsma" or "tjsma").
    gamma: float
            Maximum percentage of perturbed features.
    first_index:
        The index of the first image attacked.
    last_index: int
        The index of the last image attacked.
    batch_size: int
        The size of the image batches.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sess = tf.Session()

    img_rows, img_cols, channels = x_set.shape[1:4]
    nb_classes = y_set.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))

    with sess.as_default():
        model = load(file_path)

    assert len(model.get_params()) > 0

    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1, 'gamma': gamma, 'clip_min': 0., 'clip_max': 1., 'y_target': None, 'attack': attack}

    preds = model(x)

    y_set = np.argmax(y_set, axis=1).astype(int)

    indices = range(first_index, last_index)
    batch_indices = [indices[x * batch_size:batch_size * (x + 1)] for x in
                     range(len(indices) // batch_size + (len(indices) % batch_size != 0))]

    sample_count = last_index - first_index
    sample_crafted = 0

    for batch in batch_indices:
        samples = []
        sample_classes = []

        current_class_batch = []
        target_classes_batch = []

        for sample_index in batch:
            sample = x_set[sample_index]
            current_class = y_set[sample_index]
            target_classes = other_classes(nb_classes, current_class)

            current_class_batch.append(current_class)
            target_classes_batch += target_classes

            samples.append(np.repeat(sample.reshape((1,) + sample.shape), 9, axis=0))

            y_target = np.zeros((len(target_classes), nb_classes))
            y_target[np.arange(len(target_classes)), target_classes] = 1

            sample_classes.append(y_target)

        samples = np.concatenate(samples)
        sample_classes = np.concatenate(sample_classes)

        jsma_params['y_target'] = sample_classes
        adversarial_batch = jsma.generate_np(samples, **jsma_params)

        for index, sample_index in zip(range(len(batch)), batch):
            results = pd.DataFrame()

            adversarial_samples = adversarial_batch[index * (nb_classes - 1):(index + 1) * (nb_classes - 1)]
            current_class = current_class_batch[index]
            target_classes = target_classes_batch[index * (nb_classes - 1):(index + 1) * (nb_classes - 1)]

            for target, adv_sample in zip(target_classes, adversarial_samples):
                adv_sample = adv_sample.reshape((1, img_rows, img_cols, channels))

                res = int(model_argmax(sess, x, preds, adv_sample) == target)

                adv_x_reshape = adv_sample.reshape(-1)
                test_in_reshape = x_set[sample_index].reshape(-1)
                nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
                percent_perturb = float(nb_changed) / adv_x_reshape.shape[0]

                results['number_' + str(sample_index) + '_' + str(current_class) + '_to_' + str(target)] = \
                    np.concatenate(
                        [adv_x_reshape.reshape(-1), np.array([nb_changed, percent_perturb, res])]
                    )

            sample = samples[index * (nb_classes - 1)]

            results['original_image_' + str(sample_index)] = np.concatenate([sample.reshape(-1), np.zeros((3,))])

            results.to_csv(save_path + '/' + attack + '_image_' + str(sample_index) + '.csv', index=False)

        sample_crafted += len(batch)

        print("Done: ", sample_crafted, "/", sample_count)
