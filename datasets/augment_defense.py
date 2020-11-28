import numpy as np
import pandas as pd
import random
import os


SAMPLE_COUNT = 2000


def augment_dataset(sample_path, x_train, y_train, sample_per_class=SAMPLE_COUNT, threshold=None):
    """
    Adds adversarial samples to the original train dataset.

    Parameters
    ----------
    sample_path: str
        The name of the folder which contains the samples.
    x_train: numpy.ndarray
        The input array of the original train dataset.
    y_train: numpy.ndarray
        The output array of the original train dataset.
    sample_per_class: int
        The number of samples per class added to the dataset.
    threshold: int
        The L0 threshold below which the attacks are considered as successful.
        Set this to None to keep every samples.

    Returns
    -------
    x_train, y_train: numpy.ndarray
        The augmented dataset.
    """

    width, height, depth = x_train.shape[1:]
    nb_classes = y_train.shape[1]
    nb_features = width * height * depth

    if not threshold:
        threshold = nb_features

    samples = []
    sample_counts = [0 for _ in range(nb_classes)]

    files = os.listdir(f"results/{sample_path}/")
    random.shuffle(files)

    for file in files:
        df = pd.read_csv(f"results/{sample_path}/{file}")
        df_values = df.to_numpy()

        label = int(df.columns[0].split("_")[1])

        if sample_counts[label] == sample_per_class:
            for k in range(nb_classes):
                if sample_counts[k] < sample_per_class:
                    break
            else:
                break

            continue

        label_one_hot = [0] * nb_classes
        label_one_hot[label] = 1

        org_sample = df_values[:, -1]

        for k in range(df_values.shape[1] - 1):
            adv_sample = df_values[:, k]

            l0 = norm_l0(adv_sample, org_sample)

            if l0 < threshold:
                samples.append([adv_sample.reshape((width, height, depth)), label_one_hot])
                sample_counts[label] += 1

                if sample_counts[label] == sample_per_class:
                    break

    for k in range(nb_classes):
        if sample_counts[k] < sample_per_class:
            raise ValueError("Not enough samples to create the augmented dataset")

    random.shuffle(samples)

    x_augment = np.array([samples[k][0] for k in range(len(samples))])
    y_augment = np.array([samples[k][1] for k in range(len(samples))])

    return np.concatenate([x_train, x_augment], axis=0).astype(np.float32), \
        np.concatenate([y_train, y_augment], axis=0).astype(np.float32)


def norm_l0(a: np.ndarray, b: np.ndarray):
    return np.sum(np.logical_not(np.isclose(a, b)))
