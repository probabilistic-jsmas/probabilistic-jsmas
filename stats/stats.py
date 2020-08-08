"""
Computation of the statistics of the generated attacks in folder
"""

import pandas
import numpy as np
import os

from models.model_utils import get_labels


def average_stat(model, set_type, attack, with_max_threshold=True):
    """
    Prints out the stats of the attack.

    Parameters
    ----------
    model: str
        The joblib name.
    set_type: str
        The type of set used (either "train" or "test")
    attack: str
        The type of attack used (either "jsma", "wjsma" or "tjsma")
    with_max_threshold: bool, optional
        Uses the max threshold as the upper limit to compute stats for unsuccessful samples if set to True.
    """

    if "mnist" in model:
        image_size = 784
        max_iter = 57
        max_distortion = 2 * max_iter / image_size
        max_pixel_number = int(image_size * max_distortion / 2) * 2

        from cleverhans.dataset import MNIST

        x_set, y_set = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000).get_set(set_type)
    elif "cifar10" in model:
        image_size = 3072
        max_iter = 57
        max_distortion = 2 * max_iter / image_size
        max_pixel_number = int(image_size * max_distortion / 2) * 2

        from cleverhans.dataset import CIFAR10

        x_set, y_set = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000).get_set(set_type)
        y_set = y_set.reshape((y_set.shape[0], 10))
    else:
        raise ValueError(
            "Invalid folder name, it must have the name of the dataset somewhere either 'mnist' or 'cifar10'"
        )

    y_set = np.argmax(y_set, axis=1)

    average_distortion = 0
    average_distortion_successful = 0
    average_pixel_number = 0
    average_pixel_number_successful = 0

    total_samples = 0
    total_samples_successful = 0

    predicted = np.argmax(get_labels(model, x_set), axis=1)

    folder = "attack/" + model + "/" + attack + "_" + set_type + "/"

    for file in os.listdir(folder):
        df = pandas.read_csv(folder + file)
        df_values = df.to_numpy()

        index = int(file.split("_")[2][:-4])

        if y_set[index] != predicted[index]:
            continue

        for i in range(9):
            total_samples += 1

            if with_max_threshold:
                average_pixel_number += min(df_values[-3, i], max_pixel_number)
                average_distortion += min(df_values[-2, i], max_distortion)
            else:
                average_pixel_number += df_values[-3, i]
                average_distortion += df_values[-2, i]

            if df_values[-3, i] < max_iter * 2:  # 2 modified pixel per iteration
                total_samples_successful += 1

                average_pixel_number_successful += df_values[-3, i]
                average_distortion_successful += df_values[-2, i]

    print(folder)
    print("----------------------")
    print("WELL PREDICTED ORIGINAL SAMPLES:", total_samples / 9)
    print("SUCCESS RATE (MISS CLASSIFIED):", total_samples_successful / total_samples)
    print("AVERAGE NUMBER OF CHANGED PIXELS:", average_pixel_number / total_samples)
    print("AVERAGE DISTORTION:", average_distortion / total_samples)
    print("----------------------")
    print("AVERAGE SUCCESSFUL NUMBER OF CHANGED PIXELS:", average_pixel_number_successful / total_samples_successful)
    print("AVERAGE SUCCESSFUL DISTORTION:", average_distortion_successful / total_samples_successful)
    print("----------------------\n")
