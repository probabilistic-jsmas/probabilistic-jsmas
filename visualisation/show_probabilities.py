"""
Analysis of the results and comparison between JSMA and WJSMA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('precision', 3)

FOLDER_PATH = r"attack/mnist/"
IMAGE_NUMBER = 1
TARGET_CLASS = 6
ORIGIN_CLASS = 0


def prob_length(probabilities):
    """
    Computes index of the length of the probability vector without completing zeros
    :param probabilities: array with listof probabilities with zeros
    :return: length of probabilities until the end of the attack,
    """

    zero_matrix = np.zeros((10,))
    max_length = probabilities.shape[0]

    for i in range(0, max_length, 10):
        if np.array_equal(probabilities[i:(i + 10), 0], zero_matrix):
            return i

    return max_length


def probabilities_array(file_path, target_class):
    """
    Gets the interesting probabilities that will be displayed in graph
    :param file_path: attack file path
    :param target_class: target of the adversarial sample
    :return: array to plot
    """

    if 'mnist' in file_path:
        image_size = 784
    elif 'cifar10' in file_path:
        image_size = 3072
    else:
        raise ValueError("file_path doesn't contain the dataset name, either 'mnist or 'cifar10")

    csv = pd.read_csv(file_path)
    max_length = csv.shape[0] - 3 - image_size
    first_class = np.argmax(csv.iloc[image_size:(image_size + 10), 0].values)

    if target_class < first_class:
        index = target_class
    elif target_class > first_class:
        index = target_class - 1
    else:
        raise ValueError("The target has the same value as the predicted class.")

    probabilities = csv.iloc[image_size:(image_size + max_length), index].values.reshape((max_length, 1))
    prob_len = prob_length(probabilities)

    return probabilities[:prob_len, 0]


def visualise(folder_path=FOLDER_PATH, origin_class=ORIGIN_CLASS, target_class=TARGET_CLASS):
    """
    Shows the graph of the target and origin class probabilities in the jsma and wjsma attacks
    :param folder_path: folder of the attacks
    :param origin_class: the class of the image before the attack
    :param target_class: the target class of the attack
    """

    jsma_path = folder_path + "jsma_train/jsma_image_" + str(IMAGE_NUMBER) + ".csv"
    wjsma_path = folder_path + "wjsma_train/wjsma_image_" + str(IMAGE_NUMBER) + ".csv"
    jsma_probs = probabilities_array(jsma_path, 6)
    wjsma_probs = probabilities_array(wjsma_path, 6)

    jsma_target_probs = jsma_probs[target_class::10]
    wjsma_target_probs = wjsma_probs[target_class::10]
    jsma_origin_probs = jsma_probs[origin_class::10]
    wjsma_origin_probs = wjsma_probs[origin_class::10]

    plt.plot(jsma_target_probs, label="JSMA target")
    plt.plot(jsma_origin_probs, label="JSMA origin")
    plt.plot(wjsma_target_probs, label="WJSMA target")
    plt.plot(wjsma_origin_probs, label="WJSMA origin")

    plt.xlabel('Iterations')
    plt.ylabel('Probabilities')

    plt.legend()
    plt.show()
