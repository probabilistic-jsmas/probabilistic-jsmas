"""
Create extra dataset to train a defended model
"""
import pandas
import random
import numpy
import os

SAMPLE_COUNT = 2000


def generate_extra_set(set_type, attack, sample_per_class=SAMPLE_COUNT):
    """
    Generate an extra MNIST dataset containing adversarial samples labeled correctly

    Parameters
    ----------
    set_type: str
        The type of set used (either "train" or "test"). The train set should be used since in practice, the defender
        only have the train dataset.
    attack: str
        The type of attack used to augment the dataset (either "jsma", "wjsma" or "tjsma").
    sample_per_class: int, optional
        The number of extra samples per class.
    """

    samples = [[] for _ in range(10)]

    path = "attack/mnist/" + attack + "_" + set_type + "/"

    sample_count = 0

    for file in os.listdir(path):
        df = pandas.read_csv(path + file)
        np = df.to_numpy()

        label = int(df.columns[0][-6])

        for i in range(9):
            if np[785, i] < 0.155:
                samples[label].append(np[:784, i].reshape((28, 28)))
                sample_count += 1

        if sample_count > 50 * SAMPLE_COUNT:
            break

    for c in range(10):
        print(c, len(samples[c]))

    x_set = []

    for k in range(10):
        random.shuffle(samples[k])

        if len(samples[k]) < sample_per_class:
            raise ValueError("Not enough samples to create the augmented dataset")

        samples[k] = samples[k][:sample_per_class]

        for sample in samples[k]:
            x_set.append([sample, _one_hot(k)])

    random.shuffle(x_set)

    x = numpy.array([x_set[k][0] for k in range(len(x_set))])
    y = numpy.array([x_set[k][1] for k in range(len(x_set))])

    if not os.path.exists("defense/augmented/"):
        os.mkdir("defense/augmented/")

    numpy.save("defense/augmented/" + attack + "_x.npy", x)
    numpy.save("defense/augmented/" + attack + "_y.npy", y)


def _one_hot(index):
    vector = numpy.zeros(10)
    vector[index] = 1

    return vector
