from attack.generate_attacks import generate_attacks


MNIST_SETS = ["mnist", "lenet-5-less-training", "mnist_defense_jsma", "mnist_defense_wjsma", "mnist_defense_tjsma"]
CIFAR10_SETS = ["cifar10", "cifar10_defense_jsma", "cifar10_defense_wjsma", "cifar10_defense_tjsma"]


def save_images(model, attack, set_type, first_index, last_index, batch_size):
    """
    Applies the saliency map attack against the specified model.

    Parameters
    ----------
    model: str
        The name of the model used.
    attack: str
        The type of used attack (either "jsma", "wjsma" or "tjsma").
    set_type: str
        The type of set used (either "train" or "test").
    first_index:
        The index of the first image attacked.
    last_index: int
        The index of the last image attacked.
    batch_size: int
        The size of the image batches.
    """

    if model in MNIST_SETS:
        from cleverhans.dataset import MNIST

        x_set, y_set = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000).get_set(set_type)
        gamma = 0.155
    elif model in CIFAR10_SETS:
        from cleverhans.dataset import CIFAR10

        x_set, y_set = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000).get_set(set_type)
        y_set = y_set.reshape((y_set.shape[0], 10))
        gamma = 0.039
    else:
        raise ValueError("Invalid model: " + model)

    generate_attacks(
        save_path="attack/" + model + "/" + attack + "_" + set_type,
        file_path="models/joblibs/" + model + ".joblib",
        x_set=x_set,
        y_set=y_set,
        attack=attack,
        gamma=gamma,
        first_index=first_index,
        last_index=last_index,
        batch_size=batch_size
    )
