from datasets import load_dataset

import argparse


DEFAULT_MODEL = "mnist"
DEFAULT_DATASET = "mnist"
DEFAULT_EPOCHS = 6
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = .001
DEFAULT_LABEL_SMOOTHING = .1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batchsize", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batchsize
    learning_rate = args.lr
    label_smoothing = args.smoothing

    x_train, y_train, x_test, y_test = load_dataset(dataset)

    width, height, depth = x_train.shape[1:]
    nb_classes = y_train.shape[1]

    if nb_classes == 10 and width == 28 and height == 28 and depth == 1:
        from models.train_models import train_mnist

        train_mnist(
            model_path=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, label_smoothing=label_smoothing
        )

    elif nb_classes == 10 and width == 32 and height == 32 and depth == 3:
        from models.train_models import train_cifar10

        train_cifar10(
            model_path=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, label_smoothing=label_smoothing
        )

    elif nb_classes == 43 and width == 32 and height == 32 and depth == 3:
        from models.train_models import train_gtsrb

        train_gtsrb(
            model_path=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate, label_smoothing=label_smoothing
        )

    else:
        raise ValueError(f"Unknown dataset shape: {width}x{height}x{depth} {nb_classes} classes.")