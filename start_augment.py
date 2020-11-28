from datasets import load_dataset, augment_dataset, save_dataset

import argparse


DEFAULT_SAMPLES = "jsma_mnist_mnist_1.0"
DEFAULT_DATASET = "mnist"
DEFAULT_NAME = "mnist_augmented"
DEFAULT_SAMPLE_PER_CLASSES = 2000
DEFAULT_THRESHOLD = 114

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default=DEFAULT_SAMPLES)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--name", type=str, default=DEFAULT_NAME)
    parser.add_argument("--spp", type=int, default=DEFAULT_SAMPLE_PER_CLASSES)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    samples = args.samples
    dataset = args.dataset
    name = args.name
    sample_per_classes = args.spp
    threshold = args.threshold

    x_train, y_train, x_test, y_test = load_dataset(dataset)
    x_train, y_train = augment_dataset(samples, x_train, y_train, sample_per_classes, threshold)

    save_dataset(name, x_train, y_train, x_test, y_test)
