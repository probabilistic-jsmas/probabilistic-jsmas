import pandas as pd
import numpy as np

import argparse
import os


DEFAULT_THRESHOLD = 114


def stats(sample_path, success_threshold=DEFAULT_THRESHOLD):
    average_l0 = 0
    average_l1 = 0
    average_l2 = 0
    average_li = 0

    average_l0_success = 0
    average_l1_success = 0
    average_l2_success = 0
    average_li_success = 0

    total_samples = 0
    total_samples_success = 0

    for file in os.listdir(sample_path):
        df = pd.read_csv(sample_path + file)
        df_values = df.to_numpy().astype(float)

        org_sample = df_values[:, -1]

        for k in range(df_values.shape[1] - 1):
            adv_sample = df_values[:, k]

            l0 = norm_l0(adv_sample, org_sample)
            l1 = norm_l1(adv_sample, org_sample)
            l2 = norm_l2(adv_sample, org_sample)
            li = norm_li(adv_sample, org_sample)

            total_samples += 1

            average_l0 += l0
            average_l1 += l1
            average_l2 += l2
            average_li += li

            if l0 < success_threshold:
                average_l0_success += l0
                average_l1_success += l1
                average_l2_success += l2
                average_li_success += li

                total_samples_success += 1

    print(sample_path)
    print(f"Total samples: {total_samples}")
    print(f"Success rate: {100 * total_samples_success / total_samples:.2f}")
    print(f"L0={average_l0_success / total_samples_success:.2f} ({average_l0 / total_samples:.2f})")
    print(f"L1={average_l1_success / total_samples_success:.2f} ({average_l1 / total_samples:.2f})")
    print(f"L2={average_l2_success / total_samples_success:.2f} ({average_l2 / total_samples:.2f})")
    print(f"Li={average_li_success / total_samples_success:.2f} ({average_li / total_samples:.2f})")


def norm_l0(a: np.ndarray, b: np.ndarray):
    return np.sum(np.logical_not(np.isclose(a, b)))


def norm_l1(a: np.ndarray, b: np.ndarray):
    return np.sum(np.abs(a - b))


def norm_l2(a: np.ndarray, b: np.ndarray):
    return np.sqrt(np.sum(np.square(a - b)))


def norm_li(a: np.ndarray, b: np.ndarray):
    return np.max(np.abs(a - b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="jsma_mnist_mnist_test_1.0")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    stats("results/" + args.attack + "/", args.threshold)
