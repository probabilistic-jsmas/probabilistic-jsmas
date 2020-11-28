import pandas as pd

import argparse
import os


def stats_substitute(sample_path):
    total_samples = 0

    total_samples_success_substitute = 0
    total_samples_success_oracle = 0

    for file in os.listdir(sample_path):
        df = pd.read_csv(sample_path + file)

        cls = int(df.columns[-1].split("_")[1])
        cls_oracle = int(df.columns[-1].split("_")[2])

        if cls_oracle != cls:
            continue

        for k in range(len(df.columns) - 1):
            total_samples += 1

            adv_cls_oracle = int(df.columns[k].split("_")[2])
            adv_cls_substitute = int(df.columns[k].split("_")[3])

            if cls != adv_cls_oracle:
                total_samples_success_oracle += 1

            if cls != adv_cls_substitute:
                total_samples_success_substitute += 1

    print(sample_path)
    print("Total samples:", total_samples)
    print(f"Success rate: {100 * total_samples_success_substitute / total_samples:.2f}")
    print(f"Transferability rate: {100 * total_samples_success_oracle / total_samples:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default="jsma_mnist_mnist_test_1.0")
    args = parser.parse_args()

    stats_substitute("results/" + args.attack + "/")