from datasets import load_dataset

import argparse
import os


TARGETED_ATTACKS = ["jsma", "wjsma", "tjsma"]
NON_TARGETED_ATTACKS = ["jsma_nt", "wjsma_nt", "tjsma_nt"]
MAXIMAL_ATTACKS = ["mjsma", "mwjsma"]

DEFAULT_ATTACK = "jsma"
DEFAULT_MODEL = "mnist"
DEFAULT_DATASET = "mnist"
DEFAULT_SET_TYPE = "test"
DEFAULT_FIRST_INDEX = 0
DEFAULT_LAST_INDEX = 1000
DEFAULT_BATCH_SIZE = 1
DEFAULT_THETA = 1.
DEFAULT_CLIP_MIN = 0.
DEFAULT_CLIP_MAX = 1.
DEFAULT_MAX_ITER = 57
DEFAULT_USE_LOGITS = "true"
DEFAULT_NON_STOP = "false"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, default=DEFAULT_ATTACK)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--settype", type=str, default=DEFAULT_SET_TYPE)
    parser.add_argument("--firstindex", type=int, default=DEFAULT_FIRST_INDEX)
    parser.add_argument("--lastindex", type=int, default=DEFAULT_LAST_INDEX)
    parser.add_argument("--batchsize", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    parser.add_argument("--clipmin", type=float, default=DEFAULT_CLIP_MIN)
    parser.add_argument("--clipmax", type=float, default=DEFAULT_CLIP_MAX)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--uselogits", type=str, default=DEFAULT_USE_LOGITS)
    parser.add_argument("--nonstop", type=str, default=DEFAULT_NON_STOP)
    args = parser.parse_args()

    attack = args.attack
    model = args.model
    dataset = args.dataset
    set_type = args.settype
    first_index = args.firstindex
    last_index = args.lastindex
    batch_size = args.batchsize
    theta = args.theta
    clip_min = args.clipmin
    clip_max = args.clipmax
    max_iter = args.maxiter
    use_logits = args.uselogits == "true"
    non_stop = args.nonstop == "true"

    if not os.path.exists("results/"):
        os.mkdir("results/")

    if set_type == "test":
        _, _, x, y = load_dataset(dataset)
    else:
        x, y, _, _ = load_dataset(dataset)

    if attack in TARGETED_ATTACKS:
        from attacks import generate_attacks_targeted

        if attack == "jsma":
            from attacks import attack_jsma_targeted as attack_target
        elif attack == "wjsma":
            from attacks import attack_wjsma_targeted as attack_target
        elif attack == "tjsma":
            from attacks import attack_tjsma_targeted as attack_target

        generate_attacks_targeted(
            attack=attack_target, model_path=model, save_path=f"{attack}_{model}_{dataset}_{set_type}_{theta}", x_set=x,
            y_set=y, first_index=first_index, last_index=last_index, batch_size=batch_size, theta=theta,
            clip_min=clip_min,  clip_max=clip_max, max_iter=max_iter
        )

    elif attack in NON_TARGETED_ATTACKS:
        from attacks import generate_attacks_non_targeted

        if attack == "jsma_nt":
            from attacks import attack_jsma_non_targeted as attack_non_target
        elif attack == "wjsma_nt":
            from attacks import attack_wjsma_non_targeted as attack_non_target
        elif attack == "tjsma_nt":
            from attacks import attack_tjsma_non_targeted as attack_non_target

        save_path = f"{attack}"

        if use_logits:
            save_path += "_z"
        else:
            save_path += "_f"

        save_path += f"_{model}_{dataset}_{set_type}_{theta}"

        if non_stop:
            save_path += "_non_stop"

        generate_attacks_non_targeted(
            attack=attack_non_target, model_path=model, save_path=save_path, x_set=x, y_set=y, first_index=first_index,
            last_index=last_index, batch_size=batch_size, theta=theta, clip_min=clip_min, clip_max=clip_max,
            max_iter=max_iter, use_logits=use_logits, non_stop=non_stop
        )

    elif attack in MAXIMAL_ATTACKS:
        from attacks import generate_attacks_non_targeted

        if attack == "mjsma":
            from attacks import attack_maximal_jsma as attack_non_target
        elif attack == "mwjsma":
            from attacks import attack_maximal_wjsma as attack_non_target

        save_path = f"{attack}"

        if use_logits:
            save_path += "_z"
        else:
            save_path += "_f"

        save_path += f"_{model}_{dataset}_{set_type}"

        if non_stop:
            save_path += "_non_stop"

        generate_attacks_non_targeted(
            attack=attack_non_target, model_path=model, save_path=save_path, x_set=x, y_set=y, first_index=first_index, 
            last_index=last_index, batch_size=batch_size, theta=None, clip_min=clip_min, clip_max=clip_max, 
            max_iter=max_iter, use_logits=use_logits, non_stop=non_stop
        )

    else:
        raise ValueError(f"Unknown attack {attack}")
