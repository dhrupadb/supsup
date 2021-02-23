from copy import deepcopy
from multiprocessing import Process, Queue
from itertools import product
import sys, os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath("."))

# note: new algorithm code
def kwargs_to_cmd(kwargs):
    cmd = "/scratch/db4045/capstone_env/bin/python basis.py "
    for flag, val in kwargs.items():
        cmd += f"--{flag}={val} "
    cmd +="--train_mask_alphas "
    cmd +="--start_at_optimal "

    return cmd


def run_exp(gpu_num, in_queue):
    while not in_queue.empty():
        try:
            experiment = in_queue.get(timeout=3)
        except:
            return

        before = time.time()

        experiment["multigpu"] = gpu_num
        print(f"==> Starting experiment {kwargs_to_cmd(experiment)}")
        os.system(kwargs_to_cmd(experiment))

        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-sets', default=0, type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=1, type=int)
    parser.add_argument('--data', default='/scratch/db4045/data', type=str)
    parser.add_argument('--seed_model_dir', default='/scratch/db4045/seed_models_{num_masks}/id\=supsup~seed\={seed}~sparsity\={sparsity}~try\=0/', type=str)
    parser.add_argument('--num-masks', default=20, type=int)
    parser.add_argument('--logdir-prefix', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    gpus = args.gpu_sets
    seeds = list(range(args.seeds))
    data = args.data

    config = "experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml"
    log_dir = "{scratch}/runs/{logdir_prefix}/SupsupSeedBasis/rn18-supsup_basis_num_masks_{num_masks}".format(num_masks=str(args.num_masks), scratch=os.environ.get("SCRATCH"), logdir_prefix=args.logdir_prefix)
    experiments = []
    sparsities = [1, 2, 4, 8, 16, 32] # Higher sparsity values mean more dense subnetworks

    # at change for 1 epoch to check dir
    for sparsity, seed in product(sparsities, seeds):
        kwargs = {
            "config": config,
            "name": f"id=basis-supsup~seed={seed}~sparsity={sparsity}",
            "log-dir": log_dir,
            "epochs": args.epochs,
            "data": data,
            "seed-model": "{}/final.pt".format(args.seed_model_dir.format(sparsity=str(sparsity), seed=str(seed), num_masks=str(args.num_masks)))
        }

        experiments.append(kwargs)

    print(experiments)
    queue = Queue()

    for e in experiments:
        queue.put(e)

    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(gpu, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
