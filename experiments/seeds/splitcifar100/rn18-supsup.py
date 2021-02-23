from copy import deepcopy
from multiprocessing import Process, Queue from itertools import product
import sys, os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath("."))

# note: new algorithm code
def kwargs_to_cmd(kwargs):
    cmd = "/ext3/miniconda3/bin/python3 main.py "
    for flag, val in kwargs.items():
        cmd += f"--{flag}={val} "

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
        cmd = kwargs_to_cmd(experiment)
        os.system(cmd)

        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-sets', default=[0], type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=[0], type=lambda x: [int(a) for a in x.split(',')])
    parser.add_argument('--sparsities', type=lambda x: [int(a) for a in x.split(',')], default=[25,30,35,40])
    parser.add_argument('--data', default='/scratch/db404/data', type=str)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--logdir-prefix', type=str, required=True)
    args = parser.parse_args()

    gpus = args.gpu_sets
    seeds = args.seeds
    data = args.data

    config = "experiments/seeds/splitcifar100/configs/rn18-supsup.yaml"
    log_dir = "/scratch/{user}/runs/{logdir_prefix}/SupsupSeed/rn18-supsup".format(user=os.environ.get("USER"), logdir_prefix=args.logdir_prefix)
    experiments = []
#    sparsities = [20, 30, 40, 50, 60, 65, 70, 75, 80, 75, 90, 95] # Higher sparsity values mean more sparse subnetworks
#    sparsities = [20, 30, 40, 50, 60, 65] # Higher sparsity values mean more sparse subnetworks
    sparsities = args.sparsities

    # at change for 1 epoch to check dir
    for sparsity, seed in product(sparsities, seeds):
        kwargs = {
            "config": config,
            "name": f"id=supsup~seed={seed}~sparsity={sparsity}",
            "sparsity": sparsity,
            "seed": seed,
            "log-dir": log_dir,
            "epochs": int(args.epochs),
            "data": data
        }

        experiments.append(kwargs)

    print(experiments)
    queue = Queue()

    for e in experiments:
        queue.put(e)

    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(0, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
