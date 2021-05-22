from copy import deepcopy
from multiprocessing import Process, Queue
from itertools import product
import sys, os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath("."))


def kwargs_to_cmd(kwargs):
    cmd = "/ext3/miniconda3/bin/python3 basis.py "
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
        os.system(kwargs_to_cmd(experiment))


        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-sets', default=[0], type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--seeds', default=[0], type=lambda x: [int(a) for a in x.split(',')])
    parser.add_argument('--sparsities', type=lambda x: [int(a) for a in x.split(',')], default=[25,30,35,40])
    parser.add_argument('--data', default='/', type=str)
    parser.add_argument('--seed_model_dir', default='/scratch/db4045/seed_models_{num_masks}/id\=supsup~seed\={seed}~sparsity\={sparsity}~try\=0/', type=str)
    parser.add_argument('--num-masks', default=50, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--al', type=float, default=0.005)
    parser.add_argument('--logdir-prefix', type=str, required=True)
    args = parser.parse_args()

    gpus = args.gpu_sets
    seeds = args.seeds
    data = args.data

    config = "experiments/basis/splitimagenet/configs/rn50-supsup-basis-adam.yaml"
    log_dir = "/scratch/{user}/runs/{logdir_prefix}/SupsupBasis/rn50-supsup-splitimagenet_num_masks_{num_masks}".format(num_masks=str(args.num_masks), user=os.environ.get("USER"), logdir_prefix=args.logdir_prefix)
    experiments = []
    sparsities = args.sparsities

    for sparsity, seed in product(sparsities, seeds):
        kwargs = {
            "config": config,
            "name": f"id=rn50-basis-supsup-imagenet~seed={seed}~sparsity={sparsity}",
            "sparsity": sparsity,
#            "task-eval": task_idx,
            "log-dir": log_dir,
            "epochs": int(args.epochs),
            "data": data,
            "seed": seed,
            "num-seed-tasks-learned": int(args.num_masks),
            "seed-model": "{}/final.pt".format(args.seed_model_dir.format(sparsity=str(sparsity), seed=str(seed))),
            "al": args.al,
            "wd": 0.01,
            "trainer": "alphareg"
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
