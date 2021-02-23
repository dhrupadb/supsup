#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=dsga1006-supsup-basis
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:k80:1
#SBATCH --output=logs/slurm_supsup_basis_gpu_3_64_1e-1_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load cuda/10.1.105
module load anaconda3/5.3.1

# Move into the directory that contains our code
SRCDIR=$(pwd)

/scratch/db4045/capstone_env/bin/python $SRCDIR/experiments/basis/splitcifar100/rn18-supsup-basis-highsparsity.py --data="/scratch/db4045/data" --seeds=1 --num-masks=3 --seed_model_dir="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_{num_masks}/id=supsup~seed={seed}~sparsity={sparsity}~try=0/" --logdir-prefix="dhrupad_runs_gpu_opt_hyperparams/lr_1e-1/" --epochs 250 --gpu-sets="0" --lr 1e-1
