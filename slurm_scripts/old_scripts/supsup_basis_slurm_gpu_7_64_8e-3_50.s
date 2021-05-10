#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=60:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=dsga1006-supsup-basis
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:k80:1
#SBATCH --output=logs/slurm_supsup_basis_gpu_7_hyperparams_64_8e-3_50_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load cuda/10.1.105
module load anaconda3/5.3.1

# Move into the directory that contains our code
SRCDIR=$(pwd)

/scratch/db4045/capstone_env/bin/python $SRCDIR/experiments/basis/splitcifar100/rn18-supsup-basis.py --data="/scratch/db4045/data" --seeds=1 --num-masks=7 --seed_model_dir="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_{num_masks}/id=supsup~seed={seed}~sparsity={sparsity}~try=0/" --logdir-prefix="dhrupad_runs_gpu_hyperparams_64_8e-3_50" --epochs 50 --gpu-sets="0" --lr 8e-3 --batch-size 64
