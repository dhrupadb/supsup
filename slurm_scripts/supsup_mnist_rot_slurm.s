#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=supsup-seed-mnist-rot
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --output=logs/slurm_supsup_mnist_rot_seed_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
export MYPY_ENV='supsup'
MYPY_ENV="${MYPY_ENV:-myenv}"
OVERLAY=$MYPY_ROOT/containers/$MYPY_ENV.ext3

# Move into the directory that contains our code
SRCDIR=$(pwd)

#SPARSITIES='25,30,35'
SPARSITY=$1
SEED=$2

/home/db4045/.mypy/bin/python $SRCDIR/mnist-basis-clean.py --seed $SEED --sparsity $SPARSITY --num_tasks 250 --model_type Supsup --log_dir /scratch/db4045/runs/logs/ --output_dir /scratch/db4045/runs/runs_final/SupsupSeed/ --dataset MNISTRot --epochs 1 --data_root /scratch/db4045/data/mnist
