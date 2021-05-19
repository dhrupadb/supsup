#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=supsup-seed-imagenet
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --account=cds
#SBATCH --output=logs/slurm_supsup_seed_imagenet_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load cuda/10.2.89

export MYPY_ENV='supsup'
MYPY_ENV="${MYPY_ENV:-myenv}"
OVERLAY=$MYPY_ROOT/containers/$MYPY_ENV.ext3

# Move into the directory that contains our code
SRCDIR=$(pwd)

#SPARSITIES='25,30,35'
SPARSITIES=$1

/home/db4045/.mypy/bin/python $SRCDIR/experiments/seeds/splitimagenet/rn18-supsup.py --data="/" --seeds 0 --logdir-prefix="runs_final" --gpu-sets="0" --sparsities="$SPARSITIES"
