#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=13:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=supsup-seed
#SBATCH --mail-type=END
#SBATCH --mail-user=$USER@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_supsup_seed_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load cuda/10.2.89

export MYPY_ENV='supsup'
MYPY_ENV="${MYPY_ENV:-myenv}"
OVERLAY=$MYPY_ROOT/containers/$MYPY_ENV.ext3
#SPARSITIES_UND=$(echo $SPARSITIES | sed 's/,/_/')

SPARSITIES='25'
#SPARSITIES=$1

# Move into the directory that contains our code
SRCDIR=$(pwd)
echo "Using sparsities: $SPARSITIES"

#/home/db4045/.mypy/bin/python $SRCDIR/experiments/seeds/splitcifar100/rn18-supsup.py --data="/scratch/db4045/data" --seeds 1,2 --logdir-prefix="dhrupad_runs" --gpu-sets="0" --sparsities="$SPARSITIES"
/home/db4045/.mypy/bin/python $SRCDIR/experiments/seeds/splitcifar100/rn18-supsup.py --data="/scratch/db4045/data" --seeds "0,1,2" --logdir-prefix="dhrupad_runs" --gpu-sets="0" --sparsities="$SPARSITIES"
