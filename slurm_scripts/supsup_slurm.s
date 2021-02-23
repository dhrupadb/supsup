#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=dsga1006-supsup
#SBATCH --mail-type=END
#SBATCH --mail-user=at2507@nyu.edu
#SBATCH --gres=gpu:2
#SBATCH --output=logs/slurm_supsup_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge

# Move into the directory that contains our code
SRCDIR=$HOME/supsup
cd $SRCDIR

#SPARSITIES='25,30,35'
SPARSITIES=$1

# Execute the script
# python ./lab0-test.py
# python ./experiments/GG/splitcifar100/rn18-supsup.py --gpu-sets="0|1|2|3" --data="./data" --seeds 1

/home/db4045/.mypy/bin/python $SRCDIR/experiments/SupsupSeed/splitcifar100/rn18-supsup.py --data="/scratch/db4045/data" --seeds "1,2" --logdir-prefix="dhrupad_runs" --gpu-sets="0" --sparsities="$SPARSITIES"
#/home/db4045/.mypy/bin/python $SRCDIR/experiments/SupsupSeed/splitcifar100/rn18-supsup.py --data="/scratch/db4045/data" --seeds 0,1,2 --logdir-prefix="dhrupad_runs" --gpu-sets="0" --sparsities="$SPARSITIES"
