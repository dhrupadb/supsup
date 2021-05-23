#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=supsup-seed-imagenet-rn50-task0
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --account=cds
#SBATCH --output=logs/slurm_supsup_seed_imagenet_rn50_task0_%j.out

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
SEEDS=$2
SEED=$3


/home/db4045/.mypy/bin/python $SRCDIR/experiments/seeds/splitimagenet/rn50-supsup-task-0.py --data="/" --seeds $SEEDS --root_seed $SEED --logdir-prefix="runs_final" --gpu-sets="0" --sparsities="$SPARSITIES" --seed_model_dir="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-splitimagenet/id=rn50-supsup-imagenet~seed="$SEED"~sparsity={sparsity}~try=0/"
