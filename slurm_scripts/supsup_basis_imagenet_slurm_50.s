#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem=45GB
#SBATCH --job-name=supsup-50-basis-imgnet
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --account=cds
#SBATCH --output=logs/slurm_supsup_basis_50_imagenet_%j.out

# Remove all unused system modules
module purge
module load cuda/10.2.89

export MYPY_ENV='supsup'
MYPY_ENV="${MYPY_ENV:-myenv}"
OVERLAY=$MYPY_ROOT/containers/$MYPY_ENV.ext3

# Move into the directory that contains our code
SRCDIR=$(pwd)

SPARSITIES=$1
SEEDS=$2

/home/db4045/.mypy/bin/python $SRCDIR/experiments/basis/splitimagenet/rn50-supsup-basis.py --data="/" --seeds "$SEEDS" --num-masks=50 --seed_model_dir="/scratch/db4045/runs/runs_final/SupsupSeed/rn50-supsup-splitimagenet/id=rn50-supsup-imagenet~seed={seed}~sparsity={sparsity}~try=0/" --logdir-prefix="runs_final" --epochs 75 --gpu-sets="0" --sparsities="$SPARSITIES"
