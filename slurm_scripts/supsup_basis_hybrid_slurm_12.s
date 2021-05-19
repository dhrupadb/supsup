#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=supsup-12-basis
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --account=cds
#SBATCH --output=logs/slurm_supsup_hybrid_basis_12_%j.out

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

/home/db4045/.mypy/bin/python $SRCDIR/experiments/basis/splitcifar100/rn18-supsup-basis-hybrid.py --data="/scratch/db4045/data" --seeds "$SEEDS" --num-masks=12 --seed_model_dir="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup/id=supsup~seed={seed}~sparsity={sparsity}~try=0/" --logdir-prefix="runs_final" --epochs 150 --gpu-sets="0" --sparsities="$SPARSITIES"