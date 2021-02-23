#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=dsga1006-supsup-basis
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --output=logs/slurm_supsup_task_eval_multimask_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load cuda/10.1.105
module load anaconda3/5.3.1

# Move into the directory that contains our code
SRCDIR=$HOME/git/supsup

# Activate the conda environment
# source ~/.bashrc
# conda activate dsga3001
# source env/bin/activate

# Execute the script
# conda activate supsup
# conda install -y pytorch==1.5.1 torchvision==0.6.1 -c pytorch
# pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

#rm -rf /scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/ --config $SRCDIR/experiments/seeds/splitcifar100/configs/rn18-supsup_5.yaml --multigpu="0" --epochs 10 --multigpu="0" --sparsity 8 --name dhrupad_seed_mask

/scratch/db4045/capstone_env/bin/python $SRCDIR/basis_cpu.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup_num_masks_5/id=supsup~seed=0~sparsity=4~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_test/SupsupBasis/ --epochs 3
