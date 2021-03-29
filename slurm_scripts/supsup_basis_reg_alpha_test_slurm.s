#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=23:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=supsup-regalpha
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --account=cds
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_supsup_regalpha_10_%j.out

# Remove all unused system modules
module purge
module load cuda/10.2.89

export MYPY_ENV='supsup'
MYPY_ENV="${MYPY_ENV:-myenv}"
OVERLAY=$MYPY_ROOT/containers/$MYPY_ENV.ext3

# Move into the directory that contains our code
SRCDIR=$(pwd)

LR=$1
WD=$2
AL=$3
AN=$4
echo "Learning Rate: "$LR
echo "Weight Decay (L2 Adam): "$WD
echo "Alpha Lambda: "$AL
echo "Alpha Norm (L "$AN")"

echo "Standard Basis (Hard Masks). Default"
/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=75~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --name 'dhrupad_supsup_regularized_alphas_hard_masks~al='$AL'~wd='$WD'~LR='$LR --log-dir=/scratch/db4045/runs/dhrupad_runs/AlphaMultiNorm/ --num-seed-tasks-learned 10 --epochs 150 --al $AL --trainer alphareg --wd $WD --lr $LR --multigpu="0" --log-alphas --train_mask_alphas --alpha-norm $AN

echo
echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
echo

echo "Soft Basis (Soft Masks)."
/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --name 'dhrupad_supsup_regularized_alphas_soft_masks~al='$AL'~wd='$WD'~LR='$LR --log-dir=/scratch/db4045/runs/dhrupad_runs/AlphaMultiNorm/ --num-seed-tasks-learned 10 --epochs 150 --al $AL --trainer alphareg --wd $WD --lr $LR --multigpu="0" --log-alphas --train_mask_alphas --alpha-norm $AN


##echo 
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo 
##
##echo "Modified Trainer, Alpha off, L2 on overall weights."
##/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --name 'dhrupad_supsup_regularized_alpha_task_'$task --log-dir=/scratch/db4045/runs/dhrupad_scratch/SupsupRegAlpha/ --num-seed-tasks-learned 10 --epochs 75 --al 0.0 --trainer alphareg --lr 0.02 --multigpu="0" 
##
##echo 
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo 
##
##echo "Baseline Trainer, Alpha off, L2 on overall weights."
##/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --name 'dhrupad_supsup_regularized_alpha_task_'$task --log-dir=/scratch/db4045/runs/dhrupad_scratch/SupsupRegAlpha/ --num-seed-tasks-learned 10 --epochs 75 --lr 0.02 --multigpu="0" 
##
##echo 
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo 
##
##echo "Restrict params in optimizer."
##echo "Modified Trainer, Alpha regularization, No L2 on overall weights."
##/home/db4045/.mypy/bin/python $SRCDIR/basis_param_restricted.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --name 'dhrupad_supsup_regularized_alpha_task_'$task --log-dir=/scratch/db4045/runs/dhrupad_scratch/SupsupRegAlpha/ --num-seed-tasks-learned 10 --epochs 75 --al 0.00001 --trainer alphareg --wd 0.0 --lr 0.02 --multigpu="0" 
##echo 
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo
##
##echo "Modified Trainer, Alpha off, L2 on overall weights."
##/home/db4045/.mypy/bin/python $SRCDIR/basis_param_restricted.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --name 'dhrupad_supsup_regularized_alpha_task_'$task --log-dir=/scratch/db4045/runs/dhrupad_scratch/SupsupRegAlpha/ --num-seed-tasks-learned 10 --epochs 75 --al 0.0 --trainer alphareg --lr 0.02 --multigpu="0" 
##
##echo 
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
##echo 
##
##echo "Baseline Trainer, Alpha off, L2 on overall weights."
##/home/db4045/.mypy/bin/python $SRCDIR/basis_param_restricted.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --name 'dhrupad_supsup_regularized_alpha_task_'$task --log-dir=/scratch/db4045/runs/dhrupad_scratch/SupsupRegAlpha/ --num-seed-tasks-learned 10 --epochs 75 --lr 0.02 --multigpu="0" 
