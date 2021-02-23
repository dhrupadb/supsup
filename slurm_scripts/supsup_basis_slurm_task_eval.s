#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=dsga1006-supsup-basis
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:k80:1
#SBATCH --output=logs/slurm_supsup_basis_task_eval_test_%j.out

# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch
# for more information about the above options

# Remove all unused system modules
module purge
module load cuda/10.1.105
module load anaconda3/5.3.1

# Move into the directory that contains our code
SRCDIR=$(pwd)


rm -rf /scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/
/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_10/id=supsup~seed=0~sparsity=30~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_highsparsity_multimask_test --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --epochs 10


#rm -rf /scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/

#echo "Training Stage"
#echo "Train alphas from equal weighted:"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_7/id=supsup~seed=0~sparsity=32~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --train_mask_alphas
#
#echo "Standard: Use the correct mask"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_7/id=supsup~seed=0~sparsity=32~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/
#
#echo "Train alphas from optimal"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_7/id=supsup~seed=0~sparsity=32~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --train_mask_alphas --start_at_optimal
#
#echo "Use specific (correct) mask"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_7/id=supsup~seed=0~sparsity=32~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --use-single-mask 3
#
#echo "Use specific (wrong) mask"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_7/id=supsup~seed=0~sparsity=32~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --use-single-mask 4




















#rm -rf /scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/

#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/ --config $SRCDIR/experiments/seeds/splitcifar100/configs/rn18-supsup_5.yaml --multigpu="0" --epochs 10 --multigpu="0" --sparsity 8 --name dhrupad_seed_mask

echo "Inference using Multimask(basis)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/
/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs_gpu/SupsupSeed/rn18-supsup_num_masks_3/id=supsup~seed=0~sparsity=32~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 2 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/

#echo "Inference using SupSup(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_supsup --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --sparsity 8 --epochs 0
#echo "Inference using Multimask(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --num_seed_tasks_learned 5 --sparsity 8 --epochs 0
#echo "Inference using SupSup(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_supsup --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --sparsity 8
#echo "Inference using Multimask(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --num_seed_tasks_learned 5 --sparsity 8
