#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=30GB
#SBATCH --job-name=supsup-task-eval
#SBATCH --mail-type=END
#SBATCH --mail-user=db4045@nyu.edu
#SBATCH --gres=gpu:1
#SbATCH --account=cds
#SBATCH --output=logs/slurm_supsup_task_eval_test_%j.out

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
srcdir=$(pwd)


#rm -rf /scratch/db4045/runs/dhrupad_seed_epoch10_single/Supsup/
#echo "Use specific (correct) mask"
#/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --num_seed_tasks_learned 3 --sparsity 25 --epochs 10 --num-tasks 5 --train_mask_alphas --single_task_only --single_task_only_task 0 --task-eval 0

/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $srcdir/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/supsupbasis/ --num_seed_tasks_learned 3 --sparsity 25 --epochs 1 --num-tasks 5 --train_mask_alphas --task-eval 0 --use-single-mask 1
#/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $srcdir/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/supsupbasis/ --num_seed_tasks_learned 15 --sparsity 25 --epochs 1 --num-tasks 5 --train_mask_alphas --task-eval 0 --use-single-mask 1
#/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $srcdir/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/supsupbasis/ --num_seed_tasks_learned 15 --sparsity 25 --epochs 1 --num-tasks 5 --train_mask_alphas --task-eval 0 --use-single-mask 2
#/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $srcdir/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/supsupbasis/ --num_seed_tasks_learned 15 --sparsity 25 --epochs 1 --num-tasks 5 --train_mask_alphas --task-eval 0 --use-single-mask 3
#/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $srcdir/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/supsupbasis/ --num_seed_tasks_learned 15 --sparsity 25 --epochs 1 --num-tasks 5 --train_mask_alphas --task-eval 0 --use-single-mask 4
#/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed=0/id=supsup~seed=1~sparsity=25~try=0/final.pt" --config $srcdir/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/supsupbasis/ --num_seed_tasks_learned 15 --sparsity 25 --epochs 1 --num-tasks 5 --train_mask_alphas --task-eval 0 --use-single-mask 0

#echo "   "
#echo "==========================="
#echo "==========================="
#echo "==========================="
#echo "   "
#
#echo "Use specific \(correct\) mask"
#/home/db4045/.mypy/bin/python $SRCDIR/basis_single_task.py --data="/scratch/db4045/data" --seed=0 --seed-model-format="/scratch/db4045/runs/runs_final/SupsupSeed/rn18-supsup-task0~root_seed={seed}/id=supsup~seed={task}~sparsity={sparsity}~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --num_seed_tasks_learned 3 --sparsity 25 --epochs 10 --num-tasks 5 --train_mask_alphas --single_task_only --single_task_only_task 0 --task-eval 1
#/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=50~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --multigpu="0" --name dhrupad_supsup_softmask_test --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --epochs 10

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

#/scratch/db4045/capstone_env/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_basis_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/

#echo "Task Eval Standard (Supsup)"
#/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=50~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup.yaml --multigpu="0" --name dhrupad_supsup_softmask_test --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --task-eval 2 --num-seed-tasks-learned 5
#
#rm -rf /scratch/db4045/runs/dhrupad_scratch/SupsupHardInference/
#echo "Inference - Standard (Supsup) on Soft Masks"
#for sparsity in 10 25 30 40 50 60 70 80 90 95 99; do
#for task in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
#    echo "Running inference on task: "$task" at sparsity: "$sparsity
#    /home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeedWeightnorm/rn18-supsup/id=supsup-weightnorm~seed=0~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup.yaml --multigpu="0" --name 'dhrupad_supsup_hard_on_soft_test_'$task'_'$sparsity --log-dir=/scratch/db4045/runs/dhrupad_scratch/SupsupHardInference/ --task-eval $task --num-seed-tasks-learned 20 --sparsity $sparsity
#done
#done


#echo "Task Eval Basis"
#/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=50~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis.yaml --multigpu="0" --name dhrupad_supsup_softmask_test --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --task-eval 2 --num-seed-tasks-learned 5
#
#echo "Inference using Weight-Normalized(Masks)"
#/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=50~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-weightnorm.yaml --multigpu="0" --name dhrupad_supsup_softmask_test --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --task-eval 2 --num-seed-tasks-learned 5

#echo "Inference using Soft(Masks)"
#/home/db4045/.mypy/bin/python $SRCDIR/basis.py --data="/scratch/db4045/data" --seed=0 --seed-model="/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=50~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-soft.yaml --multigpu="0" --name dhrupad_supsup_softmask_test --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --task-eval 2 --num-seed-tasks-learned 5
#
#echo "Inference using SupSup(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_supsup --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --sparsity 8 --epochs 0
#echo "Inference using Multimask(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --num_seed_tasks_learned 5 --sparsity 8 --epochs 0
#echo "Inference using SupSup(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_supsup --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --sparsity 8
#echo "Inference using Multimask(main)"
#/scratch/db4045/capstone_env/bin/python $SRCDIR/main.py --data="/scratch/db4045/data" --seed=0 --resume="/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupSeed/dhrupad_seed_mask~try=0/final.pt" --config $SRCDIR/experiments/basis/splitcifar100/configs/rn18-supsup-basis-multitask.yaml --multigpu="0" --task-eval 3 --name dhrupad_main_multimask --log-dir=/scratch/db4045/runs/dhrupad_seed_epoch10_single/SupsupBasis/ --num_seed_tasks_learned 5 --sparsity 8
