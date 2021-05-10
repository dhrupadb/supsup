import sys, os
sys.path.append(os.path.abspath('.'))
from main import main as run
from args import args

def main():
    args.set = 'RandSplitCIFAR100'

    args.seed = 1996
    args.multigpu = [0]
    args.model = 'GEMResNet18'
    #args.conv_type = 'MultitaskMaskConv'
    args.conv_type = 'BasisMultitaskMaskConv'
    args.bn_type = 'MultitaskNonAffineBN'
    args.conv_init = 'signed_constant'
    args.num_seed_tasks_learned = 3
    args.task_eval = 1

    args.output_size = 5
    args.er_sparsity = True
    args.sparsity = 32

    args.adaptor = "gt"
    args.hard_alphas = True

    args.batch_size = 128
    args.test_batch_size = 128
    args.num_tasks = 3

    args.optimizer = 'adam'
    args.lr = 0.001
    args.eval_ckpts = []
    args.train_mask_alphas = True

    args.name = f"id=rn18_load"

    # TODO: Change these paths!
#    args.data = '/home/mitchnw/data'
#    args.log_dir = "/home/mitchnw/ssd/checkpoints/supsup_test"
    args.data = '/scratch/db4045/data/'
    args.log_dir = "/scratch/db4045/scratch/test_bn_fix"

    # Resume the checkpoint
    args.resume = '/scratch/db4045/scratch/test_bn_fix/id=rn18~try=0/final.pt'

    # Note: changed to 0 as we do not want any more training.
    args.epochs = 0

    run()

if __name__ == '__main__':
    main()
