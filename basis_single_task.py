import os
import pathlib
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from args import args
import adaptors
import data
import schedulers
import trainers
import utils

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))  # Numpy module.
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "settings.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    print(f"=> Saving data in {run_base_dir}")

    # Get model with correct architecture and load in state dict.
    model = utils.get_model()
    model = utils.set_gpu(model)

    num_tasks_learned = args.num_seed_tasks_learned if not args.start_at_optimal else 0

    if args.er_sparsity:
        for n, m in model.named_modules():
            if hasattr(m, "sparsity"):
                m.sparsity = min(
                    0.5,
                    args.sparsity
                    * (m.weight.size(0) + m.weight.size(1))
                    / (
                        m.weight.size(0)
                        * m.weight.size(1)
                        * m.weight.size(2)
                        * m.weight.size(3)
                    ),
                )
                print(f"Set sparsity of {n} to {m.sparsity}")

    ### Load seed model into trained model:
    model_dict = model.state_dict()
    # Formats: module.conv1.weight or module.conv1.scores.4
    # also for BatchN: module.layerX.X.conv{1,2} ==> module.layerX.X.bn{1,2}.bns.{running_mean, running_va, num_batches_tracked}
    layer_fmt = ['module.conv1.{}',
                  'module.layer1.0.conv1.{}',
                  'module.layer1.0.conv2.{}',
                  'module.layer1.1.conv1.{}',
                  'module.layer1.1.conv2.{}',
                  'module.layer2.0.conv1.{}',
                  'module.layer2.0.conv2.{}',
                  'module.layer2.0.shortcut.0.{}',
                  'module.layer2.1.conv1.{}',
                  'module.layer2.1.conv2.{}',
                  'module.layer3.0.conv1.{}',
                  'module.layer3.0.conv2.{}',
                  'module.layer3.0.shortcut.0.{}',
                  'module.layer3.1.conv1.{}',
                  'module.layer3.1.conv2.{}',
                  'module.layer4.0.conv1.{}',
                  'module.layer4.0.conv2.{}',
                  'module.layer4.0.shortcut.0.{}',
                  'module.layer4.1.conv1.{}',
                  'module.layer4.1.conv2.{}',
                  'module.linear.{}']
    assert not ((args.num_tasks - args.num_seed_tasks_learned > 1) and (args.conv_type == 'BasisMaskConv')), 'BasisMaskConv only supports learning one extra task over the mask tasks. Please fix config or change conv_type!'
    # Setup where you use n mask(s) from a given task from n different model runs.
    print('Seed model format: {}'.format(args.seed_model_format))
    # Assert at each level that the backbone network remains unchanged.
    # Iterate and load n models, and update state dict parameter with the params of a given task into this model.
    task = 0
    model_path = args.seed_model_format.format(task=task, sparsity=int(args.sparsity), seed=args.seed)
    if os.path.isfile(model_path):
        print(f"=> Loading seed model from '{model_path}'")
        checkpoint = torch.load(
            model_path, map_location=f"cuda:{args.multigpu[0]}" if torch.cuda.is_available() else torch.device('cpu')
        )
        pretrained_dict = checkpoint["state_dict"]
        seed_args = checkpoint['args']
    else:
        raise RuntimeError(f"=> No seed model found at '{model_path}'!")

    sub_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
    }
#    print("Updating following keys: {}".format(','.join(sub_dict.keys())))
    model_dict.update(sub_dict)

    tlist = [i for i in range(1, args.num_seed_tasks_learned)]
    for task in tlist:
        model_path = args.seed_model_format.format(task=task, sparsity=int(args.sparsity), seed=args.seed)
        if os.path.isfile(model_path):
            print(f"=> Loading seed model from '{model_path}'")
            checkpoint = torch.load(
                model_path, map_location=f"cuda:{args.multigpu[0]}" if torch.cuda.is_available() else torch.device('cpu')
            )
            pretrained_dict = checkpoint["state_dict"]
            seed_args = checkpoint['args']
        else:
            raise RuntimeError(f"=> No seed model found at '{model_path}'!")

#        assert args.seed == seed_args.seed, "Seeds must be the same for backbone networks to match! Expected seed: {}, found: {}".format(args.seed, seed_args.seed)
        for layer in layer_fmt:
            key = layer.format('weight')
            seed_task_key = layer.format('scores.{}'.format(args.single_task_only_task))

            assert (model_dict[key] == pretrained_dict[key]).all(), "Weights must match! Match failed for layer: {}, task: {}".format(key, task)

            if 'conv' in layer and 'layer' in layer:
                bn = layer.replace('conv1.{}', 'bn1.bns.{ttask}.{stat}').replace('conv2.{}', 'bn2.bns.{ttask}.{stat}')
                for stat in ['running_mean', 'running_var', 'num_batches_tracked']:
                    model_dict[bn.format(ttask=task, stat=stat)] = pretrained_dict[bn.format(ttask=args.single_task_only_task, stat=stat)]

            model_dict[layer.format('scores.{}'.format(task))] = pretrained_dict[seed_task_key]

            if args.start_at_optimal and args.single_task_only_task == task:
                model_dict[layer.format('basis_alphas.{}'.format(task))] = model_dict[layer.format('basis_alphas.{}'.format(args.single_task_only_task))]
    model.load_state_dict(model_dict)
    #model.eval()

    # Get dataloader.
    data_loader = getattr(data, args.set)()

    # Track accuracy on all tasks.
    if args.num_tasks:
        best_acc1 = [0.0]*args.num_tasks
        curr_acc1 = [0.0]*args.num_tasks
        adapt_acc1 = [0.0]*args.num_tasks

    criterion = nn.CrossEntropyLoss().to(args.device)

    writer = SummaryWriter(log_dir=run_base_dir)

    trainer = getattr(trainers, args.trainer or "default")
    print(f"=> Using trainer {trainer}")

    train, test = trainer.train, trainer.test

    # Initialize model specific context (editorial note: avoids polluting main file)
    if hasattr(trainer, "init"):
        trainer.init(args)

    if args.task_eval is not None:
        assert 0 <= args.task_eval < args.num_tasks, "Not a valid task idx"
        print(f"Task {args.set}: {args.task_eval}")

        # Settting task to -1 tells the model to infer task identity instead of being given the task.
        model.apply(lambda m: setattr(m, "task", -1))

        # an "adaptor" is used to infer task identity.
        # args.adaptor == gt implies we are in scenario GG.

        # This will cache all of the information the model needs for inferring task identity.
        if args.adaptor != "gt":
            utils.cache_masks(model)

        # Iterate through all tasks.
        adapt = getattr(adaptors, args.adaptor)

        # Update the data loader so it is returning data for the right task.
        data_loader.update_task(args.task_eval)

        # Clear the stored information -- memory leak happens if not.
        for p in model.parameters():
            p.grad = None

        for b in model.buffers():
            b.grad = None

        torch.cuda.empty_cache()

        adapt_acc = adapt(
            model,
            writer,
            data_loader.val_loader,
            num_tasks_learned,
            args.task_eval,
        )

        torch.cuda.empty_cache()
        utils.write_adapt_results(
            name=args.name,
            task=f"{args.set}_{args.task_eval}",
            num_tasks_learned=num_tasks_learned,
            curr_acc1=0.0,
            adapt_acc1=adapt_acc,
            task_number=args.task_eval,
        )

        utils.clear_masks(model)
        torch.cuda.empty_cache()
        return

    # Iterate through all new tasks that were not used for training masks.
    for idx in range(args.num_seed_tasks_learned if not args.train_mask_alphas else 0, args.num_tasks):
        print(f"Task {args.set}: {idx}")

        # Tell the model which task it is trying to solve -- in Scenario NNs this is ignored.
        model.apply(lambda m: setattr(m, "task", idx))

        # Update the data loader so that it returns the data for the correct task, also done by passing the task index.
        assert hasattr(
            data_loader, "update_task"
        ), "[ERROR] Need to implement update task method for use with multitask experiments"

        data_loader.update_task(idx)

        # Clear the grad on all the parameters.
        for p in model.parameters():
            p.grad = None

        # Make a list of the parameters relavent to this task.
        params = []
        params_names = []
        for n, p in model.named_parameters():
            if 'Basis' in args.conv_type:
                if p.requires_grad and int(n.split('.')[-1]) == idx:
                    params.append(p)
                    params_names.append(n)
            elif p.requires_grad:
                params.append(p)

        print(f"Optimizing over the following params: \n {','.join(params_names)}")

        # train_weight_tasks specifies the number of tasks that the weights are trained for.
        # e.g. in SupSup, train_weight_tasks = 0. in BatchE, train_weight_tasks = 1.
        # If training weights, use train_weight_lr. Else use lr.
        lr = (
            args.train_weight_lr
            if args.train_weight_tasks < 0
            or num_tasks_learned < args.train_weight_tasks
            else args.lr
        )

        # get optimizer, scheduler
        if args.optimizer == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=args.wd)
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(params, lr=lr)
        else:
            optimizer = optim.SGD(
                params, lr=lr, momentum=args.momentum, weight_decay=args.wd
            )

        train_epochs = args.epochs

        if args.no_scheduler:
            scheduler = None
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)

        # Train on the current task.
        for epoch in range(1, train_epochs + 1):
            train(
                model,
                writer,
                data_loader.train_loader,
                optimizer,
                criterion,
                epoch,
                idx,
                data_loader,
                log_alphas=args.log_alphas
            )

            # Required for our PSP implementation, not used otherwise.
#            utils.cache_weights(model, num_tasks_learned + 1)

            curr_acc1[idx] = test(
                model, writer, criterion, data_loader.val_loader, epoch, idx
            )
            if curr_acc1[idx] > best_acc1[idx]:
                best_acc1[idx] = curr_acc1[idx]
            if scheduler:
                scheduler.step()

            if (
                args.iter_lim > 0
                and len(data_loader.train_loader) * epoch > args.iter_lim
            ):
                break

        utils.write_result_to_csv(
            name=f"{args.name}~set={args.set}~task={idx}",
            curr_acc1=curr_acc1[idx],
            best_acc1=best_acc1[idx],
            save_dir=run_base_dir,
        )

        # Save memory by deleting the optimizer and scheduler.
        del optimizer, scheduler, params

        # Increment the number of tasks learned.
        num_tasks_learned += 1

        # If operating in NNS scenario, get the number of tasks learned count from the model.
        if args.trainer and "nns" in args.trainer:
            model.apply(
                lambda m: setattr(
                    m, "num_tasks_learned", min(model.num_tasks_learned, args.num_tasks)
                )
            )
        else:
            model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks_learned))


    # Run inference on all the tasks.
    avg_acc = 0.0
    avg_correct = 0.0

    # Settting task to -1 tells the model to infer task identity instead of being given the task.
    model.apply(lambda m: setattr(m, "task", -1))

    # an "adaptor" is used to infer task identity.
    # args.adaptor == gt implies we are in scenario GG.

    # This will cache all of the information the model needs for inferring task identity.
    if args.adaptor != "gt":
        utils.cache_masks(model)

    # Iterate through all tasks.
    adapt = getattr(adaptors, args.adaptor)

    for i in range(args.num_tasks):
        print(f"Testing {i}: {args.set} ({i})")
        # model.apply(lambda m: setattr(m, "task", i))

        # Update the data loader so it is returning data for the right task.
        data_loader.update_task(i)

        # Clear the stored information -- memory leak happens if not.
        for p in model.parameters():
            p.grad = None

        for b in model.buffers():
            b.grad = None

        torch.cuda.empty_cache()

        adapt_acc = adapt(
            model,
            writer,
            data_loader.val_loader,
            num_tasks_learned,
            i,
        )

        adapt_acc1[i] = adapt_acc
        avg_acc += adapt_acc

        torch.cuda.empty_cache()
        utils.write_adapt_results(
            name=args.name,
            task=f"{args.set}_{i}",
            num_tasks_learned=num_tasks_learned,
            curr_acc1=curr_acc1[i],
            adapt_acc1=adapt_acc,
            task_number=i,
        )

    writer.add_scalar(
        "adapt/avg_acc", avg_acc / num_tasks_learned, num_tasks_learned
    )

    utils.clear_masks(model)
    torch.cuda.empty_cache()


    if args.save:
        torch.save(
            {
                "epoch": args.epochs,
                "arch": args.model,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "curr_acc1": curr_acc1,
                "args": args,
            },
            run_base_dir / "basis_final.pt",
        )


    return adapt_acc1


# TODO: Remove this with task-eval
def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {"params": bn_params, "weight_decay": args.wd,},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=False,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.wd,
        )
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )

    return optimizer


if __name__ == "__main__":
    main()
