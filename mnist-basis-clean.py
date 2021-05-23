import os
import click
import numpy as np
import pandas as pd
import random

from mnist import *

@click.command()
@click.option("--seed", default=0, help="Model Seed to use.", required=True)
@click.option("--sparsity", default=5, help="Model Sparsity to use", required=True)
@click.option("--num_tasks", default=50, help="Number of tasks to use", required=True)
@click.option("--num_seed_tasks_learned", default=40, help="Number of tasks to use", required=False)
@click.option("--model_type", help="Model Type: Supsup or Basis", required=True)
@click.option("--log_dir", help="output log dir", required=True)
@click.option("--output_dir", help="model output dir", required=True)
@click.option("--seed_model", help="seed_model_to_use", required=False)
@click.option("--data_root", help="data directory", required=True)
@click.option("--dataset", help="MNIST Dataset Variant: MNISTPerm vs MNISTRot vs MNISTSplit", required=True)
@click.option("--epochs", default=2, help="Number of epochs", required=False)
@click.option("--single_task_mode_task", default=-1, help="To create multiple masks from the same task. -1 to disable.", required=False)
def run(seed, sparsity, num_tasks, model_type, log_dir, data_root, output_dir, single_task_mode_task, dataset, seed_model, epochs, num_seed_tasks_learned):
    random.seed(int(seed))
    np.random.seed(int(seed))  # Numpy module.
    torch.manual_seed(int(seed))

    mnist = MNISTPerm(data_root=data_root, seed=seed) if dataset == 'MNISTPerm' else MNISTRot(data_root=data_root, seed=seed) if dataset == 'MNISTRot' else MNISTSplit(data_root=data_root, seed=seed)
    if model_type.lower() == 'supsup' and single_task_mode_task == -1:
        model = MultitaskFC(hidden_size=300, num_tasks=num_tasks, sparsity=sparsity/100)

        for task in range(num_tasks):
            print(f"Training for task {task}")
            set_model_task(model, task)
            mnist.update_task(task)

            optimizer = optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            # Train for 1 epoch
            for e in range(epochs):
                train(model, mnist.train_loader, optimizer, e)

                print("Validation")
                print("============")
                acc1 = evaluate(model, mnist.val_loader, e)


            cache_masks(model)
            print()
            set_num_tasks_learned(model, task + 1)
            print()


        gg_performance = []
        for task in range(num_tasks):
            set_model_task(model, task)
            mnist.update_task(task)
            acc1 = evaluate(model, mnist.val_loader, 0)
            gg_performance.append(acc1.item())

        print(f"Average top 1 performance: {(sum(gg_performance) / len(gg_performance)):.4f}")

        print("Per task performance")
        for t in range(num_tasks):
            print(f"Task {t}: {gg_performance[t]:.4f}")

        df = pd.Series(gg_performance).reset_index()
        df.columns = ['Task', 'Performance']
        df['Log dir'] = log_dir
        df['Seed'] = seed
        df['Dataset'] = dataset
        exp_out_dir = os.path.join(output_dir, 'supsup-LEnet~dataset={}/seed={}~sparsity={}~num_tasks={}'.format(dataset, seed, sparsity, num_tasks))
        if not os.path.exists(exp_out_dir):
            os.makedirs(exp_out_dir)

        df.to_csv(os.path.join(exp_out_dir, 'results.csv'), index=False)
        torch.save(model, os.path.join(exp_out_dir, 'model.pt'))

    elif model_type.lower() == 'supsup' and single_task_mode_task > -1:
        # ### Multimask Same Task
        model = torch.load(seed_model, map_location=torch.device('cpu'))
        num_masks_to_create = num_tasks

        loaders = {}
        for i in range(num_masks_to_create):
            loaders[i] = MNISTPerm(data_root=data_root, seed=i)

        weight_dict = {k: v for k,v in model.state_dict().items() if k.endswith('weight')}

        model_map = {}
        for i in range(num_masks_to_create):
            modeli = MultitaskFC(hidden_size=300, num_tasks=1, sparsity=sparsity/100)
            sdi = modeli.state_dict()
            sdi.update(weight_dict)
            modeli.load_state_dict(sdi)
            model_map[i] = modeli

        for idx, modeli in model_map.items():
            print(f"Training for model {idx} on task {single_task_mode_task}")
            set_model_task(modeli, 0)
            mnisti = loaders[idx]
            mnisti.update_task(single_task_mode_task)

            optimizer1 = optim.RMSprop([p for p in modeli.parameters() if p.requires_grad], lr=1e-4)
            # Train for 1 epoch
            for e in range(epochs):
                train(modeli, mnisti.train_loader, optimizer1, e)

                print("Validation")
                print("============")
                acc1 = evaluate(modeli, mnisti.val_loader, e)


            cache_masks(modeli)
            print()
            set_num_tasks_learned(modeli, 1)
            print()

            exp_out_dir = os.path.join(output_dir, 'supsup-LEnet-single-task~dataset={}/seed={}~sparsity={}~num_tasks={}'.format(dataset, seed, sparsity, num_tasks))
            if not os.path.exists(exp_out_dir):
                os.makedirs(exp_out_dir)

            torch.save(modeli, os.path.join(exp_out_dir, 'model_{}.pt'.format(idx)))



#    def overlap(mask1, mask2):
#        assert mask1.shape == mask2.shape
#        count_same = (mask1 * mask2).sum()
#        return count_same / ((mask1 + mask2) > 0).sum()
#
#    from itertools import product
#
#    task = 0
#    for layer, modelidx in product([0,2,4], product([i for i in range(15)], [i for i in range(15)])):
#        modeli, modelj = modelidx
#        if modeli == modelj:
#            continue
#        print('Layer: {}, Models: ({}, {}), Overlap: {}'.format(layer, modeli, modelj,
#            overlap(model_map[modeli].state_dict()['model.{}.stacked'.format(layer)][task],
#                    model_map[modelj].state_dict()['model.{}.stacked'.format(layer)][task])))
#
    elif model_type.lower() == 'basis':
        # # Basis Masks

        # ### Model Initialization (Only New Tasks)
        assert num_seed_tasks_learned < num_tasks
        basis_model = BasisMultitaskFC(hidden_size=300, num_tasks=num_tasks, num_seed_tasks_learned=num_seed_tasks_learned, start_at_optimal=True, sparsity=0.25)


        seed_dict = model.state_dict()
        basis_dict = basis_model.state_dict()
        load_dict = {k: seed_dict[k] for k in basis_model.state_dict().keys() if k in seed_dict.keys()}
        basis_dict.update(load_dict)
        basis_model.load_state_dict(basis_dict, False)
        cache_masks(basis_model)

        for task in range(num_seed_tasks_learned, num_tasks):
            print(f"Training for task {task}")
            set_model_task(basis_model, task)
            mnist.update_task(task)

            optimizer = optim.RMSprop([p for p in basis_model.parameters() if p.requires_grad], lr=1e-3)
            # Train for 1 epoch
            for e in range(epochs):
                train(basis_model, mnist.train_loader, optimizer, e)

                print("Validation")
                print("============")
                acc1 = evaluate(basis_model, mnist.val_loader, e)


            cache_masks(basis_model)
            print()
            set_num_tasks_learned(basis_model, task + 1)
            print()


        # In[29]:


        # When task ID we can simply set the mask and evaluate

        gg_performance = []
        for task in range(num_tasks):
            set_model_task(basis_model, task)
            mnist.update_task(task)
            acc1 = evaluate(basis_model, mnist.val_loader, 0)
            gg_performance.append(acc1.item())

        clear_output()

        print(f"Average top 1 performance: {(sum(gg_performance) / len(gg_performance)):.4f}")

        print("Per task performance")
        for t in range(num_tasks):
            print(f"Task {t}: {gg_performance[t]:.4f}")


        # In[30]:


        performance_map['basis_mnist'] = gg_performance.copy()


        # ### Model Initialization (Cross task analysis) -- Using 1 mask from task 0

        # In[31]:


        num_tasks = 10 # For demonstration purposes, we go up to 2500 in our paper
        num_seed_tasks_learned = 1
        basis_model_f = BasisMultitaskFC(hidden_size=300,
                                    num_tasks=num_tasks, num_seed_tasks_learned=num_seed_tasks_learned, start_at_optimal=True, sparsity=0.25)


        # In[32]:


        custom_state_dict = basis_model_f.state_dict().copy()
        custom_state_dict.update({k:v for k,v in model.state_dict().items() if k in custom_state_dict.keys()})


        # In[33]:


        for k in custom_state_dict.keys():
            if k.startswith('model.0.scores.'):
                custom_state_dict[k] = model.state_dict()['model.0.scores.0']
            elif k.startswith('model.2.scores.'):
                custom_state_dict[k] = model.state_dict()['model.2.scores.0']
            elif k.startswith('model.4.scores.'):
                custom_state_dict[k] = model.state_dict()['model.4.scores.0']
            elif k.startswith('model.0.basis_alphas.'):
                custom_state_dict[k] = custom_state_dict['model.0.basis_alphas.0']
            elif k.startswith('model.2.basis_alphas.'):
                custom_state_dict[k] = custom_state_dict['model.2.basis_alphas.0']
            elif k.startswith('model.4.basis_alphas.'):
                custom_state_dict[k] = custom_state_dict['model.4.basis_alphas.0']


        # In[34]:


        basis_model_f.load_state_dict(custom_state_dict, False)
        cache_masks(basis_model_f)

        # When task ID we can simply set the mask and evaluate

        gg_performance = []
        for task in range(num_tasks):
            set_model_task(basis_model_f, task)
            mnist.update_task(task)
            acc1 = evaluate(basis_model_f, mnist.val_loader, 0)
            gg_performance.append(acc1.item())

        clear_output()

        print(f"Average top 1 performance: {(sum(gg_performance) / len(gg_performance)):.4f}")

        print("Per task performance")
        for t in range(num_tasks):
            print(f"Task {t}: {gg_performance[t]:.4f}")


        # In[37]:


        performance_map['basis_mnist_frozen'] = gg_performance.copy()


        # ### Model Initialization (Only New Tasks) , 15 masks all from task 0

        num_tasks = 50 # For demonstration purposes, we go up to 2500 in our paper
        num_seed_tasks_learned = 40
        basis_model_zeros = BasisMultitaskFC(hidden_size=300, num_tasks=num_tasks, num_seed_tasks_learned=num_seed_tasks_learned, start_at_optimal=False, sparsity=0.25)

        seed_dict = basis_model_zeros.state_dict()


        update_dict = seed_dict.copy()


        assert all([(model_map[0].state_dict()['model.0.weight'] == model_map[i].state_dict()['model.0.weight']).all() for i in range(1,15)])
        update_dict['model.0.weight'] = model_map[0].state_dict()['model.0.weight']
        for task in range(40):
            for layer in [0,2,4]:
                update_dict['model.{}.scores.{}'.format(layer, task)] = model_map[task].state_dict()['model.{}.scores.0'.format(layer)]


        seed_dict.update(update_dict)
        basis_model_zeros.load_state_dict(seed_dict, False)
        cache_masks(basis_model_zeros)


        for task in range(0, num_tasks):
            print(f"Training for task {task}")
            set_model_task(basis_model_zeros, task)
            mnist.update_task(task)

            optimizer = optim.RMSprop([p for p in basis_model_zeros.parameters() if p.requires_grad], lr=1e-3)
            # Train for 1 epoch
            for e in range(epochs):
                train(basis_model_zeros, mnist.train_loader, optimizer, e)

                print("Validation")
                print("============")
                acc1 = evaluate(basis_model_zeros, mnist.val_loader, e)


            cache_masks(basis_model_zeros)
            print()
            set_num_tasks_learned(basis_model_zeros, task + 1)
            print()


        # In[96]:


        # When task ID we can simply set the mask and evaluate

        gg_performance = []
        for task in range(num_tasks):
            set_model_task(basis_model_zeros, task)
            mnist.update_task(task)
            acc1 = evaluate(basis_model_zeros, mnist.val_loader, 0)
            gg_performance.append(acc1.item())

        clear_output()

        print(f"Average top 1 performance: {(sum(gg_performance) / len(gg_performance)):.4f}")

        print("Per task performance")
        for t in range(num_tasks):
            print(f"Task {t}: {gg_performance[t]:.4f}")


        # In[97]:


        performance_map['basis_mnist_task_zero_only'] = gg_performance.copy()


if __name__ == '__main__':
    run()
