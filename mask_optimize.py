
from args import args
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models.module_util import get_subnet
from functools import reduce

CONV_LIST = ['module.layer1.1.conv1.scores',
 'module.layer4.1.conv2.scores',
 'module.layer4.0.conv1.scores',
 'module.layer4.0.conv2.scores',
 'module.layer3.0.conv1.scores',
 'module.layer2.0.conv1.scores',
 'module.layer1.1.conv2.scores',
 'module.layer3.1.conv1.scores',
 'module.layer3.1.conv2.scores',
 'module.layer2.1.conv1.scores',
 'module.conv1.scores',
 'module.layer1.0.conv2.scores',
 'module.layer4.1.conv1.scores',
 'module.layer3.0.conv2.scores',
 'module.layer3.0.shortcut.0.scores',
 'module.layer2.0.shortcut.0.scores',
 'module.layer2.0.conv2.scores',
 'module.linear.scores',
 'module.layer1.0.conv1.scores',
 'module.layer4.0.shortcut.0.scores',
 'module.layer2.1.conv2.scores']

def main():
    supsup_model = torch.load('/scratch/db4045/runs/dhrupad_runs/SupsupSeed/rn18-supsup/id=supsup~seed=0~sparsity=50~try=0/final.pt', map_location="cpu")
    state_dict = supsup_model['state_dict']

    pred_task = args.task_eval
    sparsity = 0.5
    EPOCHS=10000
    NUM_MASKS = 10
#    batch_size = 3600
    key_tasks = [i for i in range(NUM_MASKS)]
    weight_dict = {}

    for conv in CONV_LIST:
        print("Training for Conv: {}".format(conv))
        layer_key = '{}.{}'.format(conv, '{}')
        y = get_subnet(supsup_model['state_dict'][layer_key.format(pred_task)], sparsity).view(-1, )
        x = torch.stack([get_subnet(supsup_model['state_dict'][layer_key.format(i)], sparsity).view(-1,) for i in key_tasks])
        print('x: {}'.format(x.shape))
        print('y: {}'.format(y.shape))
#        train_ds = TensorDataset(x.t(), y)
#        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
#        model = nn.Linear(NUM_MASKS, 1)
#
#        # Define optimizer
#        opt = torch.optim.SGD(model.parameters(), lr=1e-4)
#        loss_fn = F.mse_loss
#
#        def fit(num_epochs, model, loss_fn, opt):
#            for epoch in range(num_epochs):
#                for xb,yb in train_dl:
#                    # Generate predictions
#                    pred = model(xb)
#                    loss = loss_fn(pred, yb)
#                    # Perform gradient descent
#                    loss.backward()
#                    opt.step()
#                    opt.zero_grad()
#            print('Training loss (end): ', loss_fn(model(x.t()), y))
#
#        fit(100, model, loss_fn, opt)
#        print('w: {}'.format(model.weight))
#        bias_size = reduce(lambda x,y: x*y, x.shape[1:], 1)
        w = torch.rand((NUM_MASKS, ), requires_grad=True)
        b = torch.rand((1, ), requires_grad=True)

        def fit(inputs):
            return inputs.t() @ w.t() + b

        def mse(t1, t2):
            diff = t1 - t2
            return torch.sum(diff * diff) / diff.numel()

        print("Training for Conv: {}".format(conv))
        print('Start loss: {}'.format(mse(fit(x), y)))

        # Train for 100 epochs
        loss = 0
        prev_loss = 1e20
        i = 0
        while (i <= EPOCHS) or ((prev_loss - loss)/prev_loss >= 0.01):
            preds = fit(x)
            loss = mse(preds, y)
            loss.backward()
            with torch.no_grad():
                if i < 300:
                    w -= w.grad * 5e-4
                else:
                    w -= w.grad * 1e-4

                b -= b.grad * 1e-5

            i = i+1
            prev_loss = loss

            if (i <= EPOCHS) or ((prev_loss - loss)/prev_loss >= 0.01):
                w.grad.zero_()
                b.grad.zero_()

        weight_dict[conv] = w.tolist()
        print('w: {}'.format(w))
        print('w\': {}'.format(w.grad))
        print('loss: {}'.format(mse(fit(x), y)))
        print("\n\n")

    pickle.dump(weight_dict, open('/scratch/db4045/runs/dhrupad_scratch/SampleModelSaved/StartingAlphas/init_alphas~seed=0~sparsity=50~masks=10~task={}.pkl'.format(pred_task), 'wb'))

if __name__ == "__main__":
    main()

