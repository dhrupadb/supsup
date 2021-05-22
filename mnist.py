import torch
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torchvision
import numpy as np
import math

# Subnetwork forward from hidden networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std


class MultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, num_tasks=1, sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.sparsity = sparsity
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    GetSubnet.apply(self.scores[j], self.sparsity)
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)
            subnet = (
                alpha_weights[idxs]
                * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            # Subnet forward pass (given task info in self.task)
            subnet = GetSubnet.apply(self.scores[self.task], self.sparsity)
        w = self.weight * subnet
        x = F.linear(x, w, self.bias)
        return x


    def __repr__(self):
        return f"MultitaskMaskLinear({self.shape()})"


# In[4]:


class BasisMultitaskMaskLinear(nn.Linear):
    def __init__(self, *args, num_tasks=1, num_seed_tasks_learned=1, start_at_optimal=True, sparsity=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_tasks >= num_seed_tasks_learned, "Seed tasks cannot be more than total tasks!"
        self.num_tasks = num_tasks
        self.num_seed_tasks_learned = num_seed_tasks_learned
        self.sparsity = sparsity
        self.scores = nn.ParameterList(
            [
                nn.Parameter(mask_init(self))
                for _ in range(num_tasks)
            ]
        )
        self.task = -1

        # Keep weights untrained
        self.weight.requires_grad = False
        for s in self.scores:
            s.requires_grad = False
        self.scores.requires_grad = False
        if start_at_optimal:
            self.basis_alphas = nn.ParameterList(
                [
                    nn.Parameter(torch.eye(self.num_seed_tasks_learned)[i])
                    for i in range(self.num_seed_tasks_learned)
                ]
                +
                [
                    nn.Parameter(torch.ones(self.num_seed_tasks_learned)/self.num_seed_tasks_learned)
                    for _ in range(self.num_seed_tasks_learned, self.num_tasks)
                ]
            )
        else:
            self.basis_alphas = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(self.num_seed_tasks_learned)/self.num_seed_tasks_learned)
                    for _ in range(self.num_tasks)
                ]
            )

        signed_constant(self)

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    GetSubnet.apply(self.scores[j], self.sparsity)
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def forward(self, x):
        if self.task < 0:
            raise NotImplemented("Need task identity at inference time.")
        else:
            # Subnet forward pass (given task info in self.task)
            subnet = self.stacked[: self.num_seed_tasks_learned][0]
            task_alpha = self.basis_alphas[self.task]
            w = self.weight * subnet * task_alpha[0]
            for i in range(1, self.num_seed_tasks_learned):
                subnet = self.stacked[: self.num_seed_tasks_learned][i]
                w += self.weight * subnet * task_alpha[i]
        x = F.linear(x, w, self.bias)
        return x


    def __repr__(self):
        return f"BasisMultitaskMaskLinear({self.shape()})"


ValidConvs = [
    MultitaskMaskLinear,
    BasisMultitaskMaskLinear,
]

def isoftype(m, cls_lst):
    return any([isinstance(m, c) for c in cls_lst])

# Utility functions
def set_model_task(model, task, verbose=True):
    for n, m in model.named_modules():
        if isoftype(m, ValidConvs):
            if verbose:
                print(f"=> Set task of {n} to {task}")
            m.task = task

def cache_masks(model):
    for n, m in model.named_modules():
        if isoftype(m, ValidConvs):
            print(f"=> Caching mask state for {n}")
            m.cache_masks()

def set_num_tasks_learned(model, num_tasks_learned):
    for n, m in model.named_modules():
        if isoftype(m, ValidConvs):
            print(f"=> Setting learned tasks of {n} to {num_tasks_learned}")
            m.num_tasks_learned = num_tasks_learned

def set_alphas(model, alphas, verbose=True):
    for n, m in model.named_modules():
        if isoftype(m, ValidConvs):
            if verbose:
                print(f"=> Setting alphas for {n}")
            m.alphas = alphas

# Multitask Model, a simple fully connected model in this case
class MultitaskFC(nn.Module):
    def __init__(self, hidden_size, num_tasks, sparsity):
        super().__init__()
        self.model = nn.Sequential(
            MultitaskMaskLinear(
                784,
                hidden_size,
                num_tasks=num_tasks,
                sparsity=sparsity,
                bias=False
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_size,
                hidden_size,
                num_tasks=num_tasks,
                sparsity=sparsity,
                bias=False
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_size,
                100,
                num_tasks=num_tasks,
                sparsity=sparsity,
                bias=False
            )
        )

    def forward(self, x):
        return self.model(x.flatten(1))


# Multitask Model, a simple fully connected model in this case
class BasisMultitaskFC(nn.Module):
    def __init__(self, hidden_size, num_tasks, num_seed_tasks_learned, sparsity, start_at_optimal=True):
        super().__init__()
        self.model = nn.Sequential(
            BasisMultitaskMaskLinear(
                784,
                hidden_size,
                num_tasks=num_tasks,
                num_seed_tasks_learned=num_seed_tasks_learned,
                start_at_optimal=start_at_optimal,
                sparsity=sparsity,
                bias=False
            ),
            nn.ReLU(),
            BasisMultitaskMaskLinear(
                hidden_size,
                hidden_size,
                num_tasks=num_tasks,
                num_seed_tasks_learned=num_seed_tasks_learned,
                start_at_optimal=start_at_optimal,
                sparsity=sparsity,
                bias=False
            ),
            nn.ReLU(),
            BasisMultitaskMaskLinear(
                hidden_size,
                100,
                num_tasks=num_tasks,
                num_seed_tasks_learned=num_seed_tasks_learned,
                start_at_optimal=start_at_optimal,
                sparsity=sparsity,
                bias=False
            )
        )

    def forward(self, x):
        return self.model(x.flatten(1))

class MNISTSplit:
    pass

class MNISTPerm:
    class permute(object):
        def __call__(self, tensor):
            out = tensor.flatten()
            out = out[self.perm]
            return out.view(1, 28, 28)

        def __repr__(self):
            return self.__class__.__name__

    def __init__(self, seed=0):
        super(MNISTPerm, self).__init__()

        data_root = "mnist"
        self.permuter = self.permute()
        self.seed = seed
        train_dataset = torchvision.datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    self.permuter,
                ]
            ),
        )

        # Data loading code
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                data_root,
                train=False,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        self.permuter,
                    ]
                ),
            ),
            batch_size=128,
            shuffle=False,
        )

    def update_task(self, i):
        np.random.seed(i + self.seed)
        self.permuter.__setattr__("perm", np.random.permutation(784))

    def unpermute(self):
        self.permuter.__setattr__("perm", np.arange(784))


def train(model, trainloader, optimizer, epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    num_correct = 0
    total_seen = 0
    for i, (batch, labels) in enumerate(trainloader):
        logits = model(batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            predictions = logits.argmax(dim=-1)
            num_correct += (predictions == labels).float().sum()
            total_seen += logits.size(0)
            print(
                f"e{epoch} {i+1}/{len(trainloader)}"
                f" => Loss {loss.item():0.4f}, "
                f"Acc@1 {(num_correct / total_seen):0.4f}")


@torch.no_grad()
def evaluate(model, val_loader, epoch):
    model.eval()
    num_correct = 0
    total_seen = 0
    for batch, labels in val_loader:
        logits = model(batch)
        predictions = logits.argmax(dim=-1)
        num_correct += (predictions == labels).float().sum()
        total_seen += logits.size(0)


    print(
        f"Val Perf after {epoch + 1} epochs "
        f"Acc@1 {(num_correct / total_seen):0.4f}",
    )
    return num_correct / total_seen

