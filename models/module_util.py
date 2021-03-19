import torch
import torch.nn as nn
import torch.autograd as autograd

import math 

def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def pspinit(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def bn_mask_init(module):
    return torch.ones(module.num_features)


def bn_mask_initv2(module):
    return torch.zeros(module.num_features)


def rank_one_init(module):
    scores = torch.Tensor(module.weight.size(0))
    nn.init.uniform_(scores, a=-1, b=1)
    scores = scores.sign().float()
    return scores


def rank_one_initv2(module):
    scores = torch.Tensor(module.weight.size(1))
    nn.init.uniform_(scores, a=-1, b=1)
    scores = scores.sign().float()
    return scores


def mask_initv2(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores[0]


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

class GetWeightsNorm(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        # Get the supermask by sorting the scores and using the top k%
        ctx.save_for_backward(scores)
        assert (scores >= 0).all(), "Scores must be greater than 0 in WeightNormalized subnet"

        out = scores.clone()

        # flat_out and out access the same memory.
        flat_out = out.flatten()

        # Weight between 0,1
        flat_out = flat_out/flat_out.abs().max()
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        scores, = ctx.saved_tensors

        ret_g = g/scores.flatten().abs().max()
        return ret_g, None

class GetSignedWeightsNorm(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        # Get the supermask by sorting the scores and using the top k%
        ctx.save_for_backward(scores)
        out = scores.clone()

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        # Weight between -1,1
        flat_out = flat_out/flat_out.abs().max()

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        scores, = ctx.saved_tensors
        ret_g = g/scores.flatten().abs().max()
        return ret_g, None


class GetSubnetSoft(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        assert (scores >= 0).all(), "Scores must be greater than 0 in Soft (Hybrid) subnet"
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        # Weight between 0,1
        max_abs = flat_out.abs().max()
        ctx.save_for_backward(max_abs)
        flat_out = flat_out/max_abs

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        max_abs, = ctx.saved_tensors
        ret_g = g/max_abs
        return ret_g, None

class GetSubentSignedSoft(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        # Weight between -1,1
        max_abs = flat_out.abs().max()
        ctx.save_for_backward(max_abs)
        flat_out = flat_out/max_abs

        return out

    @staticmethod
    def backward(ctx, g):
        max_abs, = ctx.saved_tensors
        ret_g = g/max_abs
        return ret_g, None

class GetSignedSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        ctx.save_for_backward(scores)
        out = scores.clone()
        _, idx = scores.abs().flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out * scores.sign()

    @staticmethod
    def backward(ctx, g):
        scores, = ctx.saved_tensors

        # send the gradient g straight-through on the backward pass.
        return g , None


def get_subnet(scores, k):
    out = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out

def get_subnet_signed(scores, k):
    out = scores.clone()
    _, idx = scores.abs().flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out * scores.sign()


class GetSubnetFast(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        return g, None


def get_subnet_fast(scores, a=0):
    return (scores >= a).float()


def kaiming_normal(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_normal_(scores, nonlinearity="relu")
    return scores
