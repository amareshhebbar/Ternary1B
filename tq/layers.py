import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import group_view, minmax_params

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def ste(x, xq):
    return x + (xq - x).detach()


def fq_int2(w, g):
    wg = group_view(w, g)
    s, z = minmax_params(wg.detach())
    q = torch.clamp(torch.round(wg / s[..., None]) + z[..., None], 0, 3)
    return ((q - z[..., None]) * s[..., None]).reshape(w.shape)


def fq_ternary(w):
    s = w.detach().abs().mean().clamp(min=1e-8)
    return torch.clamp(torch.round(w / s), -1, 1) * s


def fq_act8(x):
    s = 127.0 / x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return torch.clamp(torch.round(x * s), -128, 127) / s


class QATLinear(nn.Linear):
    group = 64

    def forward(self, x):
        return F.linear(x, ste(self.weight, fq_int2(self.weight, self.group)), self.bias)


class BitLinear(nn.Linear):
    lam = 1.0
    act_bits = 8

    def forward(self, x):
        lam = BitLinear.lam
        w = self.weight
        wq = ste(w, fq_ternary(w))
        xq = ste(x, fq_act8(x)) if self.act_bits else x
        if lam < 1.0:
            wq = (1 - lam) * w + lam * wq
            xq = (1 - lam) * x + lam * xq
        return F.linear(xq, wq, self.bias)


class ActQuantLinear(nn.Linear):
    def forward(self, x):
        return F.linear(fq_act8(x), self.weight, self.bias)


def quant_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGETS]


def swap(model, cls):
    for _, m in quant_linears(model):
        m.__class__ = cls
    return model


def unswap(model):
    for _, m in quant_linears(model):
        m.__class__ = nn.Linear
    return model
