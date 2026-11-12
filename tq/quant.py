import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

QMAX = {"int2": 3}


def group_view(w, g):
    out, inp = w.shape
    if inp % g:
        raise ValueError(f"in_features {inp} not divisible by group {g}")
    return w.reshape(out, inp // g, g)


def minmax_params(wg, bits=2):
    qmax = 2**bits - 1
    lo = wg.amin(-1).clamp(max=0)
    hi = wg.amax(-1).clamp(min=0)
    scale = ((hi - lo) / qmax).clamp(min=1e-8)
    zero = torch.round(-lo / scale).clamp(0, qmax)
    return scale, zero


def q_int(wg, scale, zero, bits=2):
    return torch.clamp(torch.round(wg / scale[..., None]) + zero[..., None], 0, 2**bits - 1)


def deq_int(codes, scale, zero):
    return (codes - zero[..., None]) * scale[..., None]


def rtn_int2(w, g):
    wg = group_view(w.float(), g)
    scale, zero = minmax_params(wg)
    codes = q_int(wg, scale, zero)
    return {"format": "int2", "codes": codes.reshape(w.shape).to(torch.uint8), "scales": scale, "zeros": zero}


def ternary_absmean(w):
    wf = w.float()
    scale = wf.abs().mean().clamp(min=1e-8)
    codes = torch.clamp(torch.round(wf / scale), -1, 1) + 1
    return {"format": "ternary", "codes": codes.to(torch.uint8), "scales": scale.reshape(1)}


def canon(q):
    if q["format"] == "int2":
        q["scales"] = q["scales"].half().float()
        q["zeros"] = q["zeros"].half().float()
        if q.get("awq") is not None:
            q["awq"] = q["awq"].half().float()
    return q


def dequant(q, shape=None):
    if q["format"] == "ternary":
        return (q["codes"].float() - 1.0) * q["scales"].float()
    codes = q["codes"].float()
    out, inp = codes.shape
    ng = q["scales"].shape[-1]
    w = deq_int(codes.reshape(out, ng, inp // ng), q["scales"].float(), q["zeros"].float()).reshape(out, inp)
    if q.get("awq") is not None:
        w = w / q["awq"].float()[None, :]
    return w


def hqq_int2(w, g, iters=20, lp=0.7, beta=10.0, kappa=1.01):
    wf = group_view(w.float(), g).reshape(-1, g)
    lo, hi = wf.amin(-1, keepdim=True), wf.amax(-1, keepdim=True)
    inv = (3.0 / (hi - lo).clamp(min=1e-8))
    zero = -lo * inv
    best_err, best = float("inf"), None
    for _ in range(iters):
        wq = torch.clamp(torch.round(wf * inv + zero), 0, 3)
        wr = (wq - zero) / inv
        e = wf - wr
        we = torch.sign(e) * torch.relu(e.abs() - (1.0 / beta) * e.abs().clamp(min=1e-12).pow(lp - 1))
        zero = (wq - (wf - we) * inv).mean(-1, keepdim=True)
        beta *= kappa
        err = (wf - (torch.clamp(torch.round(wf * inv + zero), 0, 3) - zero) / inv).abs().mean().item()
        if err < best_err:
            best_err, best = err, zero.clone()
        else:
            break
    zero = best
    codes = torch.clamp(torch.round(wf * inv + zero), 0, 3)
    out, inp = w.shape
    ng = inp // g
    return {"format": "int2", "codes": codes.reshape(out, inp).to(torch.uint8), "scales": (1.0 / inv).reshape(out, ng), "zeros": zero.reshape(out, ng)}


def gptq_int2(w, H, g, blocksize=128, damp=0.01, act_order=True):
    W = w.float().clone()
    out, cols = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    scales, zeros = minmax_params(group_view(W, g))
    perm = torch.argsort(torch.diag(H), descending=True) if act_order else torch.arange(cols, device=W.device)
    inv = torch.argsort(perm)
    W = W[:, perm]
    H = H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    gidx = perm // g
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            gi = gidx[i1 + i]
            s, z = scales[:, gi], zeros[:, gi]
            wcol = W1[:, i]
            q = torch.clamp(torch.round(wcol / s) + z, 0, 3)
            Q[:, i1 + i] = q
            err = (wcol - (q - z) * s) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return {"format": "int2", "codes": Q[:, inv].to(torch.uint8), "scales": scales, "zeros": zeros}


def awq_int2(w, X, act_mean, g, grid=20):
    wf = w.float()
    ref = X @ wf.t()
    best_err, best = float("inf"), None
    am = act_mean.float().clamp(min=1e-5)
    for k in range(grid):
        a = k / grid
        s = am.pow(a)
        s = s / (s.max() * s.min()).sqrt()
        q = rtn_int2(wf * s[None, :], g)
        q["awq"] = s
        err = (X @ dequant(q).t() - ref).pow(2).mean().item()
        if err < best_err:
            best_err, best = err, q
    return best


def pack_int2(codes):
    c = codes.reshape(-1).to(torch.uint8)
    pad = (-c.numel()) % 4
    if pad:
        c = torch.cat([c, c.new_zeros(pad)])
    c = c.reshape(-1, 4)
    return c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)


def unpack_int2(packed, n):
    p = packed.to(torch.uint8)
    c = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=1).reshape(-1)
    return c[:n]


POW3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def pack_trits(codes):
    c = codes.reshape(-1).to(torch.int32)
    pad = (-c.numel()) % 5
    if pad:
        c = torch.cat([c, c.new_zeros(pad)])
    return (c.reshape(-1, 5) * POW3.to(c.device)).sum(-1).to(torch.uint8)


def unpack_trits(packed, n):
    p = packed.to(torch.int32)
    digits = [(p // int(v)) % 3 for v in POW3]
    return torch.stack(digits, dim=1).reshape(-1)[:n].to(torch.uint8)


def save_packed(out_dir, qstate, rest, meta, lora=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, layout = {}, {}
    for name, q in qstate.items():
        shape = list(q["codes"].shape)
        if q["format"] == "ternary":
            tensors[f"{name}.codes"] = pack_trits(q["codes"].cpu())
        else:
            tensors[f"{name}.codes"] = pack_int2(q["codes"].cpu())
            tensors[f"{name}.zeros"] = q["zeros"].half().cpu().contiguous()
        tensors[f"{name}.scales"] = q["scales"].float().cpu().contiguous() if q["format"] == "ternary" else q["scales"].half().cpu().contiguous()
        if q.get("awq") is not None:
            tensors[f"{name}.awq"] = q["awq"].half().cpu().contiguous()
        layout[name] = {"format": q["format"], "shape": shape}
    save_file(tensors, str(out_dir / "quant.safetensors"))
    save_file({k: v.half().cpu().contiguous() for k, v in rest.items()}, str(out_dir / "rest.safetensors"))
    if lora:
        save_file({k: v.half().cpu().contiguous() for k, v in lora.items()}, str(out_dir / "lora.safetensors"))
    (out_dir / "quant_config.json").write_text(json.dumps({**meta, "layout": layout}, indent=1))
    return packed_size(out_dir)


def load_packed(pdir, device="cpu"):
    pdir = Path(pdir)
    meta = json.loads((pdir / "quant_config.json").read_text())
    t = load_file(str(pdir / "quant.safetensors"), device=device)
    qstate = {}
    for name, info in meta["layout"].items():
        n = math.prod(info["shape"])
        if info["format"] == "ternary":
            codes = unpack_trits(t[f"{name}.codes"], n).reshape(info["shape"])
            q = {"format": "ternary", "codes": codes, "scales": t[f"{name}.scales"]}
        else:
            codes = unpack_int2(t[f"{name}.codes"], n).reshape(info["shape"])
            q = {"format": "int2", "codes": codes, "scales": t[f"{name}.scales"], "zeros": t[f"{name}.zeros"]}
            if f"{name}.awq" in t:
                q["awq"] = t[f"{name}.awq"]
        qstate[name] = q
    rest = load_file(str(pdir / "rest.safetensors"), device=device)
    lora = load_file(str(pdir / "lora.safetensors"), device=device) if (pdir / "lora.safetensors").exists() else None
    return qstate, rest, lora, meta


def packed_size(pdir):
    pdir = Path(pdir)
    files = [pdir / f for f in ("quant.safetensors", "rest.safetensors", "lora.safetensors") if (pdir / f).exists()]
    return sum(f.stat().st_size for f in files)


def bits_per_weight(qstate):
    n = sum(q["codes"].numel() for q in qstate.values())
    bits = 0.0
    for q in qstate.values():
        k = q["codes"].numel()
        if q["format"] == "ternary":
            bits += k * 8 / 5 + 32
        else:
            bits += k * 2 + q["scales"].numel() * 32
            if q.get("awq") is not None:
                bits += q["awq"].numel() * 16
    return bits / max(1, n)
