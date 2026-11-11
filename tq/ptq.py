import time

import torch

from .common import banner
from .layers import quant_linears
from .quant import awq_int2, canon, dequant, gptq_int2, hqq_int2, rtn_int2, ternary_absmean

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Stop(Exception):
    pass


def apply_qstate(model, qstate):
    mods = dict(model.named_modules())
    for name, q in qstate.items():
        m = mods[name]
        canon(q)
        m.weight.data.copy_(dequant(q).to(m.weight.dtype).to(m.weight.device))
    return model


def weight_only(cfg, model, method):
    g = cfg["quant"]["group"]
    qstate = {}
    for name, m in quant_linears(model):
        w = m.weight.data.to(DEV)
        if method == "rtn2":
            q = rtn_int2(w, g)
        elif method == "rtn_ternary":
            q = ternary_absmean(w)
        elif method == "hqq":
            q = hqq_int2(w, g, iters=cfg["quant"]["hqq_iters"])
        else:
            raise ValueError(method)
        qstate[name] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in q.items()}
    apply_qstate(model, qstate)
    return qstate


@torch.no_grad()
def capture(model, calib, bs):
    layers = model.model.layers
    xs, kw = [], {}

    def pre(mod, args, kwargs):
        xs.append((args[0] if args else kwargs["hidden_states"]).detach())
        if not kw:
            kw.update({k: v for k, v in kwargs.items() if k not in ("hidden_states", "past_key_values", "past_key_value", "use_cache")})
        raise _Stop

    h = layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    for i in range(0, len(calib), bs):
        try:
            model(torch.as_tensor(calib[i : i + bs].astype("int64"), device=DEV), use_cache=False)
        except _Stop:
            pass
    h.remove()
    return torch.cat(xs), kw


def run_layer(layer, x, kw, bs):
    outs = []
    for i in range(0, x.shape[0], bs):
        o = layer(x[i : i + bs], **kw)
        outs.append(o[0] if isinstance(o, tuple) else o)
    return torch.cat(outs)


@torch.no_grad()
def sequential(cfg, model, calib, method):
    c = cfg["quant"]
    g, bs, keep = c["group"], c["calib_batch"], c["awq_rows"]
    model.to(DEV).eval()
    x, kw = capture(model, calib, bs)
    banner(f"{method}: {len(model.model.layers)} layers, calibration {tuple(x.shape)}")
    qstate = {}
    gen = torch.Generator(device=DEV).manual_seed(cfg["seed"])
    t0 = time.time()
    for li, layer in enumerate(model.model.layers):
        lin = [(n, m) for n, m in quant_linears(layer)]
        stats = {}
        hooks = []
        for n, m in lin:
            d = m.in_features
            stats[n] = {"H": torch.zeros(d, d, device=DEV), "n": 0, "abs": torch.zeros(d, device=DEV), "rows": []}

            def hook(mod, inp, out, n=n):
                a = inp[0].reshape(-1, inp[0].shape[-1]).float()
                s = stats[n]
                if method == "gptq":
                    s["H"].addmm_(a.t(), a)
                s["n"] += a.shape[0]
                s["abs"] += a.abs().sum(0)
                if method == "awq":
                    k = max(1, keep // max(1, x.shape[0] // bs))
                    idx = torch.randint(0, a.shape[0], (min(k, a.shape[0]),), device=DEV, generator=gen)
                    s["rows"].append(a[idx])

            hooks.append(m.register_forward_hook(hook))
        run_layer(layer, x, kw, bs)
        for h in hooks:
            h.remove()
        for n, m in lin:
            s = stats[n]
            w = m.weight.data
            if method == "gptq":
                q = gptq_int2(w, s["H"] / s["n"], g, damp=c["gptq_damp"])
            else:
                q = awq_int2(w, torch.cat(s["rows"])[:keep], s["abs"] / s["n"], g, grid=c["awq_grid"])
            m.weight.data.copy_(dequant(canon(q)).to(w.dtype))
            full = f"model.layers.{li}.{n}"
            qstate[full] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in q.items()}
        del stats
        x = run_layer(layer, x, kw, bs)
        print(f"  layer {li + 1}/{len(model.model.layers)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    return qstate


def quantize(cfg, model, method, calib=None):
    if method in ("gptq", "awq"):
        return sequential(cfg, model, calib, method)
    return weight_only(cfg, model, method)
