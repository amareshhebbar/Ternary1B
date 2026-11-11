import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import model_dir, read_jsonl, write_json
from .layers import BitLinear, QATLinear, quant_linears, swap, unswap
from .quant import rtn_int2, ternary_absmean

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def core(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


def _ce_chunk(hs, w, y):
    logits = F.linear(hs, w).float()
    return torch.logsumexp(logits, -1) - logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)


def losses(cfg, student, teacher, ids):
    t = cfg["train"]
    chunk = t["logits_chunk"]
    body, head = core(student)
    hs = body(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    y = ids[:, 1:]
    kd_w = t["kd_weight"]
    ht = None
    if kd_w > 0:
        with torch.no_grad():
            tb, th = core(teacher)
            ht = tb(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    kd_sum, ce_sum = 0.0, 0.0
    for s in range(0, hs.size(1), chunk):
        h = hs[:, s : s + chunk]
        if kd_w > 0:
            kd_sum = kd_sum + checkpoint(_kd_chunk_tw, h, ht[:, s : s + chunk], head.weight, th.weight, t["kd_temp"], use_reentrant=False).sum()
        if kd_w < 1:
            ce_sum = ce_sum + checkpoint(_ce_chunk, h, head.weight, y[:, s : s + chunk], use_reentrant=False).sum()
    n = y.numel()
    kd = kd_sum / n if kd_w > 0 else torch.zeros((), device=ids.device)
    ce = ce_sum / n if kd_w < 1 else torch.zeros((), device=ids.device)
    return kd_w * kd + (1 - kd_w) * ce, kd, ce


def _kd_chunk_tw(hs, ht, ws, wt, temp):
    s = F.log_softmax(F.linear(hs, ws).float() / temp, dim=-1)
    with torch.no_grad():
        t = F.log_softmax(F.linear(ht, wt).float() / temp, dim=-1)
    return (t.exp() * (t - s)).sum(-1) * temp * temp


def load_teacher(cfg):
    from transformers import AutoModelForCausalLM

    if cfg["train"]["kd_weight"] <= 0:
        return None
    m = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def batches(cfg, steps, accum, micro):
    from .data import train_chunks

    arr = train_chunks(cfg)
    need = steps * accum * micro
    if arr.shape[0] < need:
        raise RuntimeError(f"have {arr.shape[0]} training chunks, need {need}; rerun the data step")
    for i in range(0, need, micro):
        yield torch.from_numpy(np.asarray(arr[i : i + micro]).astype(np.int64)).to(DEV)


def loop(cfg, name, student, teacher, steps, lr, warmup, on_step=None, param_lrs=None):
    from transformers import get_cosine_schedule_with_warmup

    t = cfg["train"]
    accum, micro = t["grad_accum"], t["micro_batch"]
    out = model_dir(cfg, name) / "training"
    out.mkdir(parents=True, exist_ok=True)
    groups = param_lrs or [{"params": [p for p in student.parameters() if p.requires_grad], "lr": lr}]
    params = [p for g in groups for p in g["params"]]
    total = sum(p.numel() for p in student.parameters())
    print(f"{name}: training {sum(p.numel() for p in params):,} of {total:,} parameters, {steps} steps x {accum * micro} sequences of {t['seq_len']}", flush=True)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=t["weight_decay"], fused=DEV == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, warmup, steps)
    log = open(out / "log.jsonl", "w")
    t0, step, acc, agg, toks = time.time(), 0, 0, {"loss": 0.0, "kd": 0.0, "ce": 0.0}, 0
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if on_step:
        on_step(0, steps)
    for ids in batches(cfg, steps, accum, micro):
        with torch.autocast(device_type="cuda" if DEV == "cuda" else "cpu", dtype=torch.bfloat16):
            loss, kd, ce = losses(cfg, student, teacher, ids)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non finite loss at step {step}")
        (loss / accum).backward()
        agg["loss"] += loss.item()
        agg["kd"] += float(kd)
        agg["ce"] += float(ce)
        toks += ids.numel()
        acc += 1
        if acc == accum:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if on_step:
                on_step(step, steps)
            el = time.time() - t0
            rec = {k: v / acc for k, v in agg.items()}
            rec.update({"step": step, "lr": sched.get_last_lr()[0], "grad_norm": gn, "tokens": toks, "tok_s": toks / el, "elapsed_min": el / 60, "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0})
            if on_step:
                rec["lambda"] = BitLinear.lam
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or step == steps:
                eta = el / step * (steps - step) / 60
                extra = f" lambda {BitLinear.lam:.2f}" if on_step else ""
                print(f"[{name}] step {step}/{steps} loss {rec['loss']:.4f} kd {rec['kd']:.4f} ce {rec['ce']:.4f} gn {gn:.2f} tok/s {rec['tok_s']:.0f} peak {rec['peak_gib']:.1f}GiB{extra} eta_min {eta:.1f}", flush=True)
            agg, acc = {"loss": 0.0, "kd": 0.0, "ce": 0.0}, 0
            if step >= steps:
                break
    log.close()
    plot(out / "log.jsonl", out / "loss.png", name)
    write_json(out / "summary.json", {"steps": step, "tokens": toks, "minutes": (time.time() - t0) / 60})


def plot(log_path, png, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_jsonl(log_path)
    if not rows:
        return
    keys = [k for k in ("loss", "kd", "ce", "lambda") if k in rows[0] and any(r[k] for r in rows)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.plot([r["step"] for r in rows], [r[k] for r in rows])
        ax.set_title(f"{name} {k}")
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def plain_model(cfg, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype, attn_implementation="sdpa").to(DEV)


def packed_source(cfg, name):
    from huggingface_hub import snapshot_download

    from .common import merged_dir, repo_id, wpath

    local = merged_dir(cfg, name)
    if (local / "packed" / "quant_config.json").exists():
        return local
    dest = wpath(cfg, "verify", f"{name}_packed")
    snapshot_download(repo_id(cfg, name), local_dir=str(dest), allow_patterns=["packed/*", "config.json"])
    return dest


def train_heal(cfg, name):
    from peft import LoraConfig, get_peft_model

    from .export import lora_tensors, save_model
    from .ptq import apply_qstate, quantize
    from .data import tokenizer

    h = cfg["heal"]
    tok = tokenizer(cfg)
    student = plain_model(cfg)
    if name == "gptq_lora":
        from .export import load_qstate

        qstate, _, _, _ = load_qstate(packed_source(cfg, "gptq"))
        apply_qstate(student, qstate)
        fmt = "int2 gptq"
    else:
        qstate = quantize(cfg, student, "hqq")
        fmt = "int2 hqq"
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    lc = LoraConfig(r=h["r"], lora_alpha=h["alpha"], lora_dropout=0.0, target_modules=list(set(n.split(".")[-1] for n in qstate)), task_type="CAUSAL_LM")
    student = get_peft_model(student, lc)
    teacher = load_teacher(cfg)
    loop(cfg, name, student, teacher, h["steps"], h["lr"], h["warmup_steps"])
    lora = lora_tensors(student)
    del student, teacher
    torch.cuda.empty_cache()
    fresh = plain_model(cfg)
    apply_qstate(fresh, qstate)
    meta = {"method": name, "format": fmt, "group": cfg["quant"]["group"], "act_bits": 0, "lora": {"r": h["r"], "alpha": h["alpha"], "scale": h["alpha"] / h["r"]}}
    save_model(cfg, name, fresh, tok, qstate, meta, lora)


def train_qat(cfg):
    from .data import tokenizer
    from .export import save_model

    q = cfg["qat"]
    g = cfg["quant"]["group"]
    tok = tokenizer(cfg)
    student = plain_model(cfg, torch.float32)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    QATLinear.group = g
    swap(student, QATLinear)
    teacher = load_teacher(cfg)
    loop(cfg, "qat", student, teacher, q["steps"], q["lr"], q["warmup_steps"])
    del teacher
    unswap(student)
    qstate = {n: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in rtn_int2(m.weight.data, g).items()} for n, m in quant_linears(student)}
    from .ptq import apply_qstate

    student = student.to(torch.bfloat16)
    apply_qstate(student, qstate)
    save_model(cfg, "qat", student, tok, qstate, {"method": "qat", "format": "int2 qat", "group": g, "act_bits": 0})


def train_bitnet(cfg):
    from .data import tokenizer
    from .export import save_model
    from .ptq import apply_qstate

    b = cfg["bitnet"]
    tok = tokenizer(cfg)
    student = plain_model(cfg, torch.float32)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    BitLinear.act_bits = 8
    swap(student, BitLinear)
    teacher = load_teacher(cfg)
    ramp = max(1, int(b["lambda_warmup_frac"] * b["steps"]))

    def on_step(step, steps):
        BitLinear.lam = min(1.0, step / ramp)

    loop(cfg, "bitnet", student, teacher, b["steps"], b["lr"], b["warmup_steps"], on_step=on_step)
    BitLinear.lam = 1.0
    del teacher
    unswap(student)
    qstate = {n: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ternary_absmean(m.weight.data).items()} for n, m in quant_linears(student)}
    student = student.to(torch.bfloat16)
    apply_qstate(student, qstate)
    save_model(cfg, "bitnet", student, tok, qstate, {"method": "bitnet", "format": "ternary 1.58 bit", "act_bits": 8})


def train_ptq(cfg, name):
    from .data import calib, tokenizer
    from .export import save_model
    from .ptq import quantize

    tok = tokenizer(cfg)
    model = plain_model(cfg)
    qstate = quantize(cfg, model, name, calib(cfg, tok))
    save_model(cfg, name, model, tok, qstate, {"method": name, "format": f"int2 {name}", "group": cfg["quant"]["group"], "act_bits": 0})


def train(cfg, name):
    torch.manual_seed(cfg["seed"])
    if name in ("gptq", "awq"):
        return train_ptq(cfg, name)
    if name in ("gptq_lora", "hqq_lora"):
        return train_heal(cfg, name)
    if name == "qat":
        return train_qat(cfg)
    if name == "bitnet":
        return train_bitnet(cfg)
    raise SystemExit(f"unknown model {name}")
