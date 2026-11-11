import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .common import BASELINES, banner, read_json, resolve_model, results_path, wpath, write_json, model_dir
from .layers import ActQuantLinear, swap

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = ["The capital of France is", "Photosynthesis is the process by which", "def fibonacci(n):"]

BASE_LABEL = {"bf16": "original bf16", "rtn2": "plain 2 bit rounding", "rtn_ternary": "plain ternary rounding"}


def load_eval(cfg, path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    meta = read_json(Path(path) / "packed" / "quant_config.json", {})
    if meta.get("act_bits"):
        swap(model, ActQuantLinear)
        print("  activation 8 bit quantization patched in, matching training", flush=True)
    return model


@torch.no_grad()
def wikitext_ppl(cfg, model, max_windows=None):
    from .data import wikitext_ids

    ids = wikitext_ids(cfg)
    L = cfg["eval"]["ppl_seq"]
    n = len(ids) // L
    if max_windows:
        n = min(n, max_windows)
    tot, cnt = 0.0, 0
    chunk = cfg["train"]["logits_chunk"]
    for i in range(n):
        x = torch.as_tensor(ids[i * L : (i + 1) * L].astype(np.int64), device=DEV)[None]
        h = model.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1]
        y = x[:, 1:]
        for s in range(0, h.size(1), chunk * 4):
            logits = model.lm_head(h[:, s : s + chunk * 4]).float()
            tot += (torch.logsumexp(logits, -1) - logits.gather(-1, y[:, s : s + chunk * 4, None]).squeeze(-1)).sum().item()
        cnt += y.numel()
    return math.exp(tot / max(1, cnt))


def lm_eval_tasks(cfg, model, tok, quick=False):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    e = cfg["eval"]
    tasks = e["quick_tasks"] if quick else e["tasks"]
    if not tasks:
        return {}
    limit = e["quick_limit"] if quick else e.get("limit")
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=e["batch_size"], max_length=e["ppl_seq"])
    res = simple_evaluate(model=lm, tasks=tasks, limit=limit, num_fewshot=0, bootstrap_iters=0, log_samples=False)["results"]
    out = {}
    for t in tasks:
        r = res.get(t, {})
        out[t] = r.get("acc_norm,none", r.get("acc,none"))
    return out


@torch.no_grad()
def samples(cfg, model, tok):
    out = []
    for p in PROMPTS:
        x = tok(p, return_tensors="pt").input_ids.to(DEV)
        g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        out.append({"prompt": p, "completion": tok.decode(g[0, x.shape[1] :], skip_special_tokens=True)})
    return out


def size_estimate(model, qstate):
    from .quant import bits_per_weight

    nq = sum(q["codes"].numel() for q in qstate.values()) if qstate else 0
    total = sum(p.numel() for p in model.parameters())
    rest = total - nq
    bits = bits_per_weight(qstate) if qstate else 16.0
    b = nq * bits / 8 + rest * 2
    return {"packed_bytes": b, "packed_mb": b / 2**20, "bits_per_quantized_weight": bits, "bf16_bytes": total * 2, "compression": total * 2 / b, "estimated": True}


def run_suite(cfg, model, tok, tag, mode, size):
    t0 = time.time()
    quick = mode == "quick"
    banner(f"[{tag} | {mode}] WikiText2 perplexity")
    ppl = wikitext_ppl(cfg, model, cfg["eval"]["quick_windows"] if quick else None)
    print(f"  wikitext2 ppl {ppl:.3f}", flush=True)
    banner(f"[{tag} | {mode}] benchmarks")
    tasks = lm_eval_tasks(cfg, model, tok, quick)
    print(f"  {tasks}", flush=True)
    out = {"tag": tag, "mode": mode, "wikitext_ppl": ppl, "tasks": tasks, "size": size, "samples": samples(cfg, model, tok) if not quick else []}
    vals = [v for v in tasks.values() if v is not None]
    out["summary"] = {"wikitext_ppl": ppl, "task_mean": sum(vals) / len(vals) if vals else None, "size_mb": size.get("packed_mb"), "bits": size.get("bits_per_quantized_weight"), "compression": size.get("compression")}
    out["minutes"] = (time.time() - t0) / 60
    write_json(results_path(cfg, tag, mode), out)
    banner(f"[{tag}] summary {json.dumps(out['summary'])}")
    return out


def run_baseline(cfg):
    from .data import tokenizer
    from .ptq import quantize
    from .train import plain_model

    tok = tokenizer(cfg)
    for v in BASELINES:
        tag = f"base_{v}"
        if read_json(results_path(cfg, tag, "full")):
            print(f"skip {tag}, already evaluated", flush=True)
            continue
        model = plain_model(cfg).eval()
        qstate = {} if v == "bf16" else quantize(cfg, model, v)
        size = size_estimate(model, qstate)
        if v == "bf16":
            run_suite(cfg, model, tok, tag, "quick", size)
        run_suite(cfg, model, tok, tag, "full", size)
        del model
        torch.cuda.empty_cache()


def eval_model(cfg, name, mode="full", path=None):
    from transformers import AutoTokenizer

    path = path or resolve_model(cfg, name)
    size = read_json(model_dir(cfg, name) / "size.json") or read_json(Path(path) / "size.json") or {}
    model = load_eval(cfg, path)
    return run_suite(cfg, model, AutoTokenizer.from_pretrained(path), name, mode, size)
