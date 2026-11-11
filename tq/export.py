import json
import shutil
from pathlib import Path

import torch

from .common import banner, model_dir, write_json
from .quant import bits_per_weight, dequant, load_packed, packed_size, save_packed

RUNTIME_TAIL = '''

def load_model(repo_dir, device="cpu", dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoModelForCausalLM

    repo_dir = Path(repo_dir)
    qstate, rest, lora, meta = load_packed(repo_dir / "packed", device=device)
    conf = AutoConfig.from_pretrained(repo_dir)
    model = AutoModelForCausalLM.from_config(conf, torch_dtype=dtype)
    sd = {k: v.to(dtype) for k, v in rest.items()}
    for name, q in qstate.items():
        w = dequant(q)
        if lora:
            a, b = lora[f"{name}.lora_A"].float(), lora[f"{name}.lora_B"].float()
            w = w + meta["lora"]["scale"] * (b @ a)
        sd[f"{name}.weight"] = w.to(dtype)
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    if meta.get("act_bits"):
        patch_act_quant(model)
    return model.to(device).eval()


def patch_act_quant(model):
    import torch.nn as nn
    import torch.nn.functional as F

    def fq(x):
        s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        return torch.clamp(torch.round(x * s), -128, 127) / s

    class ActQuantLinear(nn.Linear):
        def forward(self, x):
            return F.linear(fq(x), self.weight, self.bias)

    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and n.split(".")[-1] in names:
            m.__class__ = ActQuantLinear
    return model
'''


def runtime_source():
    return (Path(__file__).parent / "quant.py").read_text() + RUNTIME_TAIL


def rest_state(model, qstate):
    skip = {f"{n}.weight" for n in qstate}
    tied = getattr(model.config, "tie_word_embeddings", False)
    out = {}
    for k, v in model.state_dict().items():
        if k in skip or (tied and k == "lm_head.weight"):
            continue
        out[k] = v.detach()
    return out


def lora_tensors(peft_model):
    out = {}
    for n, m in peft_model.named_modules():
        if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
            base = n.replace("base_model.model.", "", 1)
            out[f"{base}.lora_A"] = m.lora_A["default"].weight.detach().float().cpu()
            out[f"{base}.lora_B"] = m.lora_B["default"].weight.detach().float().cpu()
    return out


def save_model(cfg, name, model, tok, qstate, meta, lora=None):
    root = model_dir(cfg, name) / "merged"
    if root.exists():
        shutil.rmtree(root)
    model = model.to(torch.bfloat16)
    model.config.use_cache = True
    rest = rest_state(model, qstate)
    if lora:
        mods = dict(model.named_modules())
        for qn in qstate:
            a, b = lora[f"{qn}.lora_A"].float(), lora[f"{qn}.lora_B"].float()
            w = dequant(qstate[qn]).float() + meta["lora"]["scale"] * (b @ a)
            mods[qn].weight.data.copy_(w.to(mods[qn].weight.dtype).to(mods[qn].weight.device))
    model.save_pretrained(str(root), safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(str(root))
    size = save_packed(root / "packed", qstate, rest, meta, lora)
    (root / "tq_runtime.py").write_text(runtime_source())
    bf16 = sum(f.stat().st_size for f in root.glob("*.safetensors"))
    info = {
        "packed_bytes": size,
        "packed_mb": size / 2**20,
        "bf16_bytes": bf16,
        "compression": bf16 / max(1, size),
        "bits_per_quantized_weight": bits_per_weight(qstate),
        "quantized_params": sum(q["codes"].numel() for q in qstate.values()),
        "format": meta.get("format"),
    }
    write_json(model_dir(cfg, name) / "size.json", info)
    write_json(root / "size.json", info)
    banner(f"saved {name}: packed {info['packed_mb']:.1f} MB ({info['bits_per_quantized_weight']:.2f} bits per quantized weight), bf16 copy {bf16 / 2**20:.0f} MB")
    return info


def load_qstate(path):
    q, rest, lora, meta = load_packed(Path(path) / "packed")
    return q, rest, lora, meta


def compare_unpacked(root, model_path=None):
    from safetensors.torch import load_file

    root = Path(root)
    qstate, rest, lora, meta = load_packed(root / "packed")
    weights = {}
    for f in sorted(root.glob("*.safetensors")):
        weights.update(load_file(str(f)))
    worst = 0.0
    for name, q in qstate.items():
        w = dequant(q)
        if lora:
            w = w + meta["lora"]["scale"] * (lora[f"{name}.lora_B"].float() @ lora[f"{name}.lora_A"].float())
        ref = weights[f"{name}.weight"].float()
        rel = ((w - ref).abs().max() / ref.abs().max().clamp(min=1e-8)).item()
        worst = max(worst, rel)
    return worst, meta
