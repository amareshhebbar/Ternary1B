import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid

PARTS = ["sys", "hub", "data", "quant", "train", "cpu"]
PACKAGES = ["torch", "transformers", "peft", "accelerate", "datasets", "huggingface_hub", "lm_eval", "safetensors", "yaml", "numpy", "matplotlib", "hf_transfer"]


def ok(msg):
    print(f"  [PASS] {msg}", flush=True)


def warn(msg):
    print(f"  [WARN] {msg}", flush=True)


def fail(msg):
    print(f"  [FAIL] {msg}", flush=True)
    raise SystemExit(1)


def check_sys(cfg):
    import torch

    print(f"  python {platform.python_version()} at {sys.executable}, config {cfg['config_path']}", flush=True)
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            ok(f"import {name} {getattr(m, '__version__', '')}")
        except Exception as e:
            fail(f"import {name}: {e!r}")
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(), flush=True)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 2**30
    ok(f"GPU 0: {p.name}, {vram:.1f} GiB, compute capability {p.major}.{p.minor}, torch CUDA {torch.version.cuda}")
    if p.major < 8 or not torch.cuda.is_bf16_supported():
        fail("bf16 needs an Ampere or newer GPU (A100, A6000, L40S, H100, RTX 3090/4090)")
    if vram < 44:
        warn(f"{vram:.0f} GiB VRAM. QAT and BitNet train the full model in fp32 and target 48 GiB. Lower train.micro_batch if the train check runs out of memory.")
    free = shutil.disk_usage(cfg["paths"]["work"]).free / 2**30
    print(f"  work dir {cfg['paths']['work']}, free disk {free:.0f} GiB", flush=True)
    if free < 60:
        warn("under 60 GiB free. Six models, a bf16 copy of each, GGUF files and caches need room.")
    else:
        ok("disk space")


def check_hub(cfg):
    from huggingface_hub import HfApi

    from .hub import login

    try:
        login()
    except Exception as e:
        fail(f"login: {e}")
    api = HfApi()
    me = api.whoami()
    name = me.get("name")
    ok(f"logged in as {name}")
    if name != cfg["hf_user"] and cfg["hf_user"] not in [o.get("name") for o in me.get("orgs", [])]:
        fail(f"token belongs to {name} but config hf_user is {cfg['hf_user']}. Run: export HF_USER={name}")
    if me.get("auth", {}).get("accessToken", {}).get("role") == "read":
        fail("token is read only. Create a write token at huggingface.co/settings/tokens")
    tag = uuid.uuid4().hex[:6]
    repo = f"{cfg['hf_user']}/ternary1b_write_test_{tag}"
    try:
        api.create_repo(repo, private=True)
        api.upload_file(path_or_fileobj=b"ok", path_in_repo="probe.txt", repo_id=repo, commit_message="write probe")
        ok(f"create repo and upload file ({repo})")
        c = api.create_collection(title=f"ternary1b write test {tag}", namespace=cfg["hf_user"], private=True)
        api.add_collection_item(c.slug, repo, "model", note="probe")
        api.delete_collection(c.slug)
        ok("create, fill and delete a collection")
    finally:
        try:
            api.delete_repo(repo)
            ok("delete test repo")
        except Exception as e:
            warn(f"could not delete {repo}: {e}")


def check_data(cfg):
    from transformers import AutoConfig, AutoTokenizer

    from .data import stream_texts, wikitext_ids

    conf = AutoConfig.from_pretrained(cfg["base_model"])
    ok(f"base {cfg['base_model']}: {conf.num_hidden_layers} layers, hidden {conf.hidden_size}, ffn {conf.intermediate_size}, tied embeddings {conf.tie_word_embeddings}")
    for d in (conf.hidden_size, conf.intermediate_size):
        if d % cfg["quant"]["group"]:
            fail(f"dimension {d} not divisible by quant.group {cfg['quant']['group']}")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if len(tok) > 65535:
        fail("vocab too large for uint16 token storage")
    ok(f"tokenizer vocab {len(tok)}, eos {tok.eos_token}")
    t = time.time()
    text = next(iter(stream_texts()))
    ok(f"FineWeb Edu streams ({len(text)} characters in the first document, {time.time() - t:.0f}s)")
    ok(f"WikiText2 test: {len(wikitext_ids(cfg, tok)):,} tokens")


def check_quant(cfg):
    import torch

    from .quant import awq_int2, dequant, gptq_int2, hqq_int2, load_packed, pack_int2, pack_trits, rtn_int2, save_packed, ternary_absmean, unpack_int2, unpack_trits
    from .common import wpath

    torch.manual_seed(0)
    dev = "cuda"
    g = cfg["quant"]["group"]
    c = torch.randint(0, 4, (1000,), dtype=torch.uint8)
    t = torch.randint(0, 3, (1003,), dtype=torch.uint8)
    if not torch.equal(unpack_int2(pack_int2(c), 1000), c) or not torch.equal(unpack_trits(pack_trits(t), 1003), t):
        fail("pack and unpack round trip broke")
    ok("2 bit and ternary packing round trip exactly (4 and 5 weights per byte)")
    X = torch.randn(4096, 512, device=dev) @ torch.randn(512, 512, device=dev) * 0.3
    X[:, :8] *= 20
    W = torch.randn(256, 512, device=dev) * 0.05
    ref = X @ W.t()

    def err(q):
        return ((X @ dequant(q).t() - ref).pow(2).mean() / ref.pow(2).mean()).item()

    e = {"rtn": err(rtn_int2(W, g)), "hqq": err(hqq_int2(W, g)), "gptq": err(gptq_int2(W, X.t() @ X / X.shape[0], g)), "awq": err(awq_int2(W, X[:2048], X.abs().mean(0), g))}
    print("  relative output error: " + ", ".join(f"{k} {v:.4f}" for k, v in e.items()), flush=True)
    if not (e["gptq"] < e["rtn"] and e["awq"] <= e["rtn"] + 1e-9):
        fail("GPTQ or AWQ did not beat plain rounding on a test layer")
    ok("GPTQ and AWQ beat plain rounding on a test layer with outlier inputs")
    q2, q3 = rtn_int2(W, g), ternary_absmean(W)
    d = wpath(cfg, "verify", "_packtest")
    save_packed(d, {"a": q2, "b": q3}, {"n": torch.ones(4)}, {"format": "test"})
    back, _, _, _ = load_packed(d)
    if not (torch.equal(dequant(back["a"]).cpu(), dequant(q2).cpu().half().float()) or torch.allclose(dequant(back["a"]).cpu(), dequant(q2).cpu(), atol=1e-3, rtol=1e-3)):
        fail("packed 2 bit layer does not rebuild")
    if not torch.allclose(dequant(back["b"]).cpu(), dequant(q3).cpu(), atol=1e-6):
        fail("packed ternary layer does not rebuild")
    shutil.rmtree(d, ignore_errors=True)
    ok("save and load of packed files rebuilds the weights")


def check_train(cfg):
    import torch

    from .layers import BitLinear, QATLinear, swap
    from .train import load_teacher, losses, plain_model

    t = cfg["train"]
    B, L = t["micro_batch"], t["seq_len"]
    est = {}
    teacher = load_teacher(cfg)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    for name in ["qat", "bitnet", "heal"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if name == "heal":
            from peft import LoraConfig, get_peft_model

            m = plain_model(cfg)
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
            m = get_peft_model(m, LoraConfig(r=cfg["heal"]["r"], lora_alpha=cfg["heal"]["alpha"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
            steps = cfg["heal"]["steps"]
        else:
            m = plain_model(cfg, torch.float32)
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            QATLinear.group = cfg["quant"]["group"]
            BitLinear.lam = 1.0
            swap(m, QATLinear if name == "qat" else BitLinear)
            steps = cfg[name]["steps"]
        m.config.use_cache = False
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5, fused=True)
        ids = torch.randint(100, 40000, (B, L), device="cuda")
        times, vals = [], []
        for _ in range(3):
            torch.cuda.synchronize()
            s = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, kd, ce = losses(cfg, m, teacher, ids)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - s)
            vals.append(loss.item())
        tok_s = B * L / (sum(times[1:]) / 2)
        hours = steps * t["grad_accum"] * B * L / tok_s / 3600
        est[name] = hours
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {name}: batch {B} x {L}, losses {[round(v, 3) for v in vals]}, {tok_s:.0f} tok/s, peak {peak:.1f} of {total:.0f} GiB, about {hours:.1f} h for {steps} steps", flush=True)
        if not all(v == v for v in vals):
            fail(f"{name}: loss is NaN")
        if peak > total * 0.95:
            warn(f"{name}: memory is close to the limit; lower train.micro_batch and raise train.grad_accum")
        ok(f"{name}: training step with distillation fits")
        del m, opt
    print(f"  estimated training time: QAT {est['qat']:.1f} h, BitNet {est['bitnet']:.1f} h, each LoRA repair {est['heal']:.1f} h; total about {est['qat'] + est['bitnet'] + 2 * est['heal']:.1f} h plus quantization, evals and uploads", flush=True)


def check_cpu(cfg):
    c = cfg["cpu_bench"]
    if not c.get("enabled"):
        ok("cpu_bench disabled, nothing to check")
        return
    for tool in ["git", "gcc", "g++", "make"]:
        if not shutil.which(tool):
            warn(f"{tool} not found; the CPU benchmark will try to continue but may report n/a. Install with apt-get install -y build-essential git")
        else:
            ok(f"{tool} found")
    print(f"  {os.cpu_count()} CPU threads available, benchmark uses {c.get('threads') or max(1, (os.cpu_count() or 4) // 2)}", flush=True)
    ok("CPU benchmark prerequisites look fine (llama.cpp is built later, in the cpubench step)")


FUNCS = {"sys": check_sys, "hub": check_hub, "data": check_data, "quant": check_quant, "train": check_train, "cpu": check_cpu}


def run_part(cfg, part):
    print(f"\nCHECK {part}", flush=True)
    FUNCS[part](cfg)
    print(f"CHECK {part} DONE", flush=True)
