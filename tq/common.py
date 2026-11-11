import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ["gptq", "awq", "gptq_lora", "hqq_lora", "qat", "bitnet"]
MODELS = POLICIES
BASELINES = ["bf16", "rtn2", "rtn_ternary"]
ACTIONS = ["train", "push", "verify", "evalgen", "publish"]


def load_cfg():
    load_dotenv(ROOT / ".env", override=False)
    path = Path(os.environ.get("TQ_CONFIG") or ROOT / "configs" / "config.yaml")
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    work = os.environ.get("TQ_WORK") or cfg["paths"]["work"]
    if not Path(work).parent.exists():
        work = str(ROOT / "runs")
    cfg["paths"]["work"] = work
    cfg["hf_user"] = os.environ.get("HF_USER") or cfg["hf_user"]
    for sub in ["results", "models", "data", "logs", "verify", "report"]:
        (Path(work) / sub).mkdir(parents=True, exist_ok=True)
    return cfg


def wpath(cfg, *parts):
    return Path(cfg["paths"]["work"]).joinpath(*parts)


def repo_id(cfg, name):
    return f"{cfg['hf_user']}/{cfg['repo_prefix']}_{name.upper()}"


def dataset_repo(cfg):
    return f"{cfg['hf_user']}/{cfg['dataset_name']}"


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(p)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def state_file(cfg):
    return wpath(cfg, "state.json")


def get_state(cfg):
    return read_json(state_file(cfg), {"done": {}, "meta": {}})


def set_meta(cfg, key, value):
    s = get_state(cfg)
    s["meta"][key] = value
    write_json(state_file(cfg), s)


def get_meta(cfg, key, default=None):
    return get_state(cfg)["meta"].get(key, default)


def mark_done(cfg, step):
    s = get_state(cfg)
    s["done"][step] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(state_file(cfg), s)


def unmark(cfg, step):
    s = get_state(cfg)
    s["done"].pop(step, None)
    write_json(state_file(cfg), s)


def is_done(cfg, step):
    return step in get_state(cfg)["done"]


def model_dir(cfg, name):
    return wpath(cfg, "models", name)


def merged_dir(cfg, name):
    return model_dir(cfg, name) / "merged"


def resolve_model(cfg, name):
    if name == "base":
        return cfg["base_model"]
    local = merged_dir(cfg, name)
    if (local / "config.json").exists():
        return str(local)
    hub_copy = wpath(cfg, "verify", name)
    if (hub_copy / "config.json").exists():
        return str(hub_copy)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id(cfg, name),
        local_dir=str(hub_copy),
        ignore_patterns=["training/*", "assets/*", "eval/*"],
    )


def init_repo_for(cfg, init):
    return cfg["base_model"] if init == "base" else repo_id(cfg, init)


def results_path(cfg, tag, mode="full"):
    return wpath(cfg, "results", f"{tag}_{mode}.json")


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)
