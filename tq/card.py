from pathlib import Path

from .common import BASELINES, MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath

METHOD = {
    "gptq": "GPTQ at 2 bits, no training. Each layer is quantized using a small sample of text: after rounding each weight column, the remaining weights are adjusted to cancel the rounding error, using second order information from the layer inputs.",
    "awq": "AWQ at 2 bits, no training. Input channels that carry large activations matter most, so their weights are scaled up before rounding and scaled back after, keeping more precision where it counts. The best scaling strength is searched per layer.",
    "gptq_lora": "The GPTQ 2 bit model, frozen, plus a small LoRA adapter trained to repair the rounding damage by copying the original model's predictions (knowledge distillation).",
    "hqq_lora": "HQQ at 2 bits (a fast quantizer that needs no calibration data and optimizes each group's zero point directly), frozen, plus a small LoRA adapter trained by distillation from the original model.",
    "qat": "Quantization aware training at 2 bits. The whole model trains while its weights are rounded to 2 bits in every forward pass, using the straight through estimator for gradients and distillation from the original model, so it learns weights that survive rounding.",
    "bitnet": "BitNet b1.58 conversion. Every linear layer becomes a BitLinear layer: weights are scaled by their average absolute value and rounded to minus 1, 0 or plus 1, and activations are rounded to 8 bits. A warmup blends full precision into ternary step by step so the converted model does not collapse, with distillation from the original model.",
}

BITS = {"gptq": "2", "awq": "2", "gptq_lora": "2 plus adapter", "hqq_lora": "2 plus adapter", "qat": "2", "bitnet": "1.58 (ternary)"}


def _pct(x):
    return "" if x is None else f"{x * 100:.1f}"


def _f(x, fmt="{:.2f}"):
    return "" if x is None else fmt.format(x)


def rows(cfg):
    from .evaluate import BASE_LABEL

    out = []
    for v in BASELINES:
        r = read_json(results_path(cfg, f"base_{v}", "full"))
        if r:
            out.append((f"base_{v}", f"{cfg['base_label']} ({BASE_LABEL[v]})", r))
    for m in MODELS:
        r = read_json(results_path(cfg, m, "full"))
        if r:
            out.append((m, f"Ternary1B {m.upper()}", r))
    return out


def results_table(cfg, highlight=None):
    rs = rows(cfg)
    if not rs:
        return "<p>Evaluation results are added after the full evaluation step.</p>"
    tasks = cfg["eval"]["tasks"]
    head = "<tr><th>Model</th><th>Bits per weight</th><th>Size (MB)</th><th>Smaller by</th><th>WikiText2 perplexity</th>" + "".join(f"<th>{t}</th>" for t in tasks) + "<th>Benchmark mean</th></tr>"
    body = []
    for tag, label, r in rs:
        s = r["summary"]
        cells = [label, _f(s.get("bits")), _f(s.get("size_mb"), "{:.0f}"), _f(s.get("compression"), "{:.1f}x"), _f(r["wikitext_ppl"])]
        cells += [_pct(r["tasks"].get(t)) for t in tasks]
        cells.append(_pct(s.get("task_mean")))
        if tag == highlight:
            cells = [f"<b>{c}</b>" for c in cells]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "<table>\n" + head + "\n" + "\n".join(body) + "\n</table>"


def cpu_table(cfg):
    r = read_json(wpath(cfg, "results", "cpu_bench.json"))
    if not r or not r.get("rows"):
        return "<p>CPU speed is measured with llama.cpp in the cpubench step at the end of the run.</p>"
    names = {"base_f16": f"{cfg['base_label']} f16", "base_q4_k_m": f"{cfg['base_label']} Q4_K_M (4 bit reference)", "bitnet_tq2_0": "Ternary1B BITNET as TQ2_0", "bitnet_tq1_0": "Ternary1B BITNET as TQ1_0"}
    body = []
    for k, v in r["rows"].items():
        if "error" in v:
            body.append(f"<tr><td>{names.get(k, k)}</td><td colspan=3>not available</td></tr>")
        else:
            body.append(f"<tr><td>{names.get(k, k)}</td><td>{v['file_mb']:.0f}</td><td>{_f(v.get('prompt_tok_s'), '{:.1f}')}</td><td>{_f(v.get('gen_tok_s'), '{:.1f}')}</td></tr>")
    return (
        f"<p>llama.cpp {r['llama_cpp']}, {r['threads']} threads on {r.get('cpu') or 'the pod CPU'}.</p>\n<table>\n<tr><th>Model file</th><th>Size (MB)</th><th>Prompt tokens/s</th><th>Generated tokens/s</th></tr>\n"
        + "\n".join(body)
        + "\n</table>"
    )


def samples(cfg, name):
    r = read_json(results_path(cfg, name, "full"))
    if not r or not r.get("samples"):
        return ""
    return "\n".join(f"* **{s['prompt']}** {s['completion'].strip()}" for s in r["samples"])


def hyperparams(cfg, name):
    t = {"group_size": cfg["quant"]["group"]}
    if name in ("gptq_lora", "hqq_lora"):
        t.update({f"heal_{k}": v for k, v in cfg["heal"].items()})
    elif name in ("qat", "bitnet"):
        t.update({f"{name}_{k}": v for k, v in cfg[name].items()})
    elif name == "gptq":
        t.update({"damp": cfg["quant"]["gptq_damp"], "calibration": f"{cfg['calib']['n']} x {cfg['calib']['seq_len']} tokens"})
    elif name == "awq":
        t.update({"grid": cfg["quant"]["awq_grid"], "calibration": f"{cfg['calib']['n']} x {cfg['calib']['seq_len']} tokens"})
    if name in ("gptq_lora", "hqq_lora", "qat", "bitnet"):
        t.update({f"train_{k}": v for k, v in cfg["train"].items()})
    size = read_json(wpath(cfg, "models", name, "size.json"), {})
    t.update({f"size_{k}": v for k, v in size.items()})
    return "<table>\n" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in t.items()) + "\n</table>"


def card(cfg, name):
    repo = repo_id(cfg, name)
    slug = get_meta(cfg, "collection_slug")
    coll = f"https://huggingface.co/collections/{slug}" if slug else f"https://huggingface.co/{cfg['hf_user']}"
    imgs = "".join(f"\n![{f}](assets/{f}.png)\n" for f in ("size_vs_quality", "benchmarks") if Path(wpath(cfg, "report", f"{f}.png")).exists())
    act = name == "bitnet"
    return f"""---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - {cfg['base_model']}
base_model_relation: quantized
datasets:
  - {dataset_repo(cfg)}
language:
  - en
tags:
  - ternary1b
  - quantization
  - {"ternary" if act else "2bit"}
  - smollm2
  - {name}
---

# Ternary1B {name.upper()} ({cfg['base_label']})

`{cfg['base_model']}` stores each weight in 16 bits. This model stores the weights of every attention and feed forward layer in {BITS[name]} bits, so the whole model is several times smaller and can run on phones and cheap CPUs. It is one of six models in the [Ternary1B collection]({coll}), each using a different way to survive extreme compression, all scored on the same tests.

## Method

{METHOD[name]}

Embeddings and normalization layers stay in 16 bit, as in BitNet. Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}).

## Why train instead of just rounding

Retrieval and prompting cannot make a model smaller. Plain rounding to 2 bits or ternary destroys it, which the baseline rows show. Every gain over those rows comes from changing the weights.

## Results

Perplexity on WikiText2 test (lower is better). Benchmarks are zero shot accuracy from lm evaluation harness{f", limited to {cfg['eval']['limit']} examples per task" if cfg['eval'].get('limit') else ""}. Size is the packed file size in `packed/`: real 2 bit or ternary codes plus their scales, with embeddings in 16 bit.

{results_table(cfg, highlight=name)}
{imgs}
### Speed on a CPU

{cpu_table(cfg)}

Only the ternary model maps exactly onto llama.cpp's ternary formats, so 2 bit models are not timed on CPU.

### Sample completions (greedy)

{samples(cfg, name)}

## Files

* Repository root: the same weights expanded to bf16 safetensors, so it loads in transformers with no extra code. {"For faithful output, also quantize activations to 8 bits as below." if act else ""}
* `packed/`: the real compressed form. `quant.safetensors` holds packed codes (4 weights per byte at 2 bits, 5 weights per byte for ternary) and scales, `rest.safetensors` holds embeddings and norms{", `lora.safetensors` holds the repair adapter" if "lora" in name else ""}, and `quant_config.json` describes the layout.
* `tq_runtime.py`: unpacks `packed/` into a working model.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}", torch_dtype="bfloat16")
{"from huggingface_hub import hf_hub_download" + chr(10) + "import importlib.util" + chr(10) + f'spec = importlib.util.spec_from_file_location("tq_runtime", hf_hub_download("{repo}", "tq_runtime.py"))' + chr(10) + "rt = importlib.util.module_from_spec(spec)" + chr(10) + "spec.loader.exec_module(rt)" + chr(10) + "rt.patch_act_quant(model)" + chr(10) if act else ""}ids = tok("The capital of France is", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=30, do_sample=False)[0]))
```

To build the model from the packed files only: download the repo, then `tq_runtime.load_model(path)`.

## Training hyperparameters

{hyperparams(cfg, name)}
"""


def write_card(cfg, name, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(card(cfg, name))
