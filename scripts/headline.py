import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from tq.common import BASELINES, MODELS, dataset_repo, get_meta, read_json, repo_id, results_path, wpath, write_json

TWO_BIT = ["gptq", "awq", "gptq_lora", "hqq_lora", "qat"]


def pct(x):
    return "" if x is None else f"{x * 100:.1f}%"


def num(x):
    return "" if x is None else (f"{x:.2f}" if x < 1000 else f"{x:,.0f}")


def row(r):
    s = r["summary"]
    return {"ppl": r["wikitext_ppl"], "task_mean": s.get("task_mean"), "size_mb": s.get("size_mb"), "bits": s.get("bits"), "compression": s.get("compression"), "tasks": r["tasks"]}


def build(cfg):
    rows = {}
    for tag in [f"base_{v}" for v in BASELINES] + MODELS:
        r = read_json(results_path(cfg, tag, "full"))
        if r:
            rows[tag] = row(r)
    if "base_bf16" not in rows:
        raise SystemExit("base_bf16 results missing, run the baseline step first")
    two = [m for m in TWO_BIT if m in rows]
    if not two and "bitnet" not in rows:
        raise SystemExit("no model results yet, run the pipeline first")
    best2 = min(two, key=lambda m: rows[m]["ppl"]) if two else None
    return rows, best2


def cpu(cfg):
    r = read_json(wpath(cfg, "results", "cpu_bench.json"), {}) or {}
    rows = r.get("rows", {})
    f16, tern = rows.get("base_f16", {}), rows.get("bitnet_tq2_0", {})
    if f16.get("gen_tok_s") and tern.get("gen_tok_s"):
        return tern["gen_tok_s"] / f16["gen_tok_s"], r
    return None, r


LABEL = {"base_bf16": "original bf16", "base_rtn2": "plain 2 bit rounding", "base_rtn_ternary": "plain ternary rounding"}


def label(cfg, tag):
    return f"{cfg['base_label']} ({LABEL[tag]})" if tag in LABEL else f"Ternary1B {tag.upper()}"


def sentence(cfg, rows, best2):
    b = rows["base_bf16"]
    parts = []
    if "bitnet" in rows:
        v = rows["bitnet"]
        rt = rows.get("base_rtn_ternary", {})
        speed, _ = cpu(cfg)
        s = (
            f"Ternary1B BITNET stores {cfg['base_label']}'s layer weights as minus 1, 0 or plus 1 (1.58 bits): {v['size_mb']:.0f} MB versus {b['size_mb']:.0f} MB ({v['compression']:.1f}x smaller), "
            f"WikiText2 perplexity {num(v['ppl'])} versus {num(b['ppl'])} for the original and {num(rt.get('ppl'))} for plain ternary rounding, benchmark mean {pct(v['task_mean'])} versus {pct(b['task_mean'])}"
        )
        if speed:
            s += f", and {speed:.1f}x faster text generation on CPU than the original in f16"
        parts.append(s + ".")
    if best2:
        v = rows[best2]
        r2 = rows.get("base_rtn2", {})
        parts.append(
            f"The best 2 bit model, {best2.upper()}, is {v['compression']:.1f}x smaller with perplexity {num(v['ppl'])} (plain 2 bit rounding: {num(r2.get('ppl'))}) and benchmark mean {pct(v['task_mean'])}."
        )
    return " ".join(parts)


def markdown(cfg, rows, best2):
    coll = get_meta(cfg, "collection_slug")
    tasks = cfg["eval"]["tasks"]
    speed, cb = cpu(cfg)
    headline_model = "bitnet" if "bitnet" in rows else best2
    lines = [
        f"# Ternary1B results ({cfg['base_label']})",
        "",
        f"**Headline:** {sentence(cfg, rows, best2)}",
        "",
        "Size is the real packed file size: 2 bit or ternary codes plus their scales, with embeddings kept in 16 bit. Perplexity is on WikiText2 test (lower is better). "
        f"Benchmarks are zero shot accuracy from lm evaluation harness{f', up to {cfg['eval']['limit']} examples per task' if cfg['eval'].get('limit') else ''}.",
        "",
        "| Model | Bits per weight | Size (MB) | Smaller by | WikiText2 perplexity | " + " | ".join(tasks) + " | Benchmark mean |",
        "|" + "---|" * (6 + len(tasks)),
    ]
    order = [f"base_{x}" for x in BASELINES] + [m for m in MODELS if m in rows]
    for tag in [t for t in order if t in rows]:
        v = rows[tag]
        name = label(cfg, tag)
        if tag in MODELS:
            name = f"[{name}](https://huggingface.co/{repo_id(cfg, tag)})"
        if tag == headline_model:
            name = f"**{name}**"
        cells = [name, f"{v['bits']:.2f}" if v["bits"] else "", f"{v['size_mb']:.0f}" if v["size_mb"] else "", f"{v['compression']:.1f}x" if v["compression"] else "", num(v["ppl"])]
        cells += [pct(v["tasks"].get(t)) for t in tasks] + [pct(v["task_mean"])]
        lines.append("| " + " | ".join(cells) + " |")
    if cb.get("rows"):
        lines += ["", f"## CPU speed (llama.cpp {cb.get('llama_cpp')}, {cb.get('threads')} threads, {cb.get('cpu') or 'pod CPU'})", "", "| Model file | Size (MB) | Prompt tokens/s | Generated tokens/s |", "|---|---|---|---|"]
        for k, v in cb["rows"].items():
            if "error" in v:
                lines.append(f"| {k} | not available | | |")
            else:
                lines.append(f"| {k} | {v['file_mb']:.0f} | {v.get('prompt_tok_s') or 0:.1f} | {v.get('gen_tok_s') or 0:.1f} |")
    lines += [
        "",
        "## Check it yourself",
        "",
        f"* Every model card has the full table, charts and training settings: [collection](https://huggingface.co/collections/{coll})" if coll else "* Every model card has the full table, charts and training settings.",
        f"* The packed files behind the size numbers: [{headline_model.upper()} packed](https://huggingface.co/{repo_id(cfg, headline_model)}/tree/main/packed).",
        f"* Raw evaluation files: [{headline_model.upper()} eval](https://huggingface.co/{repo_id(cfg, headline_model)}/tree/main/eval).",
        f"* Training data: [{dataset_repo(cfg)}](https://huggingface.co/datasets/{dataset_repo(cfg)}).",
        "* Code to reproduce every number: [github.com/amareshhebbar/Ternary1B](https://github.com/amareshhebbar/Ternary1B).",
        "",
        f"Generated {time.strftime('%Y-%m-%d')} from the pipeline's own result files.",
        "",
    ]
    return "\n".join(lines), headline_model


def main():
    from huggingface_hub import HfApi

    from tq.common import load_cfg
    from tq.hub import login

    cfg = load_cfg()
    rows, best2 = build(cfg)
    for tag, v in rows.items():
        print(f"{tag:17s} bits {v['bits'] or 0:5.2f}  size {v['size_mb'] or 0:7.0f} MB  ppl {num(v['ppl']):>9s}  benchmarks {pct(v['task_mean'])}")
    md, top = markdown(cfg, rows, best2)
    wpath(cfg, "report", "HEADLINE.md").write_text(md)
    write_json(wpath(cfg, "report", "headline.json"), {"headline_model": top, "best_2bit": best2, "sentence": sentence(cfg, rows, best2), "rows": rows})
    print("\n" + sentence(cfg, rows, best2) + "\n")
    if "--local" in sys.argv:
        print(f"written to {wpath(cfg, 'report', 'HEADLINE.md')}, not uploaded")
        return
    login()
    api = HfApi()
    ds = dataset_repo(cfg)
    for f in ["HEADLINE.md", "headline.json"]:
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f"report/{f}", repo_id=ds, repo_type="dataset", commit_message=f"Add {f}")
        api.upload_file(path_or_fileobj=str(wpath(cfg, "report", f)), path_in_repo=f, repo_id=repo_id(cfg, top), commit_message=f"Add {f}")
    print("proof links:")
    print(f"  https://huggingface.co/{repo_id(cfg, top)}/blob/main/HEADLINE.md")
    print(f"  https://huggingface.co/datasets/{ds}/blob/main/report/HEADLINE.md")


if __name__ == "__main__":
    main()
