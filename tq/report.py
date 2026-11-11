from .common import BASELINES, MODELS, banner, dataset_repo, read_json, repo_id, results_path, wpath, write_json


def collect(cfg):
    out = {}
    for tag in [f"base_{v}" for v in BASELINES] + MODELS:
        r = read_json(results_path(cfg, tag, "full"))
        if r:
            out[tag] = r
    return out


def label(tag):
    return tag.replace("base_", "base ") if tag.startswith("base_") else tag.upper()


def plots(cfg, res=None):
    res = res or collect(cfg)
    if not res:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, r in res.items():
        x, y = r["summary"]["size_mb"], r["wikitext_ppl"]
        if x is None or y is None or y > 1e4:
            continue
        ax.scatter([x], [y], s=70, marker="s" if tag.startswith("base_") else "o")
        ax.annotate(label(tag), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Model size (MB, packed)")
    ax.set_ylabel("WikiText2 perplexity (log, lower is better)")
    ax.set_yscale("log")
    ax.set_title("Ternary1B: size vs quality")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "size_vs_quality.png"), dpi=140)
    plt.close(fig)

    tasks = cfg["eval"]["tasks"]
    if not tasks:
        return
    tags = list(res)
    x = np.arange(len(tags))
    w = 0.8 / len(tasks)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tags)), 4.8))
    for i, t in enumerate(tasks):
        ax.bar(x + (i - (len(tasks) - 1) / 2) * w, [(res[g]["tasks"].get(t) or 0) * 100 for g in tags], w, label=t)
    ax.set_xticks(x)
    ax.set_xticklabels([label(t) for t in tags], rotation=30, ha="right")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Ternary1B: benchmark accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(wpath(cfg, "report", "benchmarks.png"), dpi=140)
    plt.close(fig)


def run_report(cfg):
    from huggingface_hub import HfApi

    from .card import results_table
    from .hub import login, publish

    login()
    res = collect(cfg)
    if not res:
        raise RuntimeError("no full results to report")
    plots(cfg, res)
    summary = {t: {**r["summary"], "tasks": r["tasks"]} for t, r in res.items()}
    summary["cpu_bench"] = read_json(wpath(cfg, "results", "cpu_bench.json"))
    write_json(wpath(cfg, "report", "summary.json"), summary)
    wpath(cfg, "report", "table.html").write_text(results_table(cfg))
    api = HfApi()
    for f in sorted(wpath(cfg, "report").iterdir()):
        if f.suffix in (".png", ".json", ".html") and not f.name.startswith("README"):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"report/{f.name}", repo_id=dataset_repo(cfg), repo_type="dataset", commit_message=f"Add report {f.name}")
    for m in MODELS:
        if read_json(results_path(cfg, m, "full")):
            publish(cfg, m)
    banner("report done\n" + "\n".join(f"{k}: ppl {v['wikitext_ppl']:.2f} size {v['size_mb'] or 0:.0f} MB bits {v['bits'] or 0:.2f}" for k, v in summary.items() if k != "cpu_bench"))
    for m in MODELS:
        print(f"https://huggingface.co/{repo_id(cfg, m)}", flush=True)
