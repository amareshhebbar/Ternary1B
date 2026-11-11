import os
from pathlib import Path

from huggingface_hub import HfApi

from .common import banner, get_meta, merged_dir, model_dir, repo_id, results_path, set_meta, wpath


def login():
    from huggingface_hub import login as hf_login

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN=hf_...")
    hf_login(token=token, add_to_git_credential=False)
    return token


def ensure_collection(cfg):
    slug = get_meta(cfg, "collection_slug")
    api = HfApi()
    if slug:
        try:
            api.get_collection(slug)
            return slug
        except Exception:
            pass
    c = api.create_collection(title=cfg["collection"]["title"], namespace=cfg["hf_user"], description=cfg["collection"]["description"][:150], private=cfg["private"], exists_ok=True)
    set_meta(cfg, "collection_slug", c.slug)
    print(f"collection: https://huggingface.co/collections/{c.slug}", flush=True)
    return c.slug


def push_model(cfg, name):
    from .card import write_card

    login()
    api = HfApi()
    repo = repo_id(cfg, name)
    local = merged_dir(cfg, name)
    if not (local / "config.json").exists():
        raise RuntimeError(f"no saved model at {local}, run {name}:train first")
    banner(f"pushing {local} to {repo}")
    api.create_repo(repo, private=cfg["private"], exist_ok=True)
    write_card(cfg, name, local / "README.md")
    api.upload_folder(folder_path=str(local), repo_id=repo, commit_message=f"Upload Ternary1B {name.upper()} weights")
    logs = model_dir(cfg, name) / "training"
    if logs.exists():
        api.upload_folder(folder_path=str(logs), path_in_repo="training", repo_id=repo, commit_message="Upload training logs")
    api.add_collection_item(ensure_collection(cfg), repo, "model", note=f"Ternary1B {name.upper()}", exists_ok=True)
    print(f"pushed https://huggingface.co/{repo}", flush=True)


def publish(cfg, name):
    from .card import write_card
    from .report import plots

    login()
    plots(cfg)
    api = HfApi()
    repo = repo_id(cfg, name)
    out = wpath(cfg, "report", f"README_{name}.md")
    write_card(cfg, name, out)
    for png in sorted(Path(wpath(cfg, "report")).glob("*.png")):
        if png.name.startswith("needle_") and png.name != f"needle_{name}.png":
            continue
        api.upload_file(path_or_fileobj=str(png), path_in_repo=f"assets/{png.name}", repo_id=repo, commit_message=f"Add {png.name}")
    for mode in ["full", "quick"]:
        p = results_path(cfg, name, mode)
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"eval/{name}_{mode}.json", repo_id=repo, commit_message=f"Add {mode} eval results")
    sp = wpath(cfg, "results", f"{name}_short.json")
    if sp.exists():
        api.upload_file(path_or_fileobj=str(sp), path_in_repo=f"eval/{name}_short.json", repo_id=repo, commit_message="Add short context scores")
    api.upload_file(path_or_fileobj=str(out), path_in_repo="README.md", repo_id=repo, commit_message="Update model card with evaluation results")
    print(f"published card and results for https://huggingface.co/{repo}", flush=True)
