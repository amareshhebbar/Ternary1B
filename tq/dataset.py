from pathlib import Path

import numpy as np

from .common import banner, dataset_repo, read_json, wpath
from .data import FINEWEB, calib_path, train_path


def card(cfg, stats):
    return f"""
# Ternary1B Data ({cfg['base_label']})

Token ids from `{FINEWEB[0]}` ({FINEWEB[1]}), tokenized with the `{cfg['base_model']}` tokenizer and packed with no padding.

<table>
<tr><th>Config</th><th>Rows</th><th>Use</th></tr>
<tr><td>calibration</td><td>{stats['calib_chunks']} x {stats['calib_seq_len']} tokens</td><td>GPTQ and AWQ calibration</td></tr>
<tr><td>train_{stats['train_seq_len']}</td><td>{stats['train_chunks']} x {stats['train_seq_len']} tokens ({stats['train_tokens']:,} tokens)</td><td>Distillation training for the LoRA repair, QAT and BitNet models</td></tr>
</table>

Calibration documents are skipped when building the training set, so the two never overlap.
"""


def push_dataset(cfg):
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    from .hub import ensure_collection, login

    login()
    stats = read_json(wpath(cfg, "data", "stats.json"))
    if not stats:
        raise RuntimeError("run the data step first")
    api = HfApi()
    repo = dataset_repo(cfg)
    api.create_repo(repo, repo_type="dataset", private=cfg["private"], exist_ok=True)
    parts = [("calibration", calib_path(cfg)), (f"train_{stats['train_seq_len']}", train_path(cfg))]
    for name, p in parts:
        arr = np.load(p, mmap_mode="r")

        def gen(arr=arr):
            for i in range(arr.shape[0]):
                yield {"input_ids": np.asarray(arr[i]).astype(np.int32).tolist()}

        banner(f"pushing {name}: {arr.shape[0]} rows to {repo}")
        Dataset.from_generator(gen).push_to_hub(repo, config_name=name, private=cfg["private"], commit_message=f"Add {name}")
    readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)).read_text()
    front = readme[: readme.find("\n---", 3) + 4] if readme.startswith("---") else ""
    api.upload_file(path_or_fileobj=(front + "\n" + card(cfg, stats)).encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset", commit_message="Update dataset card")
    api.add_collection_item(ensure_collection(cfg), repo, "dataset", note="Calibration and training tokens shared by every model", exists_ok=True)
    for name, p in parts:
        n = len(load_dataset(repo, name, split="train", download_mode="force_redownload"))
        want = np.load(p, mmap_mode="r").shape[0]
        if n != want:
            raise RuntimeError(f"{name}: hub has {n} rows, expected {want}")
        print(f"{name}: {n} rows OK", flush=True)
