import numpy as np

from .common import banner, read_json, wpath, write_json

FINEWEB = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
WIKITEXT = ("wikitext", "wikitext-2-raw-v1")


def stream_texts():
    from datasets import load_dataset

    for r in load_dataset(FINEWEB[0], FINEWEB[1], split="train", streaming=True):
        if r.get("text"):
            yield r["text"]


def pack(tok, n_chunks, seq_len, skip_docs=0, label=""):
    buf, chunks, docs = [], [], 0
    for i, text in enumerate(stream_texts()):
        if i < skip_docs:
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        docs += 1
        while len(buf) >= seq_len and len(chunks) < n_chunks:
            chunks.append(np.asarray(buf[:seq_len], dtype=np.uint16))
            buf = buf[seq_len:]
        if docs % 20000 == 0:
            print(f"  {label}: {docs} docs, {len(chunks)} of {n_chunks} chunks", flush=True)
        if len(chunks) >= n_chunks:
            break
    return np.stack(chunks), docs


def calib_path(cfg):
    return wpath(cfg, "data", f"calib_{cfg['calib']['n']}x{cfg['calib']['seq_len']}.npy")


def train_path(cfg):
    return wpath(cfg, "data", f"train_{cfg['train']['seq_len']}.npy")


def train_chunks_needed(cfg):
    t = cfg["train"]
    per_step = t["micro_batch"] * t["grad_accum"]
    return max(cfg["qat"]["steps"], cfg["bitnet"]["steps"], cfg["heal"]["steps"]) * per_step


def calib(cfg, tok=None):
    p = calib_path(cfg)
    if p.exists():
        return np.load(p)
    tok = tok or tokenizer(cfg)
    arr, docs = pack(tok, cfg["calib"]["n"], cfg["calib"]["seq_len"], label="calibration")
    np.save(p, arr)
    write_json(wpath(cfg, "data", "calib_docs.json"), {"docs": docs})
    return arr


def train_chunks(cfg, tok=None):
    p = train_path(cfg)
    need = train_chunks_needed(cfg)
    if p.exists():
        arr = np.load(p, mmap_mode="r")
        if arr.shape[0] >= need:
            return arr
    tok = tok or tokenizer(cfg)
    calib(cfg, tok)
    skip = read_json(wpath(cfg, "data", "calib_docs.json"), {}).get("docs", 0)
    banner(f"tokenizing FineWeb Edu into {need} training chunks of {cfg['train']['seq_len']} tokens")
    arr, docs = pack(tok, need, cfg["train"]["seq_len"], skip_docs=skip, label="training")
    arr = arr[np.random.default_rng(cfg["seed"]).permutation(len(arr))]
    np.save(p, arr)
    write_json(wpath(cfg, "data", "train_stats.json"), {"docs": docs, "chunks": int(arr.shape[0]), "tokens": int(arr.size), "skipped_calibration_docs": skip})
    return np.load(p, mmap_mode="r")


def wikitext_ids(cfg, tok=None):
    p = wpath(cfg, "data", "wikitext2_test.npy")
    if p.exists():
        return np.load(p)
    from datasets import load_dataset

    tok = tok or tokenizer(cfg)
    text = "\n\n".join(load_dataset(WIKITEXT[0], WIKITEXT[1], split="test")["text"])
    ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int32)
    np.save(p, ids)
    return ids


def tokenizer(cfg):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["base_model"])


def run_data(cfg):
    tok = tokenizer(cfg)
    c = calib(cfg, tok)
    t = train_chunks(cfg, tok)
    w = wikitext_ids(cfg, tok)
    stats = {"calib_chunks": int(c.shape[0]), "calib_seq_len": int(c.shape[1]), "train_chunks": int(t.shape[0]), "train_seq_len": int(t.shape[1]), "train_tokens": int(t.shape[0] * t.shape[1]), "wikitext_tokens": int(len(w))}
    write_json(wpath(cfg, "data", "stats.json"), stats)
    banner("data ready\n" + "\n".join(f"{k}: {v}" for k, v in stats.items()))
