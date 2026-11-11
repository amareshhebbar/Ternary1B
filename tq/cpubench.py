import json
import os
import subprocess
import traceback
from pathlib import Path

from .common import ROOT, banner, merged_dir, resolve_model, wpath, write_json


def sh(cmd, cwd=None):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {(r.stderr or r.stdout)[-1500:]}")
    return r.stdout


def bench(lcpp, gguf, threads, c):
    out = sh([lcpp / "build" / "bin" / "llama-bench", "-m", gguf, "-p", c["prompt_tokens"], "-n", c["gen_tokens"], "-t", threads, "-r", c["repeats"], "-o", "json"])
    rows = json.loads(out[out.index("[") :])
    res = {}
    for r in rows:
        key = "prompt_tok_s" if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0 else "gen_tok_s"
        res[key] = r.get("avg_ts")
    return res


def run_cpubench(cfg):
    c = cfg["cpu_bench"]
    path = wpath(cfg, "results", "cpu_bench.json")
    if not c.get("enabled"):
        print("cpu_bench disabled in config", flush=True)
        return
    lcpp = ROOT / "tools" / "llama.cpp"
    out_dir = wpath(cfg, "gguf")
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = c.get("threads") or max(1, (os.cpu_count() or 4) // 2)
    report = {"threads": threads, "llama_cpp": c["llama_cpp_tag"], "cpu": "", "rows": {}}
    try:
        report["cpu"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    except Exception:
        pass
    try:
        banner(f"building llama.cpp {c['llama_cpp_tag']} for CPU")
        subprocess.run(["bash", str(ROOT / "scripts" / "build_llamacpp.sh"), c["llama_cpp_tag"], str(lcpp)], check=True)
    except Exception as e:
        report["error"] = f"llama.cpp build failed: {e}"
        write_json(path, report)
        print(f"[WARN] {report['error']}. CPU speed is reported as n/a.", flush=True)
        return
    py = lcpp / ".venv_gguf" / "bin" / "python"
    conv = lcpp / "convert_hf_to_gguf.py"
    from huggingface_hub import snapshot_download

    base_dir = snapshot_download(cfg["base_model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    jobs = [("base_f16", base_dir, "f16", None), ("base_q4_k_m", base_dir, "f16", "Q4_K_M")]
    bitnet = merged_dir(cfg, "bitnet")
    if not (bitnet / "config.json").exists():
        try:
            bitnet = Path(resolve_model(cfg, "bitnet"))
        except Exception:
            bitnet = None
    if bitnet:
        jobs += [("bitnet_tq2_0", bitnet, "tq2_0", None), ("bitnet_tq1_0", bitnet, "tq1_0", None)]
    for tag, src, outtype, requant in jobs:
        try:
            f = out_dir / f"{tag}.gguf"
            if not f.exists():
                if requant:
                    f16 = out_dir / "base_f16.gguf"
                    sh([lcpp / "build" / "bin" / "llama-quantize", f16, f, requant])
                else:
                    sh([py, conv, src, "--outtype", outtype, "--outfile", f])
            r = bench(lcpp, f, threads, c)
            r["file_mb"] = f.stat().st_size / 2**20
            report["rows"][tag] = r
            print(f"  {tag}: {r}", flush=True)
        except Exception as e:
            report["rows"][tag] = {"error": str(e)[-500:]}
            print(f"[WARN] {tag} failed: {e}\n{traceback.format_exc()[-800:]}", flush=True)
    write_json(path, report)
    banner("CPU benchmark\n" + json.dumps(report, indent=1))
