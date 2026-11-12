import json
import math
import shutil

from huggingface_hub import HfApi, snapshot_download

from .common import banner, read_json, repo_id, results_path, wpath, write_json


def verify_model(cfg, name):
    from .evaluate import eval_model
    from .export import compare_unpacked
    from .hub import login

    login()
    repo = repo_id(cfg, name)
    banner(f"verifying {repo} straight from the hub")
    files = set(HfApi().list_repo_files(repo))
    need = ["config.json", "README.md", "tq_runtime.py", "size.json", "packed/quant.safetensors", "packed/rest.safetensors", "packed/quant_config.json"]
    if "lora" in name:
        need.append("packed/lora.safetensors")
    checks = {f: f in files for f in need}
    checks["safetensors"] = any(f.endswith(".safetensors") and "/" not in f for f in files)
    for k, v in checks.items():
        print(f"  {k}: {'OK' if v else 'MISSING'}", flush=True)
    if not all(checks.values()):
        raise SystemExit(f"verify {name}: repo is missing files")
    local = wpath(cfg, "verify", name)
    if local.exists():
        shutil.rmtree(local)
    snapshot_download(repo, local_dir=str(local), ignore_patterns=["training/*", "assets/*", "eval/*"], force_download=True)
    worst, meta = compare_unpacked(local)
    print(f"  packed codes rebuild the bf16 weights with worst relative error {worst:.2e} ({meta.get('format')})", flush=True)
    res = eval_model(cfg, name, "quick", str(local))
    ppl = res["wikitext_ppl"]
    ref_tag = "base_rtn_ternary" if name == "bitnet" else "base_rtn2"
    ref = read_json(results_path(cfg, ref_tag, "full"), {}).get("wikitext_ppl")
    limit = cfg["eval"]["gate_max_ppl"]
    if ref and math.isfinite(ref):
        limit = max(limit, ref * cfg["eval"].get("gate_vs_rounding", 2.0))
    print(f"  perplexity {ppl:.4g}, plain rounding baseline {ref}, limit {limit:.4g}", flush=True)
    gates = {"packed_matches_weights": worst <= cfg["eval"]["gate_unpack_tol"], "perplexity": math.isfinite(ppl) and ppl <= limit}
    report = {"repo": repo, "unpack_worst_rel_err": worst, "quick": res["summary"], "gates": gates, "passed": all(gates.values())}
    write_json(wpath(cfg, "results", f"{name}_verify.json"), report)
    banner(f"verify {name}\n" + json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"verify {name} FAILED, see {wpath(cfg, 'results', name + '_verify.json')}")
    print(f"verify {name} PASSED", flush=True)
