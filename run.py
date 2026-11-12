import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

USAGE = """
python run.py check                 run every environment check (or: check sys hub data quant train cpu)
python run.py pipeline              run every step in order, skipping finished ones
python run.py pipeline from=sft:train to=sft:publish
TQ_CONFIG=configs/config_big_budget.yaml python run.py pipeline   use the big budget config
python run.py pipeline force        rerun finished steps too
python run.py run STEP              run one step, for example: run bitnet:verify
python run.py status                show which steps are finished
python run.py reset STEP            mark one step as not finished (reset all clears everything)
python run.py steps                 list every step name
"""


def stream(cmd, log_path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env or os.environ.copy())
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        return p.wait()


def step_env(step):
    env = os.environ.copy()
    if step.endswith(":train"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def run_one(cfg, step):
    from tq.common import banner, mark_done, wpath

    banner(f"STEP {step}")
    t = time.time()
    rc = stream([sys.executable, str(ROOT / "run.py"), "_step", step], wpath(cfg, "logs", f"{step.replace(':', '_')}.log"), step_env(step))
    mins = (time.time() - t) / 60
    if rc == 0:
        mark_done(cfg, step)
        banner(f"STEP {step} finished in {mins:.1f} min")
    else:
        banner(f"STEP {step} FAILED after {mins:.1f} min (exit {rc}). Log: {wpath(cfg, 'logs', step.replace(':', '_') + '.log')}\nFix the issue, then rerun: python run.py pipeline")
    return rc


def cmd_check(cfg, parts):
    from tq.check import PARTS
    from tq.common import banner, mark_done, wpath

    parts = parts or PARTS
    results = {}
    for part in parts:
        if part not in PARTS:
            sys.exit(f"unknown check {part}, choose from {PARTS}")
        rc = stream([sys.executable, str(ROOT / "run.py"), "_check", part], wpath(cfg, "logs", f"check_{part}.log"))
        results[part] = rc == 0
        if part in ("sys", "hub") and rc != 0:
            break
    lines = [f"  {p:<6} {'PASS' if ok else 'FAIL'}" for p, ok in results.items()]
    for p in parts:
        if p not in results:
            lines.append(f"  {p:<6} SKIPPED")
    banner("ENVIRONMENT CHECK\n" + "\n".join(lines))
    if all(results.get(p) for p in parts):
        if set(parts) == set(PARTS):
            mark_done(cfg, "check")
        print("everything works, next: python run.py pipeline", flush=True)
        return 0
    return 1


def cmd_pipeline(cfg, args):
    from tq.common import is_done
    from tq.steps import pipeline_steps

    steps = pipeline_steps()
    opts = dict(a.split("=", 1) for a in args if "=" in a)
    force = "force" in args
    if "from" in opts:
        steps = steps[steps.index(opts["from"]) :]
    if "to" in opts:
        steps = steps[: steps.index(opts["to"]) + 1]
    if not is_done(cfg, "check") and "skipcheck" not in args:
        sys.exit("run python run.py check first (or add skipcheck)")
    for step in steps:
        if is_done(cfg, step) and not force:
            print(f"skip {step} (finished)", flush=True)
            continue
        rc = run_one(cfg, step)
        if rc != 0:
            return rc
    print("pipeline complete", flush=True)
    return 0


def cmd_status(cfg):
    from tq.common import get_state
    from tq.steps import pipeline_steps

    done = get_state(cfg)["done"]
    for step in ["check"] + pipeline_steps():
        print(f"  {'DONE ' + done[step] if step in done else 'todo':<26} {step}")


def main():
    args = sys.argv[1:]
    if not args:
        print(USAGE)
        return 0
    from tq.common import load_cfg, unmark, write_json, state_file

    cfg = load_cfg()
    cmd, rest = args[0], args[1:]
    if cmd == "_step":
        from tq.steps import run_step

        run_step(cfg, rest[0])
        return 0
    if cmd == "_check":
        from tq.check import run_part

        run_part(cfg, rest[0])
        return 0
    if cmd == "check":
        return cmd_check(cfg, rest)
    if cmd == "pipeline":
        return cmd_pipeline(cfg, rest)
    if cmd == "run":
        if not rest:
            sys.exit("usage: python run.py run STEP")
        return run_one(cfg, rest[0])
    if cmd == "status":
        cmd_status(cfg)
        return 0
    if cmd == "steps":
        from tq.steps import pipeline_steps

        print("\n".join(pipeline_steps()))
        return 0
    if cmd == "reset":
        if not rest:
            sys.exit("usage: python run.py reset STEP or reset all")
        if rest[0] == "all":
            write_json(state_file(cfg), {"done": {}, "meta": {}})
        else:
            unmark(cfg, rest[0])
        return 0
    print(USAGE)
    return 1


if __name__ == "__main__":
    sys.exit(main())
