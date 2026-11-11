from .common import ACTIONS, MODELS


def pipeline_steps():
    steps = ["data", "dataset_push", "baseline"]
    for m in MODELS:
        steps += [f"{m}:{a}" for a in ACTIONS]
    steps += ["cpubench", "report"]
    return steps


def run_step(cfg, step):
    if step == "data":
        from .data import run_data

        return run_data(cfg)
    if step == "dataset_push":
        from .dataset import push_dataset

        return push_dataset(cfg)
    if step == "baseline":
        from .evaluate import run_baseline

        return run_baseline(cfg)
    if step == "cpubench":
        from .cpubench import run_cpubench

        return run_cpubench(cfg)
    if step == "report":
        from .report import run_report

        return run_report(cfg)
    if ":" not in step:
        raise SystemExit(f"unknown step {step}")
    name, action = step.split(":", 1)
    if name not in MODELS or action not in ACTIONS:
        raise SystemExit(f"unknown step {step}")
    if action == "train":
        from .train import train

        return train(cfg, name)
    if action == "push":
        from .hub import push_model

        return push_model(cfg, name)
    if action == "verify":
        from .verify import verify_model

        return verify_model(cfg, name)
    if action == "evalgen":
        from .evaluate import eval_model

        return eval_model(cfg, name, "full")
    if action == "publish":
        from .hub import publish

        return publish(cfg, name)
