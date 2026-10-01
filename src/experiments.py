"""
The four runs, plus the dense phase they branch from.

    dense phase     steps 0 .. switch          dense model, trained once
    A dense_cont    steps switch .. total      the same dense model, keeps training     (control)
    B moe_drop      steps switch .. total      dense -> MoE by drop-upcycling            (main)
    D moe_clone     steps switch .. total      dense -> MoE by cloning slices           (Rohan's V4 failure case)
    C moe_scratch   steps 0 .. total           MoE from random init, full budget        (does upcycling save compute?)

Every run caches its result in results/<name>.json and checkpoints into runs/<name>/.
Each run prints one of three words when it starts, so you always know what happened:
    TRAINING        - nothing on disk, training from the beginning
    RESUMING        - found a checkpoint (e.g. after a Colab disconnect), continuing from it
    LOADED FROM CACHE - the finished result is already on disk; nothing was trained.
                       Delete results/<name>.json to force a re-run.
"""
import os

import torch

from .model import GPT
from .train import (carry_optimizer_state, load_ckpt, make_optimizer, new_log, read_json,
                    save_ckpt, train_segment, write_json)
from .upcycle import upcycle


def _scaler(device):
    return torch.amp.GradScaler("cuda") if device == "cuda" else None


def _paths(root, name):
    os.makedirs(os.path.join(root, "runs", name), exist_ok=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    return (os.path.join(root, "runs", name, "ckpt.pt"),
            os.path.join(root, "results", f"{name}.json"))


@torch.no_grad()
def expert_similarity(model):
    """Per MoE layer: how alike the routed experts are.
    mean = average cosine similarity over all expert pairs, max = the most alike pair.
    Exact clones give max = 1.0. Computed on each expert's input matrix, flattened.
    (Neuron order inside an expert is fixed at birth, so clones stay aligned.)"""
    out = []
    for m in model.moe_layers():
        w = m.w_in.detach().float().reshape(m.E, -1)
        w = w / w.norm(dim=1, keepdim=True)
        sim = w @ w.t()
        off = sim[~torch.eye(m.E, dtype=torch.bool, device=sim.device)]
        out.append({"mean": off.mean().item(), "max": off.max().item()})
    return out


def run_dense_phase(root, mc, tc, data, device):
    name = "dense_phase"
    ckpt, res = _paths(root, name)
    switch_ckpt = os.path.join(root, "runs", name, "at_switch.pt")
    if os.path.exists(res) and os.path.exists(switch_ckpt):
        print(f"[{name}] LOADED FROM CACHE ({res})")
        return read_json(res), switch_ckpt
    torch.manual_seed(tc.seed)
    model = GPT(mc, moe=False).to(device)
    opt, scaler = make_optimizer(model, tc), _scaler(device)
    start, log = 0, new_log(name, model, mc, tc, "dense")
    if os.path.exists(ckpt):
        start, log, _ = load_ckpt(ckpt, model, opt, scaler, device)
        print(f"[{name}] RESUMING from step {start}")
    else:
        print(f"[{name}] TRAINING  params={log['params_total']:,}")
    train_segment(model, opt, scaler, data, tc, start, tc.switch_step, log, device, ckpt, tag=f"[{name}]")
    save_ckpt(switch_ckpt, model, opt, scaler, tc.switch_step, log)
    write_json(res, log)
    return log, switch_ckpt


def _load_dense_at_switch(switch_ckpt, mc, tc, device):
    model = GPT(mc, moe=False).to(device)
    opt, scaler = make_optimizer(model, tc), _scaler(device)
    step, log, _ = load_ckpt(switch_ckpt, model, opt, scaler, device)
    return model, opt, scaler, step, log


def run_branch(root, name, method, mc, tc, data, device, switch_ckpt, dense_log):
    """method: None (keep dense), "drop" or "clone"."""
    ckpt, res = _paths(root, name)
    if os.path.exists(res):
        print(f"[{name}] LOADED FROM CACHE ({res})")
        return read_json(res)
    dense, dopt, dscaler, sw, _ = _load_dense_at_switch(switch_ckpt, mc, tc, device)
    dense_val_at_switch = next(v["loss"] for v in dense_log["val"] if v["step"] == sw)
    if method is None:
        model, opt, scaler = dense, dopt, dscaler
        log = new_log(name, model, mc, tc, "dense")
    else:
        model = upcycle(dense, mc, method, drop_ratio=tc.drop_ratio, seed=tc.seed)
        opt = make_optimizer(model, tc)
        carried = carry_optimizer_state(dense, dopt, model, opt)
        scaler = _scaler(device)
        if scaler is not None:
            scaler.load_state_dict(dscaler.state_dict())
        log = new_log(name, model, mc, tc, f"moe_{method}")
        log["optimizer_state_carried"] = len(carried)
        log["expert_similarity_at_birth"] = expert_similarity(model)
        del dense
    log["branched_from"] = "dense_phase"
    log["switch_step"] = sw
    log["dense_val_at_switch"] = dense_val_at_switch
    start = sw
    if os.path.exists(ckpt):
        start, log, _ = load_ckpt(ckpt, model, opt, scaler, device)
        print(f"[{name}] RESUMING from step {start}")
    else:
        print(f"[{name}] TRAINING  from step {sw}  params total={log['params_total']:,} active={log['params_active']:,}")
    extra = [sw + o for o in tc.post_switch_evals]
    train_segment(model, opt, scaler, data, tc, start, tc.total_steps, log, device, ckpt,
                  explore_until=sw + tc.explore_steps, extra_eval_steps=extra, tag=f"[{name}]")
    if model.is_moe:
        log["expert_similarity_final"] = expert_similarity(model)
    write_json(res, log)
    return log


def run_scratch(root, name, mc, tc, data, device):
    ckpt, res = _paths(root, name)
    if os.path.exists(res):
        print(f"[{name}] LOADED FROM CACHE ({res})")
        return read_json(res)
    torch.manual_seed(tc.seed)
    model = GPT(mc, moe=True).to(device)
    opt, scaler = make_optimizer(model, tc), _scaler(device)
    start, log = 0, new_log(name, model, mc, tc, "moe_scratch")
    log["expert_similarity_at_birth"] = expert_similarity(model)
    if os.path.exists(ckpt):
        start, log, _ = load_ckpt(ckpt, model, opt, scaler, device)
        print(f"[{name}] RESUMING from step {start}")
    else:
        print(f"[{name}] TRAINING  params total={log['params_total']:,} active={log['params_active']:,}")
    train_segment(model, opt, scaler, data, tc, start, tc.total_steps, log, device, ckpt,
                  explore_until=tc.explore_steps, tag=f"[{name}]")
    log["expert_similarity_final"] = expert_similarity(model)
    write_json(res, log)
    return log


def run_all(root, mc, tc, data, device, which=("A", "B", "D", "C")):
    dense_log, switch_ckpt = run_dense_phase(root, mc, tc, data, device)
    logs = {"dense_phase": dense_log}
    if "A" in which:
        logs["A_dense_cont"] = run_branch(root, "A_dense_cont", None, mc, tc, data, device, switch_ckpt, dense_log)
    if "B" in which:
        logs["B_moe_drop"] = run_branch(root, "B_moe_drop", "drop", mc, tc, data, device, switch_ckpt, dense_log)
    if "D" in which:
        logs["D_moe_clone"] = run_branch(root, "D_moe_clone", "clone", mc, tc, data, device, switch_ckpt, dense_log)
    if "C" in which:
        logs["C_moe_scratch"] = run_scratch(root, "C_moe_scratch", mc, tc, data, device)
    return logs
