"""
The training loop, shared by every run.

One function, `train_segment`, trains a model from step `start` to step `end` on the
global step clock. Because the learning-rate schedule and the data are both keyed on
the GLOBAL step number, a run that converts to MoE at step 3000 continues on exactly
the same schedule and the same batches as the dense run it branched from.

What gets logged (all into the run's JSON):
  train  - every log_every steps: mean train loss, lr, grad norm, step time
  val    - every eval_every steps (+ extra evals right after the switch)
  load   - every load_window steps, per MoE layer: tokens each expert received
  bias   - the balancing bias per layer, at the same times
"""
import json
import math
import os
import time

import numpy as np
import torch

from .model import count_params


# ---------------------------------------------------------------- schedule
def lr_at(step, tc):
    """Warmup -> stable -> linear decay (WSD), on the global step clock."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    decay_start = int(tc.total_steps * (1 - tc.decay_frac))
    if step < decay_start:
        return tc.lr
    frac = (step - decay_start) / max(1, tc.total_steps - decay_start)
    return tc.lr * (1 - frac * (1 - tc.min_lr_frac))


# ---------------------------------------------------------------- optimizer
def make_optimizer(model, tc):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=tc.lr, betas=tc.betas, fused=torch.cuda.is_available())


def carry_optimizer_state(old_model, old_opt, new_model, new_opt):
    """After conversion: attention/embeddings/norms keep their Adam memory (m, v, step).
    New tensors (experts, router, shared expert) start with fresh Adam state.
    Returns the names whose state was carried."""
    old_by_name = dict(old_model.named_parameters())
    carried = []
    for n, p in new_model.named_parameters():
        op = old_by_name.get(n)
        if op is not None and op.shape == p.shape and op in old_opt.state:
            st = old_opt.state[op]
            new_opt.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
            carried.append(n)
    return carried


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, val_batches, device):
    model.eval()
    for m in model.moe_layers():
        m.collect_counts = False
    losses = []
    for x, y in val_batches:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    for m in model.moe_layers():
        m.collect_counts = True
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------- balancing
@torch.no_grad()
def update_balance_bias(model, gamma):
    """DeepSeek-V3 auxiliary-loss-free balancing (arXiv:2412.19437, section 2.1.2):
    expert busier than average -> bias down by gamma; less busy -> bias up by gamma.
    Nothing is added to the loss."""
    for m in model.moe_layers():
        c = m.last_counts.float()
        m.balance_bias += gamma * torch.sign(c.mean() - c)


# ---------------------------------------------------------------- checkpoints
def save_ckpt(path, model, opt, scaler, step, log, extra=None):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "step": step, "log": log, "extra": extra or {}}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    if scaler is not None and ck["scaler"] is not None:
        scaler.load_state_dict(ck["scaler"])
    return ck["step"], ck["log"], ck.get("extra", {})


def new_log(name, model, mc, tc, kind):
    total, active = count_params(model)
    return {"run": name, "kind": kind, "params_total": total, "params_active": active,
            "train": [], "val": [], "load": [], "bias": [], "events": [],
            "model_config": mc.__dict__, "train_config": {k: (list(v) if isinstance(v, tuple) else v)
                                                          for k, v in tc.__dict__.items()}}


# ---------------------------------------------------------------- the loop
def train_segment(model, opt, scaler, data, tc, start, end, log, device, ckpt_path=None,
                  explore_until=-1, extra_eval_steps=(), tag=""):
    """Train global steps [start, end). Returns nothing; appends to `log`."""
    val_batches = data.val_batches(tc.eval_batches)
    is_moe = model.is_moe
    window_counts = None
    t_acc, n_acc, loss_acc = 0.0, 0, 0.0
    extra_eval_steps = set(extra_eval_steps)
    model.train()
    for step in range(start, end):
        # ---- evaluation BEFORE taking the step (so val@step = model that has done `step` updates)
        if step % tc.eval_every == 0 or step in extra_eval_steps:
            if not any(v["step"] == step for v in log["val"]):
                log["val"].append({"step": step, "loss": evaluate(model, val_batches, device)})
        if is_moe:
            model.set_explore(step < explore_until)

        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.train_batch(step)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if is_moe and tc.balance:
            update_balance_bias(model, tc.bias_update_speed)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # ---- bookkeeping
        if is_moe:
            c = torch.stack([m.last_counts for m in model.moe_layers()]).cpu()
            window_counts = c if window_counts is None else window_counts + c
            if (step + 1) % tc.load_window == 0:
                log["load"].append({"step": step + 1, "counts": window_counts.tolist()})
                log["bias"].append({"step": step + 1,
                                    "bias": [m.balance_bias.cpu().tolist() for m in model.moe_layers()]})
                window_counts = None
        lv = loss.item()
        if not math.isfinite(lv):
            log["events"].append({"step": step, "event": "non-finite loss"})
        t_acc += dt; n_acc += 1; loss_acc += lv
        if (step + 1) % tc.log_every == 0:
            log["train"].append({"step": step + 1, "loss": loss_acc / n_acc, "lr": lr,
                                 "grad_norm": gnorm, "sec_per_step": t_acc / n_acc,
                                 "tok_per_sec": data.tokens_per_step() * n_acc / t_acc})
            if (step + 1) % (tc.log_every * 20) == 0:
                last = log["val"][-1]["loss"] if log["val"] else float("nan")
                print(f"{tag} step {step + 1:5d} | train {loss_acc / n_acc:.4f} | val@{log['val'][-1]['step'] if log['val'] else '-'} "
                      f"{last:.4f} | lr {lr:.2e} | {data.tokens_per_step() * n_acc / t_acc:,.0f} tok/s", flush=True)
            t_acc, n_acc, loss_acc = 0.0, 0, 0.0
        if ckpt_path and (step + 1) % tc.ckpt_every == 0 and (step + 1) < end:
            save_ckpt(ckpt_path, model, opt, scaler, step + 1, log)

    # final evaluation at `end`
    if not any(v["step"] == end for v in log["val"]):
        log["val"].append({"step": end, "loss": evaluate(model, val_batches, device)})


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
