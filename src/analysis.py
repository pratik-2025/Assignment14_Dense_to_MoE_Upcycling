"""
Turns the run logs into numbers (results/summary.json) and charts (figures/*.png).

Every number in the README comes from summary.json, and summary.json is computed
by re-reading results/*.json from disk - not from whatever happened to be in memory.
"""
import glob
import json
import os

import numpy as np

COL = {"A_dense_cont": "#2a78d6", "B_moe_drop": "#eb6834", "D_moe_clone": "#1baf7a",
       "C_moe_scratch": "#eda100", "dense_phase": "#2a78d6"}
LABEL = {"A_dense_cont": "A  dense, keeps training", "B_moe_drop": "B  MoE, drop-upcycled",
         "D_moe_clone": "D  MoE, cloned slices", "C_moe_scratch": "C  MoE from scratch"}
RUNS = ["A_dense_cont", "B_moe_drop", "D_moe_clone", "C_moe_scratch"]
DEAD_FRAC = 0.1   # an expert getting < 10% of its fair share in a window counts as "dead"


def load_logs(root):
    logs = {}
    for p in glob.glob(os.path.join(root, "results", "*.json")):
        name = os.path.basename(p)[:-5]
        if name in ("summary",):
            continue
        with open(p, "r", encoding="utf-8") as f:
            logs[name] = json.load(f)
    return logs


def full_val_curve(name, logs):
    """Branch runs only log from the switch onward; prepend the dense phase they came from."""
    lg = logs[name]
    pts = sorted((v["step"], v["loss"]) for v in lg["val"])
    if lg.get("branched_from"):
        sw = lg["switch_step"]
        pre = sorted((v["step"], v["loss"]) for v in logs["dense_phase"]["val"] if v["step"] < sw)
        pts = pre + pts
    return np.array(pts)


def val_at(lg, step):
    for v in lg["val"]:
        if v["step"] == step:
            return v["loss"]
    return None


def load_stats(lg):
    """Per logged window: dead experts (summed over layers) and busiest-expert load / fair share."""
    out = []
    for w in lg["load"]:
        c = np.array(w["counts"], dtype=float)          # (layers, experts)
        fair = c.sum(axis=1, keepdims=True) / c.shape[1]
        ratio = c / fair
        out.append({"step": w["step"], "dead": int((ratio < DEAD_FRAC).sum()),
                    "never": int((c == 0).sum()), "max_ratio": float(ratio.max()),
                    "cv": float((c.std(axis=1) / c.mean(axis=1)).mean())})
    return out


def seg_speed(lg, start=None):
    rows = [t for t in lg["train"] if start is None or t["step"] > start]
    if not rows:
        return None, None
    tps = np.median([t["tok_per_sec"] for t in rows])
    secs = sum(t["sec_per_step"] for t in rows) * lg["train_config"]["log_every"]
    return float(tps), float(secs)


def summarize(logs):
    d = logs["dense_phase"]
    tc = d["train_config"]
    sw, total = tc["switch_step"], tc["total_steps"]
    S = {"switch_step": sw, "total_steps": total,
         "tokens_per_step": tc["batch_size"] * d["model_config"]["block_size"],
         "dense_params": d["params_total"], "dense_val_at_switch": val_at(d, sw),
         "dense_phase_tok_per_sec": seg_speed(d)[0], "dense_phase_gpu_seconds": seg_speed(d)[1],
         "runs": {}}
    for name in RUNS:
        if name not in logs:
            continue
        lg = logs[name]
        r = {"params_total": lg["params_total"], "params_active": lg["params_active"],
             "final_val": val_at(lg, total),
             "val_by_step": {str(v["step"]): v["loss"] for v in lg["val"] if v["step"] % 500 == 0}}
        r["tok_per_sec"], r["gpu_seconds_own_segment"] = seg_speed(lg)
        r["gpu_seconds_total"] = r["gpu_seconds_own_segment"] + (S["dense_phase_gpu_seconds"] if lg.get("branched_from") else 0)
        if lg.get("branched_from") and lg["kind"].startswith("moe"):
            r["val_right_after_switch"] = val_at(lg, sw)
            r["jump"] = r["val_right_after_switch"] - S["dense_val_at_switch"]
            rec = [v["step"] for v in sorted(lg["val"], key=lambda v: v["step"])
                   if v["step"] > sw and v["loss"] <= S["dense_val_at_switch"]]
            r["steps_to_recover"] = (rec[0] - sw) if rec else None
            r["post_switch_curve"] = [(v["step"], v["loss"]) for v in sorted(lg["val"], key=lambda v: v["step"])
                                      if sw <= v["step"] <= sw + 300]
        if lg["load"]:
            ls = load_stats(lg)
            r["load_first_window"] = ls[0]
            r["load_final_window"] = ls[-1]
            r["dead_experts_peak"] = max(x["dead"] for x in ls)
            pk = max(ls, key=lambda x: x["max_ratio"])
            r["max_ratio_peak"], r["max_ratio_peak_step"] = pk["max_ratio"], pk["step"]
            pd = max(ls, key=lambda x: x["dead"])
            r["dead_experts_peak_step"] = pd["step"]
            born = sw if lg.get("branched_from") else 0
            r["explore_end_step"] = born + lg["train_config"]["explore_steps"]
            during = [x["max_ratio"] for x in ls if x["step"] <= r["explore_end_step"]]
            r["max_ratio_during_explore"] = max(during) if during else None
            # "collapse": windows where the busiest expert is near the maximum possible load
            # (every token picks it -> E/k times its fair share)
            E, k = lg["model_config"]["n_experts"], lg["model_config"]["top_k"]
            r["max_ratio_possible"] = E / k
            col = [x["step"] for x in ls if x["max_ratio"] >= 0.9 * E / k]
            r["collapse_first_step"] = col[0] if col else None
            r["collapse_last_step"] = col[-1] if col else None
            r["n_expert_slots"] = len(lg["load"][0]["counts"]) * len(lg["load"][0]["counts"][0])
            b = np.array(lg["bias"][-1]["bias"])
            r["bias_final_min"], r["bias_final_max"] = float(b.min()), float(b.max())
        for k in ("expert_similarity_at_birth", "expert_similarity_final"):
            if k in lg:
                r[k + "_max"] = float(np.mean([x["max"] for x in lg[k]]))
                r[k + "_mean"] = float(np.mean([x["mean"] for x in lg[k]]))
        S["runs"][name] = r
    R = S["runs"]
    if "A_dense_cont" in R:
        Av = {v["step"]: v["loss"] for v in logs["A_dense_cont"]["val"]}
        for n in ("B_moe_drop", "D_moe_clone", "C_moe_scratch"):
            if n in R:
                R[n]["gap_vs_A_by_step"] = {str(v["step"]): v["loss"] - Av[v["step"]] for v in logs[n]["val"]
                                            if v["step"] in Av and v["step"] % 500 == 0}
        if "B_moe_drop" in R:
            S["dense_speed_over_moe"] = R["A_dense_cont"]["tok_per_sec"] / R["B_moe_drop"]["tok_per_sec"]
        for n in ("B_moe_drop", "D_moe_clone", "C_moe_scratch"):
            if n in R:
                R[n]["final_gap_vs_A"] = R[n]["final_val"] - R["A_dense_cont"]["final_val"]
    if "C_moe_scratch" in R and "B_moe_drop" in R:
        target = R["B_moe_drop"]["final_val"]
        hit = [v["step"] for v in sorted(logs["C_moe_scratch"]["val"], key=lambda v: v["step"]) if v["loss"] <= target]
        S["scratch_steps_to_match_B"] = hit[0] if hit else None
        target_a = R["A_dense_cont"]["final_val"] if "A_dense_cont" in R else None
        hit_a = [v["step"] for v in sorted(logs["C_moe_scratch"]["val"], key=lambda v: v["step"])
                 if target_a is not None and v["loss"] <= target_a]
        S["scratch_steps_to_match_A"] = hit_a[0] if hit_a else None
        # wall-clock view: how many GPU-seconds did C need to reach B's final loss?
        if hit:
            cl = logs["C_moe_scratch"]["train"]
            S["scratch_gpu_seconds_to_match_B"] = float(sum(t["sec_per_step"] for t in cl if t["step"] <= hit[0])
                                                        * logs["C_moe_scratch"]["train_config"]["log_every"])
        else:
            S["scratch_gpu_seconds_to_match_B"] = None
    return S


# ----------------------------------------------------------------- figures
def _style(plt):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                         "font.size": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "legend.frameon": False})


def make_figures(root, logs, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_dir = os.path.join(root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sw, total = S["switch_step"], S["total_steps"]
    present = [n for n in RUNS if n in logs]

    # ---- 1. loss curves: full run + zoom on the switch
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8.5))
    for n in present:
        c = full_val_curve(n, logs)
        if logs[n].get("branched_from"):
            post = c[c[:, 0] >= sw]
            a1.plot(post[:, 0], post[:, 1], color=COL[n], lw=2, label=LABEL[n])
        else:
            a1.plot(c[:, 0], c[:, 1], color=COL[n], lw=2, label=LABEL[n])
    dp = full_val_curve("dense_phase", logs)
    a1.plot(dp[:, 0], dp[:, 1], color=COL["dense_phase"], lw=2)
    a1.axvline(sw, color="#52514e", lw=1, ls="--")
    lo = min(full_val_curve(n, logs)[:, 1].min() for n in present)
    hi = dp[dp[:, 0] >= sw // 2, 1].max() if len(dp) else lo + 1
    a1.set_ylim(lo - 0.03, hi + 0.05)
    a1.text(sw, 0.98, "  dense -> MoE", transform=a1.get_xaxis_transform(), va="top", ha="left",
            color="#52514e", fontsize=9)
    a1.set_title("Validation loss over the whole budget (y-axis cut to the second half)")
    a1.set_xlabel("step"); a1.set_ylabel("val loss (nats/byte)")
    a1.legend(loc="upper right")
    win = int(0.15 * total)
    dsw = S["dense_val_at_switch"]
    pre = dp[(dp[:, 0] >= sw - win // 3) & (dp[:, 0] <= sw)]
    a2.plot(pre[:, 0], pre[:, 1], color=COL["dense_phase"], lw=2, marker="o", ms=4)
    top = dsw + 0.15
    off_chart = []
    for n in present:
        c = full_val_curve(n, logs)
        lo_s = sw if logs[n].get("branched_from") else sw - win // 3
        m = (c[:, 0] >= lo_s) & (c[:, 0] <= sw + win)
        a2.plot(c[m, 0], np.minimum(c[m, 1], top), color=COL[n], lw=2, marker="o", ms=4, label=LABEL[n])
        if logs[n].get("branched_from") and c[m, 1].max() > top:
            off_chart.append((n, c[m, 1].max()))
    for i, (n, v) in enumerate(off_chart):
        a2.annotate(f"{LABEL[n][:1]}: {v:.2f} at step {sw} (off the chart)", (sw, top), xytext=(60, -14 - 13 * i),
                    textcoords="offset points", fontsize=8, color="#0b0b0b")
    a2.set_ylim(min(full_val_curve(n, logs)[:, 1][(full_val_curve(n, logs)[:, 0] >= sw - win // 3)
                                                     & (full_val_curve(n, logs)[:, 0] <= sw + win)].min()
                    for n in present) - 0.01, top + 0.01)
    a2.axvline(sw, color="#52514e", lw=1, ls="--")
    a2.axhline(dsw, color="#9a9893", lw=1, ls=":")
    a2.text(0.99, dsw, "dense loss at the switch ", transform=a2.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#52514e")
    a2.set_title("Zoom on the switch: the jump (capped, values noted), and the climb back")
    a2.set_xlabel("step"); a2.set_ylabel("val loss (nats/byte)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig1_loss_curves.png"), dpi=140); plt.close(fig)

    # ---- 2. difference vs the dense control
    if "A_dense_cont" in logs:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        A = dict(map(tuple, full_val_curve("A_dense_cont", logs)))
        for n in [x for x in present if x != "A_dense_cont"]:
            c = full_val_curve(n, logs)
            pts = [(s, l - A[s]) for s, l in c if s >= sw + 100 and s in A]
            if pts:
                p = np.array(pts)
                ax.plot(p[:, 0], p[:, 1], color=COL[n], lw=2, label=LABEL[n])
                ax.annotate(f"{p[-1, 1]:+.3f}", (p[-1, 0], p[-1, 1]), xytext=(4, {"B_moe_drop": 10, "D_moe_clone": -2, "C_moe_scratch": -9}.get(n, 0)),
                            textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
        ax.axhline(0, color="#52514e", lw=1)
        ax.set_title(f"Val loss minus run A, from step {sw + 100} (below zero = better than staying dense)")
        ax.set_xlabel("step"); ax.set_ylabel("loss difference (nats/byte)")
        ax.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig2_gap_vs_dense.png"), dpi=140); plt.close(fig)

    # ---- 3. expert health over time
    moe_runs = [n for n in present if logs[n]["load"]]
    if moe_runs:
        fig, (b1, b2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for n in moe_runs:
            ls = load_stats(logs[n])
            st = [x["step"] for x in ls]
            b1.plot(st, [x["dead"] for x in ls], color=COL[n], lw=2, label=LABEL[n])
            b2.plot(st, [x["max_ratio"] for x in ls], color=COL[n], lw=2, label=LABEL[n])
        slots = S["runs"][moe_runs[0]]["n_expert_slots"]
        b1.set_title(f"'Dead' experts (< {int(DEAD_FRAC * 100)}% of fair share in a window), out of {slots} (all layers)")
        b1.set_ylabel("dead experts"); b1.legend(loc="upper right")
        b2.axhline(1, color="#52514e", lw=1, ls=":")
        b2.set_title("Busiest expert's load / fair share (1.0 = perfectly even)")
        b2.set_ylabel("max load ratio"); b2.set_xlabel("step")
        for ax in (b1, b2):
            ax.axvline(sw, color="#52514e", lw=1, ls="--")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_expert_health.png"), dpi=140); plt.close(fig)

        # ---- 4. final load heatmaps
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])
        fig, axes = plt.subplots(len(moe_runs), 1, figsize=(9, 2.2 * len(moe_runs) + 0.8), gridspec_kw={"hspace": 0.6})
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, moe_runs):
            ls = load_stats(logs[n])
            wi = max(range(len(ls)), key=lambda i: ls[i]["dead"])
            c = np.array(logs[n]["load"][wi]["counts"], dtype=float)
            ratio = c / (c.sum(axis=1, keepdims=True) / c.shape[1])
            im = ax.imshow(np.log2(np.clip(ratio, 1 / 16, 16)), aspect="auto", cmap=cmap,
                           norm=TwoSlopeNorm(vcenter=0, vmin=-4, vmax=4))
            ax.set_title(f"{LABEL[n]}: expert load in its worst window (ending step {ls[wi]['step']}, {ls[wi]['dead']} dead)", fontsize=10)
            ax.set_ylabel("layer"); ax.grid(False)
            ax.set_yticks(range(c.shape[0])); ax.set_yticklabels([str(i + 1) for i in range(c.shape[0])])
            ax.set_xticks(range(0, c.shape[1], max(1, c.shape[1] // 16)))
        axes[-1].set_xlabel("expert")
        cb = fig.colorbar(im, ax=list(axes), shrink=0.8)
        cb.set_ticks([-4, -2, 0, 2, 4]); cb.set_ticklabels(["1/16x", "1/4x", "fair", "4x", "16x"])
        fig.savefig(os.path.join(fig_dir, "fig4_worst_load_heatmap.png"), dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- 5. speed
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = ["dense_phase"] + present
    vals = [S["dense_phase_tok_per_sec"]] + [S["runs"][n]["tok_per_sec"] for n in present]
    labels = ["dense phase"] + [LABEL[n] for n in present]
    cols = [COL["dense_phase"]] + [COL[n] for n in present]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:,.0f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("training tokens per second (median)")
    ax.set_title("Throughput: same active compute, different speed")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig5_throughput.png"), dpi=140); plt.close(fig)


def summary_table(S):
    lines = ["| run | total params | active params | final val loss | vs A | jump at switch | steps to recover | dead experts (final / peak) | tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in RUNS:
        if n not in S["runs"]:
            continue
        r = S["runs"][n]
        gap = r.get("final_gap_vs_A")
        jump = r.get("jump")
        rec = r.get("steps_to_recover", "")
        dead = (f"{r['load_final_window']['dead']} / {r['dead_experts_peak']}"
                if "load_final_window" in r else "-")
        lines.append(f"| {LABEL[n]} | {r['params_total']:,} | {r['params_active']:,} | {r['final_val']:.4f} | "
                     f"{'-' if gap is None else f'{gap:+.4f}'} | {'-' if jump is None else f'{jump:+.4f}'} | "
                     f"{'-' if 'jump' not in r else ('not within budget' if rec is None else rec)} | {dead} | {r['tok_per_sec']:,.0f} |")
    return "\n".join(lines)


def make_all(root, logs=None, smoke=False):
    logs = load_logs(root)                      # always re-read from disk
    S = summarize(logs)
    S["smoke_test"] = smoke
    with open(os.path.join(root, "results", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1)
    make_figures(root, logs, S)
    table = summary_table(S)
    with open(os.path.join(root, "results", "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    return S
