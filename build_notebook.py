"""
Builds session14_dense_to_moe.ipynb from the cells below (so the notebook is never hand-edited).

    python build_notebook.py
"""
import json
import os

REPO_URL = "https://github.com/pratik-2025/Assignment14_Dense_to_MoE_Upcycling"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r"""
# Session 14 — Train a dense model, convert it into a Mixture of Experts, keep training

**What this notebook does.** It trains a ~10.8M-parameter dense GPT on TinyStories (read as raw bytes)
for 3,000 steps, then branches it four ways for the remaining 3,000 steps:

| run | what it is | why it is here |
|---|---|---|
| **A** | the dense model keeps training | control: is converting better than not converting? |
| **B** | dense → MoE by **drop-upcycling** (half of each expert's neurons redrawn) | the assignment |
| **D** | dense → MoE by **cloning slices** (no neurons redrawn) | Rohan's V4 failure case |
| **C** | the same MoE trained **from scratch** for all 6,000 steps | does starting from the dense model save compute? |

**How to run it.** Runtime → Change runtime type → **T4 GPU**. Then run the cells **one at a time with
Shift+Enter** and read each cell's output before moving on (not "Run all" — that hid problems in Session 13).

**Every training cell prints one of three words first:**
* `TRAINING` — nothing on disk, it is training now.
* `RESUMING` — it found a checkpoint (e.g. after a disconnect) and continues from it.
* `LOADED FROM CACHE` — the finished result is already on disk, **nothing was trained**, so the cell
  finishes in a second. Delete `results/<run>.json` if you want to force a re-run.

Total time on a T4: roughly 75–90 minutes (the dense parts are fast, the MoE parts are slower).
""")

code(r"""
# --- 1. Settings -------------------------------------------------------------------------
USE_DRIVE = True     # True = results and checkpoints live on Google Drive, so a disconnect
                     # or a new runtime does not lose them. Strongly recommended.
REPO_URL = "%s"
""" % REPO_URL)

code(r"""
# --- 2. Get the code, pick where results go ---------------------------------------------
import os, sys, subprocess
if not os.path.exists("/content/repo"):
    subprocess.run(["git", "clone", "-q", REPO_URL, "/content/repo"], check=True)
os.chdir("/content/repo"); sys.path.insert(0, "/content/repo")

if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/session14_dense_to_moe"
else:
    OUT = "/content/repo"
os.makedirs(OUT, exist_ok=True)
print("results will be written to:", OUT)

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to T4")
""")

md(r"""
## Step 1 — check the construction before trusting any number
20 checks. The two that matter most: **partition identity** (if the router picks every slice once
with weight 1, the MoE gives exactly the dense output — proves the neurons were copied into the
right places) and **full-copy identity**. All must say PASS.
""")
code(r"""
!python -m tests.test_all
""")

md("## Step 2 — data (first 200 MB of TinyStories train + the full validation file, ~1 minute)")
code(r"""
from src.config import ModelConfig, TrainConfig
from src.data import ByteData, prepare_data
from src.model import GPT, count_params
mc, tc = ModelConfig(), TrainConfig()
tr, va = prepare_data(os.path.join(OUT, "data"))
data = ByteData(tr, va, mc.block_size, tc.batch_size, tc.seed, "cuda")
d_total, _ = count_params(GPT(mc, moe=False)); m_total, m_active = count_params(GPT(mc, moe=True))
print(f"dense params: {d_total:,}   MoE params: total {m_total:,}, active per token {m_active:,}")
print(f"tokens per step: {data.tokens_per_step():,}   steps: {tc.total_steps}  (switch at {tc.switch_step})")
""")

md("## Step 3 — the dense phase (steps 0 → 3,000), shared by A, B and D  ·  ~6 min")
code(r"""
from src.experiments import run_dense_phase, run_branch, run_scratch
dense_log, switch_ckpt = run_dense_phase(OUT, mc, tc, data, "cuda")
print("dense val loss at the switch:", [v["loss"] for v in dense_log["val"] if v["step"] == tc.switch_step][0])
""")

md("## Step 4 — run A: the dense model just keeps training (control)  ·  ~6 min")
code(r"""
log_A = run_branch(OUT, "A_dense_cont", None, mc, tc, data, "cuda", switch_ckpt, dense_log)
""")

md("## Step 5 — run B: convert by drop-upcycling, keep training (the main run)  ·  ~15 min")
code(r"""
log_B = run_branch(OUT, "B_moe_drop", "drop", mc, tc, data, "cuda", switch_ckpt, dense_log)
print("val right after conversion:", [v["loss"] for v in log_B["val"] if v["step"] == tc.switch_step][0])
""")

md("## Step 6 — run D: convert by cloning slices (no redraw)  ·  ~15 min")
code(r"""
log_D = run_branch(OUT, "D_moe_clone", "clone", mc, tc, data, "cuda", switch_ckpt, dense_log)
""")

md("## Step 7 — run C: the same MoE from scratch, full 6,000 steps  ·  ~30 min")
code(r"""
log_C = run_scratch(OUT, "C_moe_scratch", mc, tc, data, "cuda")
""")

md("## Step 8 — numbers and charts (re-read from disk, not from memory)")
code(r"""
from src.analysis import make_all
from IPython.display import Image, display
S = make_all(OUT)
for f in ["fig1_loss_curves", "fig2_gap_vs_dense", "fig3_expert_health", "fig4_final_load_heatmap", "fig5_throughput"]:
    display(Image(os.path.join(OUT, "figures", f + ".png")))
""")

md("## Step 9 — download the results bundle and send it back (results/*.json + figures)")
code(r"""
import zipfile
# only results/ and figures/ - small; checkpoints and data stay behind
with zipfile.ZipFile("/content/session14_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("results", "figures"):
        for root, _, files in os.walk(os.path.join(OUT, sub)):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, OUT))
print(os.path.getsize("/content/session14_results.zip") // 1024, "KB")
from google.colab import files
files.download("/content/session14_results.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:02d}"
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session14_dense_to_moe.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", path, "with", len(cells), "cells")
