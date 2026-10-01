"""
Run everything from the command line (the notebook calls the same functions).

    python run_all.py            # the real experiment (needs a GPU; ~75-90 min on a Colab T4)
    python run_all.py --smoke    # tiny CPU version on synthetic text, ~3 min, checks the plumbing only

Re-running is safe: finished runs are LOADED FROM CACHE, interrupted runs RESUME.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.analysis import make_all          # noqa: E402
from src.config import ModelConfig, TrainConfig, pilot_cpu_configs, small_cpu_configs  # noqa: E402
from src.data import ByteData, prepare_data, write_synthetic_data  # noqa: E402
from src.experiments import run_all       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run on synthetic data")
    ap.add_argument("--pilot", action="store_true", help="scaled-down run on real TinyStories, CPU, ~30 min")
    ap.add_argument("--out", default=None, help="where results/, runs/, figures/ go")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        mc, tc = small_cpu_configs()
        out = args.out or os.path.join(ROOT, "smoke")
        tr, va = write_synthetic_data(os.path.join(out, "data"))
        device = "cpu"
    elif args.pilot:
        mc, tc = pilot_cpu_configs()
        out = args.out or os.path.join(ROOT, "pilot")
        tr, va = prepare_data(args.data or os.path.join(ROOT, "data"))
        device = "cpu"
    else:
        mc, tc = ModelConfig(), TrainConfig()
        out = args.out or ROOT
        if device != "cuda":
            print("WARNING: no GPU found. The full run on CPU would take many hours. Use --smoke to test.")
        tr, va = prepare_data(args.data or os.path.join(out, "data"))
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    data = ByteData(tr, va, mc.block_size, tc.batch_size, tc.seed, device)
    logs = run_all(out, mc, tc, data, device)
    make_all(out, logs, smoke=args.smoke or args.pilot)


if __name__ == "__main__":
    main()
