"""
Checks that the construction is correct BEFORE any number is trusted.
Run:  python -m tests.test_all      (from the repo root; CPU, ~10 s)

The two most important checks:
  * partition identity - if the router picks every slice once with weight 1, the
    upcycled MoE gives the SAME output as the dense model. This proves the neurons
    were copied into the experts correctly (wrong row/column = big mismatch).
  * full-copy identity - every expert a whole copy of the dense block + softmax gates
    renormalised to sum to 1 -> output identical to dense, whatever the router says.
"""
import os
import sys
import tempfile
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ModelConfig, small_cpu_configs
from src.data import ByteData, write_synthetic_data
from src.model import GPT, count_params
from src.train import carry_optimizer_state, lr_at, make_optimizer, update_balance_bias
from src.upcycle import full_copy_moe, upcycle

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def trained_tiny_dense(mc, steps=30):
    torch.manual_seed(0)
    m = GPT(mc, moe=False)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):  # a few steps so weights are not just their init
        x = torch.randint(0, 256, (4, mc.block_size))
        _, loss = m(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m, opt


def main():
    torch.set_grad_enabled(True)
    mc_full = ModelConfig()

    # 1 - sizes of the real configuration
    dense = GPT(mc_full, moe=False)
    moe = GPT(mc_full, moe=True)
    dt, _ = count_params(dense)
    mt, ma = count_params(moe)
    ffn_dense = mc_full.dense_hidden
    ffn_active = mc_full.shared_hidden + mc_full.top_k * mc_full.expert_hidden
    ffn_total = mc_full.shared_hidden + mc_full.n_experts * mc_full.expert_hidden
    check("active FFN neurons per token equal dense", ffn_active == ffn_dense, f"{ffn_active} vs {ffn_dense}")
    check("total FFN neurons = 4.25x dense", ffn_total / ffn_dense == 4.25, f"{ffn_total}")
    router = mc_full.n_layer * mc_full.n_embd * mc_full.n_experts
    check("MoE active params = dense + router only", ma - dt == router, f"dense {dt:,} active {ma:,} total {mt:,}")
    check("clone slicing fits exactly", (mc_full.dense_hidden - mc_full.shared_hidden) % mc_full.expert_hidden == 0)
    del dense, moe

    # tiny config for the rest
    mc, tc = small_cpu_configs()
    dense, dopt = trained_tiny_dense(mc)
    x = torch.randint(0, 256, (3, mc.block_size))
    with torch.no_grad():
        ref = dense(x)

    # 2 - partition identity
    n_slices = (mc.dense_hidden - mc.shared_hidden) // mc.expert_hidden
    pc = replace(mc, n_experts=n_slices, top_k=n_slices, routed_scale=float(n_slices))
    pm = upcycle(dense, pc, "clone", seed=3)
    for m in pm.moe_layers():
        m.router.weight.data.zero_()       # all scores 0.5 -> each gate = 1/n * n = 1
    pm.eval()
    with torch.no_grad():
        out = pm(x)
    err = (out - ref).abs().max().item()
    check("partition identity: shared + every slice once = dense output", err < 1e-4, f"max abs diff {err:.2e}")

    # 3 - full-copy identity with softmax renormalised top-2, RANDOM router
    fc = replace(mc, n_experts=4, top_k=2, expert_hidden=mc.dense_hidden, shared_hidden=0,
                 score_fn="softmax", routed_scale=1.0)
    fm = full_copy_moe(dense, fc)
    fm.eval()
    with torch.no_grad():
        out = fm(x)
    err = (out - ref).abs().max().item()
    check("full-copy + softmax top-2 = dense output exactly", err < 1e-4, f"max abs diff {err:.2e}")
    # 4 - drop-upcycling: kept neurons are real dense neurons, redrawn ones are not
    dm = upcycle(dense, mc, "drop", drop_ratio=0.5, seed=5)
    w_fc = dense.blocks[0].ffn.fc.weight.detach()                     # (H, C)
    w_in = dm.blocks[0].ffn.w_in.detach()                              # (E, C, He)
    matches = []
    for e in range(mc.n_experts):
        cols = w_in[e].t()                                             # (He, C)
        d = torch.cdist(cols, w_fc, compute_mode="donot_use_mm_for_euclid_dist")                                    # distance to every dense neuron
        matches.append(int((d.min(dim=1).values < 1e-6).sum()))
    expect = mc.expert_hidden - int(round(0.5 * mc.expert_hidden))
    check("drop-upcycling keeps exactly (1-r) of each expert's neurons", all(m == expect for m in matches),
          f"kept per expert {matches[:4]}..., expected {expect}")
    sh = dm.blocks[0].ffn.shared.fc.weight.detach()
    d = torch.cdist(sh, w_fc, compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values
    check("shared expert is an exact copy of dense neurons", bool((d < 1e-6).all()))
    # non-FFN weights copied 1:1
    same = all(torch.equal(a, b) for (na, a), (nb, b) in zip(dense.state_dict().items(), dm.state_dict().items())
               if ".ffn." not in na and na == nb)
    check("attention/embeddings/norms copied exactly", same)

    # 5 - dropless routing + explore picks distinct experts
    layer = dm.blocks[0].ffn
    layer.train()
    for explore in (False, True):
        layer.explore = explore
        h = torch.randn(2, 16, mc.n_embd)
        layer(h)
        c = layer.last_counts
        check(f"dropless: every token gets k experts (explore={explore})", int(c.sum()) == 32 * mc.top_k)
        s = layer.scores(h.reshape(-1, mc.n_embd))
        idx = layer.select(s)
        distinct = all(len(set(r.tolist())) == mc.top_k for r in idx)
        check(f"k distinct experts per token (explore={explore})", distinct)
    layer.explore = False

    # 6 - balancing bias changes WHO is picked, not the gate weight
    layer.balance_bias.zero_()
    layer.balance_bias[0] = 100.0
    h = torch.randn(1, 4, mc.n_embd)
    s = layer.scores(h.reshape(-1, mc.n_embd))
    idx = layer.select(s)
    check("big bias forces expert 0 to be picked", bool((idx == 0).any(dim=1).all()))
    w = s.gather(1, idx)
    check("gate weights come from scores only (bias not inside)", bool((w <= 1.0).all()))
    layer.balance_bias.zero_()

    # 7 - balancing direction
    for m in dm.moe_layers():
        m.last_counts = torch.tensor([100] + [10] * (mc.n_experts - 1))
    update_balance_bias(dm, 0.001)
    b = dm.blocks[0].ffn.balance_bias
    check("busy expert's bias goes down, idle experts' go up",
          b[0].item() < 0 and bool((b[1:] > 0).all()), f"{b[0].item():+.4f} / {b[1].item():+.4f}")

    # 8 - optimizer state carried for non-FFN params only
    xx = torch.randint(0, 256, (2, mc.block_size))
    _, loss = dense(xx, xx); loss.backward(); dopt.step()
    nopt = make_optimizer(dm, tc)
    carried = carry_optimizer_state(dense, dopt, dm, nopt)
    expect_names = [n for n, _ in dm.named_parameters() if ".ffn." not in n]
    check("Adam state carried for every non-FFN tensor and nothing else",
          sorted(carried) == sorted(expect_names), f"{len(carried)} tensors")

    # 9 - schedule and data determinism
    check("WSD: warmup ramps, flat in middle, decays at end",
          lr_at(0, tc) < lr_at(tc.warmup_steps, tc) == lr_at(tc.switch_step, tc) > lr_at(tc.total_steps - 1, tc))
    with tempfile.TemporaryDirectory() as d:
        tr, va = write_synthetic_data(d, 50_000, 10_000)
        data = ByteData(tr, va, mc.block_size, 4, 1, "cpu")
        a, _ = data.train_batch(7)
        b2, _ = data.train_batch(7)
        c2, _ = data.train_batch(8)
        check("same step -> same batch in every run; different step -> different batch",
              torch.equal(a, b2) and not torch.equal(a, c2))
        del data

    # 10 - a training step through the MoE actually moves the router
    r0 = dm.blocks[0].ffn.router.weight.detach().clone()
    o = torch.optim.SGD(dm.parameters(), lr=0.1)
    dm.train()
    _, loss = dm(xx, xx); o.zero_grad(); loss.backward(); o.step()
    check("router receives gradient through the gate weights",
          not torch.equal(r0, dm.blocks[0].ffn.router.weight.detach()))

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS)} checks, {n_fail} failures")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
