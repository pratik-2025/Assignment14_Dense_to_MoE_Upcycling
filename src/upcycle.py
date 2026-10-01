"""
Turning a trained dense model into an MoE model.

A dense feed-forward block has `dense_hidden` neurons. Neuron j is:
    fc.weight[j, :]     (what it reads from the token)
    proj.weight[:, j]   (what it writes back)
The block's output is simply the SUM of all neurons' contributions. That is the
fact everything below relies on: if you split the neurons into groups and add
the groups' outputs back together, you get the dense output exactly.

Three ways to build experts (all three are in the Session 14 transcript):

  partition / clone  (run D)
      Shared expert = a random 384 of the 1536 neurons, copied exactly.
      The other 1152 neurons are cut into 6 slices of 192. The 32 experts are
      clones of those 6 slices (5-6 clones each). Nothing is redrawn.
      -> Rohan's published V4 failure case: identical clones the router can't tell apart.

  drop-upcycling  (run B, the main run)
      Shared expert: same as above.
      Each routed expert takes a random 192 neurons from the 1152, then REDRAWS
      a fraction r (=0.5) of them with fresh random weights (with the same mean
      and spread as the dense weights). Experts start different from each other.
      Reference: Nakamura et al., "Drop-Upcycling", ICLR 2025 (arXiv:2502.19261).

  full copy  (used only in a unit test)
      Every expert is a complete copy of the dense block, no shared expert. With
      softmax gates renormalised to sum to 1, the MoE output equals the dense
      output exactly at the moment of conversion - but every expert is identical.
      This needs experts as wide as the whole dense block, so it is not the
      architecture used in the runs.

The router is new in every case (small random weights) and the balancing bias starts at 0.
"""
import torch

from .model import GPT


def _neurons(dense_mlp):
    w_fc = dense_mlp.fc.weight.detach()        # (H, C)
    w_proj = dense_mlp.proj.weight.detach()    # (C, H)
    return w_fc, w_proj


def _copy_shared(moe, w_fc, w_proj, idx):
    moe.shared.fc.weight.data.copy_(w_fc[idx])
    moe.shared.proj.weight.data.copy_(w_proj[:, idx])


def _set_expert(moe, e, w_fc_rows, w_proj_cols):
    # w_fc_rows: (He, C) -> w_in[e] is (C, He);  w_proj_cols: (C, He) -> w_out[e] is (He, C)
    moe.w_in.data[e].copy_(w_fc_rows.t())
    moe.w_out.data[e].copy_(w_proj_cols.t())


def _copy_shared_layers(dense, moe_model):
    """Everything that is not a feed-forward block is copied 1:1."""
    dsd = dense.state_dict()
    msd = moe_model.state_dict()
    copied = []
    for k, v in dsd.items():
        if ".ffn." in k:
            continue
        msd[k].copy_(v)
        copied.append(k)
    return copied


def upcycle(dense, cfg, method, drop_ratio=0.5, seed=0):
    """Returns a new MoE GPT built from `dense`. method in {"drop", "clone"}."""
    assert method in ("drop", "clone")
    moe_model = GPT(cfg, moe=True).to(next(dense.parameters()).device)
    _copy_shared_layers(dense, moe_model)
    g = torch.Generator().manual_seed(seed)
    H, Hs, He, E = cfg.dense_hidden, cfg.shared_hidden, cfg.expert_hidden, cfg.n_experts
    for li, (db, mb) in enumerate(zip(dense.blocks, moe_model.blocks)):
        w_fc, w_proj = _neurons(db.ffn)
        moe = mb.ffn
        perm = torch.randperm(H, generator=g).to(w_fc.device)
        shared_idx, pool = perm[:Hs], perm[Hs:]
        _copy_shared(moe, w_fc, w_proj, shared_idx)
        if method == "clone":
            assert len(pool) % He == 0, "pool must split into whole slices"
            n_slices = len(pool) // He
            for e in range(E):
                s = e % n_slices
                idx = pool[s * He:(s + 1) * He]
                _set_expert(moe, e, w_fc[idx], w_proj[:, idx])
        else:
            n_redraw = int(round(drop_ratio * He))
            for e in range(E):
                pick = pool[torch.randperm(len(pool), generator=g)[:He].to(pool.device)]
                rows = w_fc[pick].clone()
                cols = w_proj[:, pick].clone()
                redraw = torch.randperm(He, generator=g)[:n_redraw]
                # as in the paper: the new values use the mean/std of the weights being replaced
                fc_mean, fc_std = rows[redraw].mean().item(), rows[redraw].std().item()
                pj_mean, pj_std = cols[:, redraw].mean().item(), cols[:, redraw].std().item()
                rows[redraw] = (torch.randn(n_redraw, rows.shape[1], generator=g) * fc_std + fc_mean).to(rows)
                cols[:, redraw] = (torch.randn(cols.shape[0], n_redraw, generator=g) * pj_std + pj_mean).to(cols)
                _set_expert(moe, e, rows, cols)
    return moe_model


def full_copy_moe(dense, cfg):
    """Test-only: every expert = the whole dense block, no shared expert.
    cfg must have expert_hidden == dense_hidden and shared_hidden == 0."""
    assert cfg.expert_hidden == cfg.dense_hidden and cfg.shared_hidden == 0
    moe_model = GPT(cfg, moe=True).to(next(dense.parameters()).device)
    _copy_shared_layers(dense, moe_model)
    for db, mb in zip(dense.blocks, moe_model.blocks):
        w_fc, w_proj = _neurons(db.ffn)
        for e in range(cfg.n_experts):
            _set_expert(mb.ffn, e, w_fc, w_proj)
    return moe_model
