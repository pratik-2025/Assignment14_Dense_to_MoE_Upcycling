"""
All the knobs for the experiment, in one place.

Plain-English summary
---------------------
* The model reads raw bytes (vocabulary = 256). No tokenizer to download, and it
  means almost all of the model's weights sit in the transformer blocks - which is
  where the MoE conversion happens.
* The DENSE model (Rohan calls it the "linear" model) has one feed-forward block
  per layer: 384 -> 1536 -> 384.
* The MOE model replaces that block with:
    - 1 SHARED expert, 384 neurons wide, always on
    - 32 ROUTED experts, 192 neurons wide each, the router picks 6 per token
  Active feed-forward neurons per token = 384 + 6 x 192 = 1536  (same as dense)
  Total feed-forward neurons            = 384 + 32 x 192 = 6528 (4.25x dense)
  So every token pays the same compute as the dense model, but the model owns
  4.25x more feed-forward weights.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 256          # bytes
    block_size: int = 256          # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dense_hidden: int = 1536       # 4 x n_embd
    # --- MoE shape ---
    n_experts: int = 32
    top_k: int = 6
    expert_hidden: int = 192
    shared_hidden: int = 384       # 0 means "no shared expert"
    score_fn: str = "sigmoid"      # "sigmoid" or "softmax"
    routed_scale: float = 6.0      # gate weights are normalised to sum to 1, then x this
                                   # (= top_k, so each picked expert gets weight ~1 on average)


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 64
    total_steps: int = 6000        # the whole budget, for every run
    switch_step: int = 3000        # where dense becomes MoE
    lr: float = 1e-3
    min_lr_frac: float = 0.1       # WSD decays to 10% of peak
    warmup_steps: int = 200
    decay_frac: float = 0.2        # last 20% of total_steps is the linear decay ("D" of WSD)
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    # --- router / balancing ---
    balance: bool = True           # DeepSeek auxiliary-loss-free bias balancing
    bias_update_speed: float = 1e-3  # gamma in the DeepSeek-V3 paper
    explore_steps: int = 300       # probabilistic top-k for this many steps after experts are born
    drop_ratio: float = 0.5        # drop-upcycling: fraction of each expert's neurons redrawn
    # --- logging ---
    log_every: int = 10
    eval_every: int = 100
    eval_batches: int = 20
    load_window: int = 100         # expert-load statistics are summed over this many steps
    ckpt_every: int = 500
    # extra evals right after the switch, to see the jump and the recovery
    post_switch_evals: list = field(default_factory=lambda: [0, 10, 25, 50, 100, 200])


def small_cpu_configs():
    """A tiny version of everything, used for the CPU smoke test and the unit tests.
    Same code path, same proportions, just small."""
    m = ModelConfig(block_size=64, n_layer=2, n_head=2, n_embd=64, dense_hidden=256,
                    n_experts=8, top_k=2, expert_hidden=64, shared_hidden=128,
                    routed_scale=2.0)
    t = TrainConfig(batch_size=16, total_steps=240, switch_step=120, warmup_steps=20,
                    explore_steps=30, log_every=5, eval_every=20, eval_batches=4,
                    load_window=20, ckpt_every=60, post_switch_evals=[0, 5, 10, 20, 40])
    return m, t


def to_dict(cfg):
    return asdict(cfg)


def pilot_cpu_configs():
    """A scaled-down version on REAL TinyStories that fits a 2-core CPU in ~30 min.
    Same proportions as the real run (shared = n_embd, 32 experts, top-6,
    active FFN = dense FFN). Used to de-risk the design before spending GPU time."""
    m = ModelConfig(block_size=128, n_layer=4, n_head=4, n_embd=128, dense_hidden=512,
                    n_experts=32, top_k=6, expert_hidden=64, shared_hidden=128, routed_scale=6.0)
    t = TrainConfig(batch_size=32, total_steps=1600, switch_step=800, warmup_steps=100,
                    explore_steps=100, log_every=10, eval_every=100, eval_batches=10,
                    load_window=50, ckpt_every=400, post_switch_evals=[0, 10, 25, 50, 100, 200])
    return m, t
