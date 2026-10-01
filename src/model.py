"""
The model: a small GPT whose feed-forward block is either
  * DenseMLP  - one big block (the "linear"/dense model), or
  * MoELayer  - a shared expert + routed experts + a router.

Everything else (attention, embeddings, norms) is identical between the two, so
when we convert dense -> MoE only the feed-forward blocks change.
All Linear layers have no bias, which makes "a neuron" a clean thing to copy:
neuron j = row j of the input matrix + column j of the output matrix.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMLP(nn.Module):
    def __init__(self, n_embd, hidden):
        super().__init__()
        self.fc = nn.Linear(n_embd, hidden, bias=False)    # weight shape (hidden, n_embd)
        self.proj = nn.Linear(hidden, n_embd, bias=False)  # weight shape (n_embd, hidden)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class MoELayer(nn.Module):
    """
    Shared expert (always on) + E routed experts + a router.

    Routing, step by step, for one token:
      1. router logits = token @ W_router            (E numbers, computed in fp32)
      2. scores = sigmoid(logits)  [or softmax]       (how much the router likes each expert)
      3. pick k experts:
           normal:      the k highest  (scores + balance_bias)
           exploring:   sample k without replacement, probability ~ (scores + bias)
         The balance bias only changes WHO is picked, never the weights - same as DeepSeek-V3.
      4. gate weights = the k picked scores, normalised to sum to 1, times routed_scale
      5. output = shared(token) + sum over picked experts of  weight * expert(token)
    No token is ever dropped ("dropless"): every token always gets exactly k experts.
    """

    def __init__(self, n_embd, n_experts, top_k, expert_hidden, shared_hidden,
                 score_fn="sigmoid", routed_scale=1.0):
        super().__init__()
        self.E, self.k, self.H = n_experts, top_k, expert_hidden
        self.score_fn, self.routed_scale = score_fn, routed_scale
        self.shared = DenseMLP(n_embd, shared_hidden) if shared_hidden > 0 else None
        # experts stored as two stacked tensors so they are easy to copy into
        self.w_in = nn.Parameter(torch.empty(n_experts, n_embd, expert_hidden))
        self.w_out = nn.Parameter(torch.empty(n_experts, expert_hidden, n_embd))
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        # the DeepSeek balancing bias: NOT a trained parameter, nudged by hand after each step
        self.register_buffer("balance_bias", torch.zeros(n_experts))
        self.explore = False                 # set by the training loop
        self.last_counts = None              # tokens per expert in the last forward pass
        self.collect_counts = True

    def scores(self, x_flat):
        with torch.autocast(device_type=x_flat.device.type, enabled=False):
            logits = F.linear(x_flat.float(), self.router.weight.float())   # router in fp32
            if self.score_fn == "softmax":
                return torch.softmax(logits, dim=-1)
            return torch.sigmoid(logits)

    def select(self, s):
        biased = s + self.balance_bias
        if self.explore and self.training:
            probs = biased.clamp_min(1e-6)
            return torch.multinomial(probs, self.k, replacement=False)
        return biased.topk(self.k, dim=-1).indices

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        N = xf.shape[0]
        s = self.scores(xf)                                    # (N, E) fp32
        idx = self.select(s.detach())                          # (N, k)  who is picked
        w = s.gather(1, idx)                                   # (N, k)  their scores (gradient flows here)
        w = w / w.sum(dim=-1, keepdim=True) * self.routed_scale

        out = self.shared(xf) if self.shared is not None else torch.zeros_like(xf)
        out = out.to(xf.dtype) if out.dtype != xf.dtype else out

        flat_e = idx.reshape(-1)                               # (N*k,)
        flat_tok = torch.arange(N, device=x.device).repeat_interleave(self.k)
        flat_w = w.reshape(-1)
        order = torch.argsort(flat_e)
        flat_e, flat_tok, flat_w = flat_e[order], flat_tok[order], flat_w[order]
        counts = torch.bincount(flat_e, minlength=self.E)
        if self.collect_counts:
            self.last_counts = counts.detach()
        bounds = counts.cumsum(0).tolist()
        start = 0
        for e in range(self.E):
            end = bounds[e]
            if end > start:
                tok = flat_tok[start:end]
                h = F.gelu(xf[tok] @ self.w_in[e]) @ self.w_out[e]
                out = out.index_add(0, tok, (h * flat_w[start:end, None]).to(out.dtype))
            start = end
        return out.reshape(B, T, C)


class Block(nn.Module):
    def __init__(self, cfg, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn_qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.attn_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = ffn
        self.n_head = cfg.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn_qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg, moe=False):
        super().__init__()
        self.cfg, self.is_moe = cfg, moe
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        blocks = []
        for _ in range(cfg.n_layer):
            if moe:
                ffn = MoELayer(cfg.n_embd, cfg.n_experts, cfg.top_k, cfg.expert_hidden,
                               cfg.shared_hidden, cfg.score_fn, cfg.routed_scale)
            else:
                ffn = DenseMLP(cfg.n_embd, cfg.dense_hidden)
            blocks.append(Block(cfg, ffn))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight      # tied
        self.apply(self._init)
        # GPT-2 style: scale down the layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("attn_proj.weight") or name.endswith("proj.weight") or name.endswith("w_out"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, MoELayer):
            nn.init.normal_(m.w_in, 0.0, 0.02)
            nn.init.normal_(m.w_out, 0.0, 0.02)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoELayer)]

    def set_explore(self, on):
        for m in self.moe_layers():
            m.explore = on

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_params(model):
    """Total parameters, and parameters ACTIVE for one token (what it actually computes with)."""
    total = sum(p.numel() for p in model.parameters())
    if not model.is_moe:
        return total, total
    active = total
    for m in model.moe_layers():
        per_expert = m.w_in[0].numel() + m.w_out[0].numel()
        active -= (m.E - m.k) * per_expert
    return total, active
