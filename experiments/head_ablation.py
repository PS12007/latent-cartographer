"""Experiment 8 — which attention heads matter for dead reckoning vs localisation?

Mean-ablate one head at a time (replace its output with its average over the
dataset at that position) and measure the increase in KL to the Bayes-optimal
predictor, separately for start-given and start-unknown walks.

    python experiments/head_ablation.py --run runs/main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.env import BayesFilter, Maze, sample_walks  # noqa: E402
from cartographer.model import load_model  # noqa: E402


@torch.no_grad()
def kl_split(model, tok, pred, known, bs=500):
    kls = []
    for i in range(0, len(tok), bs):
        logp = F.log_softmax(model(tok[i:i + bs]), -1)
        p = pred[i:i + bs]
        kls.append((p * (p.clamp_min(1e-30).log() - logp)).sum(-1))   # (b,S)
    kl = torch.cat(kls)
    return kl[known].mean().item(), kl[~known].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/head_ablation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(dev)
    maze = Maze.from_dict(ck["maze"])
    T = ck["args"]["T"]
    rng = np.random.default_rng(5)
    tokens, _, known = sample_walks(maze, 3000, T, 0.5, rng)
    tok = torch.tensor(tokens[:, :-1], device=dev)
    _, pred = BayesFilter(maze, device=dev).run(torch.tensor(tokens, device=dev))
    pred = pred[:, :-1].float()
    known = torch.tensor(known, device=dev)

    # per-position mean of each head's output
    sums = {}
    with torch.no_grad():
        for i in range(0, len(tok), 500):
            cache = {}
            model(tok[i:i + 500], cache=cache)
            for l in range(model.cfg.n_layer):
                v = cache[f"L{l}.head_out"].sum(0, keepdim=True)
                sums[l] = sums.get(l, 0) + v
    means = {l: sums[l] / len(tok) for l in sums}  # (1,H,S,dh) per-position mean of each head
    base = kl_split(model, tok, pred, known)
    print(f"baseline KL known {base[0]:.4f} unknown {base[1]:.4f}")

    res = {"baseline": base, "heads": {}}
    for l, blk in enumerate(model.blocks):
        for h in range(model.cfg.n_head):
            def hook(y, l=l, h=h):
                y = y.clone()
                y[:, h] = means[l][:, h, : y.shape[2]]
                return y
            blk.attn.head_hook = hook
            k, u = kl_split(model, tok, pred, known)
            blk.attn.head_hook = None
            res["heads"][f"L{l}H{h}"] = {"known": k, "unknown": u}
            print(f"ablate block {l + 1} head {h}: KL known {k:.4f} (+{k - base[0]:.4f}) | unknown {u:.4f} (+{u - base[1]:.4f})",
                  flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
