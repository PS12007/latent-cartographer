"""Train the move-sequence GPT.

    python -m cartographer.train --out runs/main

The model is trained with plain next-token cross-entropy on *sampled* moves.
We never give it the Bayes-optimal distribution; we only use the filter to
measure how close it gets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .env import BayesFilter, make_maze, sample_walks
from .model import GPT, GPTConfig


@torch.no_grad()
def evaluate(model, maze, bf, n=2048, T=64, seed=12345, p_known=0.5, slip=0.0):
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    tokens, cells, known = sample_walks(maze, n, T, p_known, rng, slip=slip)
    tok = torch.tensor(tokens, device=dev)
    _, pred = bf.run(tok)                       # (n, T+1, 4) optimal predictive
    pred = pred[:, :-1].float().to(dev)                 # predict m_{t+1} from prefix ending at t
    logits = model(tok[:, :-1])
    logp = F.log_softmax(logits, -1)
    tgt = tok[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, 4), tgt.reshape(-1)).item()
    ent = -(pred * pred.clamp_min(1e-30).log()).sum(-1)       # optimal CE per position
    kl = (pred * (pred.clamp_min(1e-30).log() - logp)).sum(-1)  # KL(opt || model)
    kn = torch.tensor(known, device=dev)
    return {
        "ce": ce,
        "ce_opt": ent.mean().item(),
        "kl": kl.mean().item(),
        "kl_known": kl[kn].mean().item() if kn.any() else float("nan"),
        "kl_unknown": kl[~kn].mean().item() if (~kn).any() else float("nan"),
        "kl_by_t": kl.mean(0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--braid", type=float, default=0.35)
    ap.add_argument("--maze_seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--p_known", type=float, default=0.35)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--snapshots", type=int, default=12, help="checkpoints saved along training (for dynamics)")
    ap.add_argument("--slip", type=float, default=0.0, help="probability a move silently fails")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    maze = make_maze(args.H, args.W, args.braid, args.maze_seed)
    bf = BayesFilter(maze, device=args.device, slip=args.slip)
    cfg = GPTConfig(vocab_size=maze.vocab_size, block_size=args.T + 1, n_layer=args.n_layer,
                    n_head=args.n_head, d_model=args.d_model, d_mlp=4 * args.d_model)
    model = GPT(cfg).to(args.device)
    print(maze.ascii())
    print(f"params: {model.n_params():,}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    snap_steps = set(np.unique(np.geomspace(50, args.steps, args.snapshots).astype(int)).tolist())
    rng = np.random.default_rng(args.seed + 1)
    log = []
    t0 = time.time()

    def save(path, step):
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "maze": maze.to_dict(),
                    "args": vars(args), "step": step}, path)

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev = evaluate(model, maze, bf, T=args.T, slip=args.slip)
            model.train()
            ev.update(step=step, time=time.time() - t0)
            log.append(ev)
            print(f"step {step:5d} | ce {ev['ce']:.4f} (opt {ev['ce_opt']:.4f}) | "
                  f"KL known {ev['kl_known']:.4f} unknown {ev['kl_unknown']:.4f} | {ev['time']:.0f}s", flush=True)
        if step in snap_steps:
            os.makedirs(os.path.join(args.out, "snapshots"), exist_ok=True)
            save(os.path.join(args.out, "snapshots", f"step{step:06d}.pt"), step)
        if step == args.steps:
            break
        tokens, _, _ = sample_walks(maze, args.batch, args.T, args.p_known, rng, slip=args.slip)
        tok = torch.from_numpy(tokens).to(args.device, non_blocking=True)
        logits = model(tok[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 4), tok[:, 1:].reshape(-1))
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    save(os.path.join(args.out, "model.pt"), args.steps)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
