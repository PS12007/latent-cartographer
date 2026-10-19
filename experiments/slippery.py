"""Experiment 12 — a slippery maze: does the model keep a *graded* belief?

Moves fail silently with probability `slip` (token emitted, walker stays put).
Even with a known start the position is then uncertain, and the posterior
spreads as possible slips accumulate and sharpens when a move would have been
illegal from some candidate cells. Only start-given walks are analysed here, so
all uncertainty comes from slips.

Measures, per layer: KL(exact posterior || belief probe), against
  * time-only baseline (average posterior at that step), and
  * a slip-unaware belief (dead reckoning: start + displacement, lightly smoothed)
and whether the decoded belief's entropy rises and falls with the true one.

    python experiments/slippery.py --run runs/slip15
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
from cartographer.analysis import collect, fit_belief_probe, kl  # noqa: E402
from cartographer.env import DC, DR, BayesFilter, Maze, belief_entropy, sample_walks  # noqa: E402
from cartographer.model import load_model  # noqa: E402


def dataset(maze, n, T, slip, seed):
    rng = np.random.default_rng(seed)
    tok, cells, known = sample_walks(maze, n, T, 1.0, rng, slip=slip)
    beliefs, _ = BayesFilter(maze, slip=slip).run(torch.tensor(tok))
    return tok, cells, beliefs.float()


def naive_belief(maze, tok, eps=1e-3):
    """Ignore slips: position = start + displacement (clipped to the grid), lightly smoothed."""
    B, S = tok.shape
    r0, c0 = np.divmod(tok[:, 0] - 5, maze.W)
    dr = np.concatenate([np.zeros((B, 1), int), np.cumsum(DR[tok[:, 1:]], 1)], 1)
    dc = np.concatenate([np.zeros((B, 1), int), np.cumsum(DC[tok[:, 1:]], 1)], 1)
    r = np.clip(r0[:, None] + dr, 0, maze.H - 1)
    c = np.clip(c0[:, None] + dc, 0, maze.W - 1)
    P = np.full((B, S, maze.N), eps / maze.N)
    np.put_along_axis(P, (r * maze.W + c)[..., None], 1 - eps + eps / maze.N, axis=2)
    return torch.tensor(P, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/slip15")
    ap.add_argument("--out", default="results/slippery.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    T, slip = ck["args"]["T"], ck["args"]["slip"]
    nL = model.cfg.n_layer
    tr_tok, _, tr_P = dataset(maze, 4000, T, slip, 101)
    te_tok, te_cells, te_P = dataset(maze, 1000, T, slip, 202)
    keys = [f"resid_{l}" for l in range(nL + 1)]
    A_tr, A_te = collect(model, tr_tok, keys), collect(model, te_tok, keys)

    P_tr = tr_P.reshape(-1, maze.N)
    P_te = te_P.reshape(-1, maze.N)
    S = T + 1
    t_idx = torch.arange(S).repeat(len(te_tok))
    mean_by_t = tr_P.mean(0)
    base_time = kl(P_te, mean_by_t[t_idx].clamp_min(1e-8).log())
    base_naive = kl(P_te, naive_belief(maze, te_tok).reshape(-1, maze.N).log())
    print(f"baselines: time-only KL {base_time:.3f} | slip-unaware dead reckoning KL {base_naive:.3f}")

    res = {"slip": slip, "baseline_time_kl": base_time, "baseline_naive_kl": base_naive, "layers": {}}
    best = None
    for k in keys:
        probe = fit_belief_probe(A_tr[k].reshape(-1, A_tr[k].shape[-1]), P_tr, device=args.device).cpu()
        with torch.no_grad():
            logq = F.log_softmax(probe(A_te[k].reshape(-1, A_te[k].shape[-1])), -1)
        v = kl(P_te, logq)
        res["layers"][k] = v
        print(f"{k}: belief KL {v:.3f}", flush=True)
        if best is None or v < best[1]:
            best = (k, v, logq)
    k, v, logq = best
    H_true = belief_entropy(P_te.double()).reshape(-1, S)
    H_probe = belief_entropy(logq.exp().double()).reshape(-1, S)
    res["best_layer"] = k
    res["entropy_by_t"] = {"true": H_true.mean(0).tolist(), "probe": H_probe.mean(0).tolist()}
    res["entropy_corr"] = float(np.corrcoef(H_true.flatten().numpy(), H_probe.flatten().numpy())[0, 1])
    # model's next-move prediction vs optimal (from train log)
    with open(os.path.join(args.run, "train_log.json")) as f:
        last = json.load(f)[-1]
    res["kl_to_optimal"] = {"known": last["kl_known"], "unknown": last["kl_unknown"]}
    # an example walk for plotting: the posterior and the decoded belief over time
    ex = int(np.argmax(H_true[:, 20:40].mean(1).numpy()))
    res["example"] = {"tokens": te_tok[ex].tolist(), "cells": te_cells[ex].tolist(),
                      "true": te_P[ex].tolist(), "probe": logq.exp().reshape(-1, S, maze.N)[ex].tolist()}
    res["maze"] = maze.to_dict()
    print(f"best layer {k}: KL {v:.3f}; per-step entropy correlation {res['entropy_corr']:.2f}")
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
