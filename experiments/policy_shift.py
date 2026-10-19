"""Experiment 10 — does the map survive a change in how the walker moves?

The model was trained on uniform random walks. Here we feed it walks from
policies it has never seen:
  uniform       (training distribution)
  no-backtrack  never immediately reverse unless at a dead end
  momentum      keep the previous direction with prob. 0.8 when possible
and measure, at moments where the walker's position is pinned down,
  * legal-set accuracy: does {moves the model gives > 10%} equal the open sides of the true cell?
  * probe cell accuracy: argmax of the belief probe (fitted on uniform walks only) == true cell.
For start-unknown walks the exact posterior depends on the policy, so we only
score known-start walks for legal-set accuracy and use the probe on both.

    python experiments/policy_shift.py --run runs/main
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
from cartographer.analysis import collect, fit_belief_probe, ground_truth, make_dataset  # noqa: E402
from cartographer.env import START0, UNK, Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402


def walks(maze, B, T, policy, rng):
    nxt = maze.nxt()
    cur = rng.integers(maze.N, size=B)
    cells = np.empty((B, T + 1), dtype=np.int64)
    moves = np.empty((B, T), dtype=np.int64)
    cells[:, 0] = cur
    prev = np.full(B, -1)
    for t in range(T):
        legal = maze.open[cur].astype(float)
        w = legal.copy()
        if policy == "no-backtrack":
            back = np.where(prev >= 0, (prev + 2) % 4, -1)
            for b in range(B):
                if back[b] >= 0 and legal[b].sum() > 1:
                    w[b, back[b]] = 0
        elif policy == "momentum":
            for b in range(B):
                if prev[b] >= 0 and legal[b, prev[b]]:
                    w[b] = legal[b] * 0.2 / max(legal[b].sum() - 1, 1)
                    w[b, prev[b]] = 0.8 if legal[b].sum() > 1 else 1.0
        w = w / w.sum(1, keepdims=True)
        m = np.array([rng.choice(4, p=w[b]) for b in range(B)])
        moves[:, t] = m
        prev = m
        cur = nxt[cur, m]
        cells[:, t + 1] = cur
    tokens = np.concatenate([(START0 + cells[:, :1]), moves], 1)  # known start
    return tokens, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/policy_shift.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    T = ck["args"]["T"]
    nL = model.cfg.n_layer
    train = make_dataset(maze, 3000, T, 0.35, seed=101)
    A = collect(model, train["tokens"], [f"resid_{nL - 1}"])[f"resid_{nL - 1}"]
    probe = fit_belief_probe(A.reshape(-1, A.shape[-1]), torch.tensor(train["belief"]).reshape(-1, maze.N),
                             device=args.device).cpu()
    legal = torch.tensor(maze.open)
    rng = np.random.default_rng(0)
    res = {}
    for policy in ["uniform", "no-backtrack", "momentum"]:
        tok, cells = walks(maze, 800, T, policy, rng)
        acts = collect(model, tok, [f"resid_{nL - 1}"])
        p = F.softmax(acts["logits"], -1)                         # (B,S,4)
        cell = torch.tensor(cells)
        ok = ((p > 0.1) == legal[cell]).all(-1).float()
        q = probe(acts[f"resid_{nL - 1}"]).argmax(-1)
        res[policy] = {"legal_set_acc": ok.mean().item(),
                       "legal_set_acc_late": ok[:, 32:].mean().item(),
                       "probe_cell_acc": (q == cell).float().mean().item()}
        # start unknown: score only moments where exactly one cell is consistent with the walls
        # (the consistent set does not depend on the policy, only on the maze)
        tok_u = tok.copy()
        tok_u[:, 0] = UNK
        gt = ground_truth(maze, tok_u, cells, np.zeros(len(tok_u), dtype=bool))
        single = torch.tensor((gt["belief"] > 1e-12).sum(-1) == 1)
        acts_u = collect(model, tok_u, [f"resid_{nL - 1}"])
        pu = F.softmax(acts_u["logits"], -1)
        oku = ((pu > 0.1) == legal[cell]).all(-1)
        qu = probe(acts_u[f"resid_{nL - 1}"]).argmax(-1)
        res[policy]["unknown_legal_set_acc_localised"] = oku[single].float().mean().item()
        res[policy]["unknown_probe_cell_acc_localised"] = (qu == cell)[single].float().mean().item()
        res[policy]["unknown_frac_localised_by_t32"] = single[:, 32].float().mean().item()
        print(policy, {k: round(v, 3) for k, v in res[policy].items()}, flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
