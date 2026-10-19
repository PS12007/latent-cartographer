"""Experiment 7 — is the egocentric map *used* for localisation? (causal test)

Find moments in unknown-start walks where the exact posterior has exactly two
candidate cells: A (true) and B. Both are consistent with everything seen so far,
so they differ only in parts of the maze the walker has not explored.

Choose a passage, at some offset relative to the walker, that has NOT been
walked yet, and write "I have walked this passage" into the residual stream
(after block 3, newest token only) along the egocentric-passage probe direction.

  pro-B  : passage open around B, walled around A  -> belief should move to B
  pro-A  : passage open around A, walled around B  -> belief should move to A
  neutral: passage open around both                -> no change expected

The belief is read out after block 4 with a belief probe, as P(B) / (P(A)+P(B)).

    python experiments/egocentric_causal.py --run runs/main
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
from cartographer.analysis import collect, fit_belief_probe, fit_multilabel_probe, make_dataset  # noqa: E402
from cartographer.env import Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402
from egocentric import R, ego_targets  # noqa: E402

K = 2 * R + 1
ALPHAS = [0, 2, 4, 8, 12, 16]


def passage_open(maze, cell, dr, dc, d):
    """Is the passage east (d=0) / south (d=1) of the cell at (cell + offset) open? None if off-grid."""
    r, c = divmod(int(cell), maze.W)
    r, c = r + dr, c + dc
    if not (0 <= r < maze.H and 0 <= c < maze.W):
        return None
    if (d == 0 and c + 1 >= maze.W) or (d == 1 and r + 1 >= maze.H):
        return None
    return bool(maze.open[r * maze.W + c, 1 if d == 0 else 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="results/egocentric_causal.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, Le = args.device, args.layer
    rng = np.random.default_rng(0)

    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(dev)
    maze = Maze.from_dict(ck["maze"])
    nL = model.cfg.n_layer
    T = ck["args"]["T"]

    train = make_dataset(maze, 3000, T, 0.0, seed=111)
    _, E = ego_targets(train)
    A = collect(model, train["tokens"], [f"resid_{Le}", f"resid_{nL}"])
    D = A[f"resid_{Le}"].shape[-1]
    ego_probe = fit_multilabel_probe(A[f"resid_{Le}"].reshape(-1, D), torch.tensor(E).reshape(-1, E.shape[-1]), device=dev).cpu()
    Wego = ego_probe.lin.weight.detach()
    bel_probe = fit_belief_probe(A[f"resid_{nL}"].reshape(-1, D), torch.tensor(train["belief"]).reshape(-1, maze.N), device=dev).cpu()

    test = make_dataset(maze, 8000, T, 0.0, seed=9)
    _, Et = ego_targets(test)
    items = []
    for b in range(len(test["tokens"])):
        for t in range(4, 48):
            sup = np.where(test["belief"][b, t] > 1e-6)[0]
            if len(sup) != 2:
                continue
            a = int(test["cell"][b, t])
            bb = int(sup[sup != a][0])
            groups = {"pro_B": [], "pro_A": [], "neutral": []}
            walked = Et[b, t].reshape(K, K, 2)
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    for d in range(2):
                        if walked[dr + R, dc + R, d]:
                            continue
                        oa, ob = passage_open(maze, a, dr, dc, d), passage_open(maze, bb, dr, dc, d)
                        if oa is None or ob is None:
                            continue
                        key = "pro_B" if ob and not oa else "pro_A" if oa and not ob else "neutral" if oa and ob else None
                        if key:
                            groups[key].append((dr, dc, d))
            if all(groups.values()):
                pick = {k: v[rng.integers(len(v))] for k, v in groups.items()}
                items.append((b, t, a, bb, pick))
            break
        if len(items) >= args.n:
            break
    print(f"{len(items)} two-candidate moments")

    def fidx(dr, dc, d):
        return ((dr + R) * K + (dc + R)) * 2 + d

    @torch.no_grad()
    def belief_after(tokens, deltas):
        tt = torch.tensor(tokens, device=dev)
        dl = deltas.to(dev)
        cache = {}

        def hook(l, x):
            if l == Le:
                x = x.clone()
                x[:, -1] += dl
            return x
        model(tt, cache=cache, resid_hook=hook)
        return F.softmax(bel_probe(cache[f"resid_{nL}"][:, -1].cpu()), -1)

    out = {"n": len(items), "layer": Le, "alphas": ALPHAS, "conditions": {}}
    for cond in ["pro_B", "pro_A", "neutral"]:
        out["conditions"][cond] = {}
        for alpha in ALPHAS:
            shifts = []
            for (b, t, a, bb, pick) in items:
                tok = test["tokens"][b:b + 1, :t + 1]
                w = Wego[fidx(*pick[cond])]
                p0 = belief_after(tok, torch.zeros(D))[0]
                p1 = belief_after(tok, alpha * w / w.norm())[0]
                f0 = (p0[bb] / (p0[a] + p0[bb])).item()
                f1 = (p1[bb] / (p1[a] + p1[bb])).item()
                shifts.append(f1 - f0)
            s = np.array(shifts)
            out["conditions"][cond][alpha] = {"mean": float(s.mean()), "sem": float(s.std() / np.sqrt(len(s))),
                                              "frac_toward_B": float((s > 0).mean())}
            print(f"{cond:8s} alpha {alpha:3d}: shift toward B {s.mean():+.3f} ± {s.std() / np.sqrt(len(s)):.3f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
