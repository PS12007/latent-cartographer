"""Experiment 15 — do localisation heads look up *where you have been nearby*?

For every head, average its attention from the newest token (time t) to each
earlier token j as a function of the spatial distance between where the walker
is now and where it was at j (Manhattan distance of displacements, computable
without knowing the start), and as a function of temporal distance t - j.
A head that builds the egocentric sketch should prefer past moments that were
physically close, regardless of how long ago they were.

Score per head: attention lift at spatial distance 0-1 relative to what a head
that ignores space would give, controlling for time by comparing against the
same head's attention to equally-old tokens.

    python experiments/spatial_attention.py --run runs/main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.analysis import collect, make_dataset  # noqa: E402
from cartographer.env import Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402

MAXD = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/spatial_attention.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    d = make_dataset(maze, 600, ck["args"]["T"], 0.0, seed=31)
    nL, H = model.cfg.n_layer, model.cfg.n_head
    A = collect(model, d["tokens"], [f"L{l}.attn" for l in range(nL)])
    disp = d["disp"]                                             # (B,S,2)
    B, S = disp.shape[:2]
    ts = np.arange(16, S)                                        # queries late enough to have history
    # spatial distance between position at query t and at key j, and time gap
    sd = np.abs(disp[:, ts, None, :] - disp[:, None, :, :]).sum(-1)       # (B, len(ts), S)
    tg = ts[:, None] - np.arange(S)[None, :]                                # (len(ts), S)
    valid = (tg >= 1)[None].repeat(B, 0)                                    # keys strictly before the query, excluding the start token
    valid[:, :, 0] = False
    res = {}
    for l in range(nL):
        att = A[f"L{l}.attn"][:, :, ts, :].numpy()                         # (B,H,len(ts),S)
        for h in range(H):
            a = att[:, h]
            prof_space = []
            for dist in range(MAXD + 1):
                m = valid & (np.minimum(sd, MAXD) == dist)
                prof_space.append(float(a[m].mean()) if m.any() else float("nan"))
            # time-controlled lift: for each time gap, attention to nearby (<=1) vs far (>1) keys
            lifts = []
            for gap in range(1, 48):
                mg = valid & (tg[None] == gap)
                near, far = mg & (sd <= 1), mg & (sd > 1)
                if near.sum() > 50 and far.sum() > 50:
                    lifts.append(a[near].mean() / max(a[far].mean(), 1e-9))
            res[f"L{l}H{h}"] = {"by_space": prof_space, "near_far_lift": float(np.exp(np.mean(np.log(lifts)))) if lifts else None}
            print(f"block {l + 1} head {h}: attention to nearby vs far past moments of the same age: x{res[f'L{l}H{h}']['near_far_lift']:.2f}",
                  flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
