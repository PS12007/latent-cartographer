"""Experiment 6 — does the model keep an egocentric map of where it has been?

Target at position t: for each offset (dr, dc) in a 7x7 window around the
*current* cell, has the walker visited that relative location so far?
(and for each such offset, has it traversed the passage east / south of it?)
These are defined without knowing the start, so they are exactly the
information available to a walker who is lost. A linear probe per layer;
random-init control.

    python experiments/egocentric.py --run runs/main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.analysis import collect, fit_multilabel_probe, make_dataset  # noqa: E402
from cartographer.env import DC, DR, Maze  # noqa: E402
from cartographer.model import GPT, GPTConfig, load_model  # noqa: E402
from probes import balanced_acc, stream_points  # noqa: E402

R = 3  # window radius


def ego_targets(gt):
    """visited (B,S,(2R+1)^2), edges (B,S,2*(2R+1)^2) relative to current position."""
    tok, disp = gt["tokens"], gt["disp"]
    B, S = tok.shape
    K = 2 * R + 1
    vis = np.zeros((B, S, K, K), dtype=bool)
    edg = np.zeros((B, S, K, K, 2), dtype=bool)   # passage E (0) / S (1) of a relative cell
    for b in range(B):
        seen_cells = [tuple(disp[b, 0])]
        seen_edges = []
        for t in range(S):
            if t > 0:
                p, q = tuple(disp[b, t - 1]), tuple(disp[b, t])
                seen_cells.append(q)
                m = tok[b, t]
                if m == 1: seen_edges.append((p, 0))
                elif m == 3: seen_edges.append((q, 0))
                elif m == 2: seen_edges.append((p, 1))
                else: seen_edges.append((q, 1))
            cur = disp[b, t]
            for (r, c) in set(seen_cells):
                i, j = r - cur[0] + R, c - cur[1] + R
                if 0 <= i < K and 0 <= j < K:
                    vis[b, t, i, j] = True
            for ((r, c), d) in set(seen_edges):
                i, j = r - cur[0] + R, c - cur[1] + R
                if 0 <= i < K and 0 <= j < K:
                    edg[b, t, i, j, d] = True
    return vis.reshape(B, S, -1), edg.reshape(B, S, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/egocentric.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(dev)
    maze = Maze.from_dict(ck["maze"])
    train = make_dataset(maze, 3000, ck["args"]["T"], 0.0, seed=111)
    test = make_dataset(maze, 600, ck["args"]["T"], 0.0, seed=222)
    V_tr, E_tr = (torch.tensor(a).reshape(-1, a.shape[-1]) for a in ego_targets(train))
    V_te, E_te = (torch.tensor(a).reshape(-1, a.shape[-1]) for a in ego_targets(test))
    # drop the centre cell (always visited) and constant columns
    keepV = (V_tr.float().mean(0) > 0.01) & (V_tr.float().mean(0) < 0.99)
    keepE = (E_tr.float().mean(0) > 0.01) & (E_tr.float().mean(0) < 0.99)
    V_tr, V_te, E_tr, E_te = V_tr[:, keepV], V_te[:, keepV], E_tr[:, keepE], E_te[:, keepE]

    res = {}
    for tag, m in [("trained", model), ("random", None)]:
        if m is None:
            torch.manual_seed(0)
            m = GPT(GPTConfig(**ck["config"])).to(dev).eval()
        pts = stream_points(m.cfg.n_layer)
        A_tr = collect(m, train["tokens"], [k for k, _ in pts])
        A_te = collect(m, test["tokens"], [k for k, _ in pts])
        res[tag] = {}
        for k, label in pts:
            X_tr, X_te = A_tr[k].reshape(-1, A_tr[k].shape[-1]), A_te[k].reshape(-1, A_te[k].shape[-1])
            pv = fit_multilabel_probe(X_tr, V_tr, device=dev)
            pe = fit_multilabel_probe(X_tr, E_tr, device=dev)
            with torch.no_grad():
                bv = balanced_acc(V_te, pv(X_te.to(dev)).cpu())
                be = balanced_acc(E_te, pe(X_te.to(dev)).cpu())
            res[tag][k] = {"label": label, "visited_bacc": bv, "edges_bacc": be}
            print(f"[{tag}] {label:10s} ego visited bacc {bv:.3f} | ego passages bacc {be:.3f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
