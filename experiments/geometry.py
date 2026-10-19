"""Experiment 9 — what shape is the internal map?

Average the residual stream over localised moments for each of the 64 cells
(class means), then ask how distances between those 64 points relate to
  * grid distance   (Manhattan |dr| + |dc|, ignoring walls)
  * maze distance   (shortest walking path through the corridors)
  * wall similarity (how many of the 4 sides differ — the "affordance" a cell offers)
using Spearman correlations, per layer. Also writes a 2D PCA of the class means.

    python experiments/geometry.py --run runs/main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.analysis import collect, make_dataset  # noqa: E402
from cartographer.env import Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, c):
    """rank correlation of a and b controlling for c"""
    ra, rb, rc = (np.argsort(np.argsort(x)).astype(float) for x in (a, b, c))
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)  # noqa: E731
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def maze_distances(maze):
    nxt = maze.nxt()
    D = np.zeros((maze.N, maze.N))
    for s in range(maze.N):
        dist = {s: 0}
        q = deque([s])
        while q:
            c = q.popleft()
            for m in range(4):
                if maze.open[c, m] and nxt[c, m] not in dist:
                    dist[nxt[c, m]] = dist[c] + 1
                    q.append(nxt[c, m])
        D[s] = [dist[c] for c in range(maze.N)]
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/geometry.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    nL = model.cfg.n_layer
    data = make_dataset(maze, 3000, ck["args"]["T"], 0.35, seed=77)
    keys = [f"resid_{l}" for l in range(nL + 1)]
    A = collect(model, data["tokens"], keys)
    loc = torch.tensor(data["belief"]).reshape(-1, maze.N).max(-1).values > 0.99
    cell = torch.tensor(data["cell"]).reshape(-1)

    r, c = np.divmod(np.arange(maze.N), maze.W)
    grid = np.abs(r[:, None] - r[None]) + np.abs(c[:, None] - c[None])
    path = maze_distances(maze)
    walls = (maze.open[:, None, :] != maze.open[None, :, :]).sum(-1)
    iu = np.triu_indices(maze.N, 1)

    out = {"layers": {}}
    for k in keys:
        X = A[k].reshape(-1, A[k].shape[-1])
        mu = torch.stack([X[loc & (cell == i)].mean(0) for i in range(maze.N)]).numpy()
        D = np.linalg.norm(mu[:, None] - mu[None], axis=-1)
        d, g, p, w = D[iu], grid[iu], path[iu], walls[iu]
        row = {
            "grid": spearman(d, g), "path": spearman(d, p), "walls": spearman(d, w),
            "path_given_grid": partial_spearman(d, p, g), "grid_given_path": partial_spearman(d, g, p),
        }
        # 2D PCA of the class means
        Z = mu - mu.mean(0)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        row["pca2"] = (Z @ Vt[:2].T).tolist()
        row["pca_var2"] = float((S[:2] ** 2).sum() / (S ** 2).sum())
        out["layers"][k] = row
        print(f"{k}: rank-corr with grid {row['grid']:.2f} | maze path {row['path']:.2f} | walls {row['walls']:.2f} | "
              f"path|grid {row['path_given_grid']:.2f} | grid|path {row['grid_given_path']:.2f} | PCA2 var {row['pca_var2']:.2f}")
    out["maze"] = maze.to_dict()
    out["grid_vs_path"] = spearman(grid[iu], path[iu])
    print("grid vs path distance correlation in this maze:", round(out["grid_vs_path"], 2))
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
