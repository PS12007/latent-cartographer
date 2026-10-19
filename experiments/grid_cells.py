"""Experiment 16 — periodic ("grid-like") spatial codes in an open arena?

RNNs trained to path-integrate in open arenas develop grid-like, periodic
tuning. Here a transformer is trained on 128-move walks in a 16x16 arena with no
internal walls (start always given). For every MLP neuron we build its ratemap
(mean activation per cell) and ask how periodic it is:

  periodicity = share of the ratemap's non-DC spectral power held by its single
                strongest component with >= 2 cycles per arena (counting the
                component and its mirror). High for repeating, grid-like tuning;
                low for one broad gradient, a single bump, or noise.
  n_fields    = number of separate firing fields (connected regions above half max).

Each neuron is compared with a null made by shuffling the cells of its own
ratemap (same values, no spatial structure).

    python experiments/grid_cells.py --run runs/arena16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.env import Maze, sample_walks  # noqa: E402
from cartographer.model import load_model  # noqa: E402


def periodicity(rm):
    z = rm - rm.mean()
    P = np.abs(np.fft.fft2(z)) ** 2
    H, W = rm.shape
    fy = np.abs(np.fft.fftfreq(H) * H)[:, None]
    fx = np.abs(np.fft.fftfreq(W) * W)[None, :]
    f = np.maximum(fy, fx)
    tot = P[f > 0].sum()
    if tot <= 0:
        return 0.0
    return float(2 * P[f >= 2].max() / tot)       # x2: a real ratemap's spectrum is symmetric


def n_fields(rm):
    lo, hi = rm.min(), rm.max()
    if hi - lo < 1e-9:
        return 0
    m = rm > lo + 0.5 * (hi - lo)
    seen = np.zeros_like(m)
    k = 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] and not seen[i, j]:
                k += 1
                stack = [(i, j)]
                seen[i, j] = True
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < m.shape[0] and 0 <= y < m.shape[1] and m[x, y] and not seen[x, y]:
                            seen[x, y] = True
                            stack.append((x, y))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arena16")
    ap.add_argument("--out", default="results/grid_cells.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    T = ck["args"]["T"]
    nL = model.cfg.n_layer
    rng = np.random.default_rng(0)
    tok, cells, _ = sample_walks(maze, 4000, T, 1.0, rng)
    sums = {l: np.zeros((maze.N, model.cfg.d_mlp)) for l in range(nL)}
    cnt = np.zeros(maze.N)
    with torch.no_grad():
        for i in range(0, len(tok), 250):
            cache = {}
            model(torch.tensor(tok[i:i + 250], device=args.device), cache=cache)
            c = cells[i:i + 250].reshape(-1)
            np.add.at(cnt, c, 1)
            for l in range(nL):
                a = cache[f"L{l}.mlp_post"].reshape(-1, model.cfg.d_mlp).cpu().numpy()
                np.add.at(sums[l], c, a)
    res = {"layers": {}, "H": maze.H, "W": maze.W}
    for l in range(nL):
        R = (sums[l] / cnt[:, None]).T.reshape(-1, maze.H, maze.W)          # (D, H, W)
        active = R.std(axis=(1, 2)) > 1e-3
        per = np.array([periodicity(r) for r in R])
        nf = np.array([n_fields(r) for r in R])
        null = np.array([periodicity(rng.permutation(r.reshape(-1)).reshape(r.shape)) for r in R])
        thr = np.quantile(null[active], 0.99)
        sel = active & (per > thr)
        # border vs coordinate tuning. Border features: "on the outer ring" and "on wall k" (k = N/E/S/W)
        yy, xx = np.mgrid[0:maze.H, 0:maze.W]
        walls = [yy == 0, xx == maze.W - 1, yy == maze.H - 1, xx == 0]
        feats = [np.any(walls, 0)] + walls
        flat = R.reshape(len(R), -1)
        def corr(a, b):
            a = a - a.mean(1, keepdims=True); b = b - b.mean()
            return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-12)
        border_r = np.max(np.abs(np.stack([corr(flat, f.reshape(-1).astype(float)) for f in feats])), 0)
        Xc = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(maze.N)], 1).astype(float)
        beta, *_ = np.linalg.lstsq(Xc, flat.T, rcond=None)
        resid = flat.T - Xc @ beta
        coord_r2 = 1 - (resid ** 2).sum(0) / (((flat.T - flat.T.mean(0)) ** 2).sum(0) + 1e-12)
        is_border = active & (border_r >= 0.7)
        is_coord = active & (coord_r2 >= 0.7)
        res["layers"][l] = {
            "n_border": int(is_border.sum()), "n_coordinate": int(is_coord.sum()),
            "border_examples": [{"neuron": int(n), "border_r": float(border_r[n]), "ratemap": R[n].tolist()}
                                for n in np.argsort(-border_r * active)[:60]],
            "coord_examples": [{"neuron": int(n), "r2": float(coord_r2[n]), "ratemap": R[n].tolist()}
                               for n in np.argsort(-coord_r2 * active)[:3]],
            "n_active": int(active.sum()),
            "periodicity_median": float(np.median(per[active])),
            "null_median": float(np.median(null[active])),
            "null_q99": float(thr),
            "n_periodic_beyond_null": int(sel.sum()),
            "n_multi_field": int((active & (nf >= 3)).sum()),
            "examples": [{"neuron": int(n), "periodicity": float(per[n]), "n_fields": int(nf[n]), "ratemap": R[n].tolist()}
                         for n in np.argsort(-(per * active))[:8]],
            "smooth_examples": [{"neuron": int(n), "periodicity": float(per[n]), "ratemap": R[n].tolist()}
                                for n in np.argsort(per + (~active) * 9)[:4]],
        }
        print(f"block {l + 1}: active {active.sum()} | periodicity median {np.median(per[active]):.2f} (shuffled null {np.median(null[active]):.2f}) "
              f"| beyond null 99th pct: {sel.sum()} | >=3 fields: {(active & (nf >= 3)).sum()} "
              f"| border cells (|r|>=0.7 with a wall) {is_border.sum()} | linear-coordinate cells (R2>=0.7) {is_coord.sum()}", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
