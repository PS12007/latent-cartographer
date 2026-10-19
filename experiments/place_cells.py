"""Experiment 11 — place cells, and whether they encode position or belief.

For every MLP neuron in every block:
  * spatial tuning: eta^2 = fraction of the neuron's variance (over localised moments)
    explained by which cell the walker is in; and the same for the cell's wall
    configuration (16 classes) — a neuron can look spatial just because it
    responds to "dead end facing east".
  * place-cell score: eta^2(cell) high, but the cell explains much more than its
    walls do (eta^2(cell) - eta^2(walls)), and the field is compact.

Then the key test, on *uncertain* moments (start unknown, several candidate cells):
for a place neuron tuned to cell c, is its activation better predicted by
  (a) whether the walker is really at c (truth), or
  (b) the posterior probability of c (belief)?
A neuron that tracks (b) is a belief neuron: it fires for places the walker
might be, in proportion to how likely they are.

    python experiments/place_cells.py --run runs/main
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


def eta2(x, groups, n_groups):
    """Fraction of variance of x (n,k) explained by group means."""
    tot = ((x - x.mean(0)) ** 2).sum(0)
    means = np.zeros((n_groups, x.shape[1]))
    cnt = np.bincount(groups, minlength=n_groups)
    np.add.at(means, groups, x)
    means /= np.maximum(cnt, 1)[:, None]
    between = (cnt[:, None] * (means - x.mean(0)) ** 2).sum(0)
    return between / np.maximum(tot, 1e-12), means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/place_cells.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    nL = model.cfg.n_layer
    T = ck["args"]["T"]
    data = make_dataset(maze, 3000, T, 0.0, seed=303)   # start unknown: both certain and uncertain moments
    keys = [f"L{l}.mlp_post" for l in range(nL)]
    A = collect(model, data["tokens"], keys)
    P = data["belief"].reshape(-1, maze.N)
    cell = data["cell"].reshape(-1)
    wallcode = (maze.open.astype(int) * np.array([1, 2, 4, 8])).sum(1)  # 16 configurations
    loc = P.max(-1) > 0.99
    unc = (P.max(-1) < 0.9) & (np.arange(len(P)) % (T + 1) > 0)

    out = {"layers": {}, "maze": maze.to_dict()}
    for l, k in enumerate(keys):
        X = A[k].reshape(-1, A[k].shape[-1]).numpy()
        e_cell, ratemap = eta2(X[loc], cell[loc], maze.N)
        e_wall, _ = eta2(X[loc], wallcode[cell[loc]], 16)
        pref = ratemap.argmax(0)                                   # preferred cell
        # field size: number of cells above half of the peak (relative to the minimum)
        lo, hi = ratemap.min(0), ratemap.max(0)
        field = ((ratemap - lo) > 0.5 * (hi - lo)).sum(0)
        place = (e_cell > 0.5) & (e_cell - e_wall > 0.3) & (field <= 3)

        # belief vs truth on uncertain moments, for place neurons
        Xu, Pu, cu = X[unc], P[unc], cell[unc]
        r_belief, r_truth, r_map = [], [], []
        map_cell = Pu.argmax(-1)
        for n in np.where(place)[0]:
            y = Xu[:, n]
            b = Pu[:, pref[n]]
            t = (cu == pref[n]).astype(float)
            if t.std() == 0 or b.std() == 0:
                continue
            r_belief.append(np.corrcoef(y, b)[0, 1])
            r_truth.append(np.corrcoef(y, t)[0, 1])
            mp = (map_cell == pref[n]).astype(float)
            r_map.append(np.corrcoef(y, mp)[0, 1] if mp.std() > 0 else np.nan)
        row = {
            "n_neurons": int(X.shape[1]),
            "n_place": int(place.sum()),
            "median_eta2_cell": float(np.median(e_cell)),
            "median_eta2_walls": float(np.median(e_wall)),
            "frac_cell_dominated": float(((e_cell - e_wall) > 0.3).mean()),
            "frac_wall_dominated": float(((e_wall > 0.5) & (e_wall / np.maximum(e_cell, 1e-9) > 0.8)).mean()),
            "belief_vs_truth": {"n": len(r_belief), "r_belief_mean": float(np.mean(r_belief)) if r_belief else None,
                                "r_truth_mean": float(np.mean(r_truth)) if r_truth else None,
                                "frac_belief_better": float(np.mean(np.array(r_belief) > np.array(r_truth))) if r_belief else None,
                                "r_map_mean": float(np.nanmean(r_map)) if r_map else None,
                                "frac_belief_better_than_map": float(np.nanmean(np.array(r_belief) > np.array(r_map))) if r_map else None,
                                "r_belief": [float(v) for v in r_belief], "r_map": [float(v) for v in r_map],
                                "r_truth": [float(v) for v in r_truth]},
            # store the strongest place neurons' ratemaps for plotting
            "examples": [{"neuron": int(n), "pref": int(pref[n]), "eta2": float(e_cell[n]), "ratemap": ratemap[:, n].tolist()}
                         for n in np.argsort(-(e_cell - e_wall) * place)[:64] if place[n]],
        }
        out["layers"][l] = row
        bv = row["belief_vs_truth"]
        print(f"block {l + 1}: place neurons {row['n_place']}/{row['n_neurons']} | cell-dominated {row['frac_cell_dominated']:.0%} "
              f"| wall-dominated {row['frac_wall_dominated']:.0%} | uncertain moments: r(belief) {bv['r_belief_mean']} "
              f"vs r(truth) {bv['r_truth_mean']} (belief wins {bv['frac_belief_better']}) | r(best guess) {bv['r_map_mean']} "
              f"(belief wins {bv['frac_belief_better_than_map']})", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
