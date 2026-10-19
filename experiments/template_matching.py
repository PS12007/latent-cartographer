"""Experiment 13 — do place cells localise by template matching?

Hypothesis: a block-4 place neuron for cell c compares the egocentric sketch of
walked passages with the maze layout around c. It should be *excited* by walked
passages that exist around c and *inhibited* by walked passages that would cross
a wall there (walking them is impossible if you are at c).

For each place neuron we fit a ridge regression of its activation on the 98
egocentric passage features (start-unknown walks), giving a passage receptive
field beta(offset, direction). The template for cell c is +1 where the passage
at that offset from c is open, -1 where it is a wall, and ignored off-grid.
Score = correlation(beta, template of the neuron's own cell), compared with the
templates of every other cell (null distribution).

    python experiments/template_matching.py --run runs/main
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
from cartographer.env import Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402
from egocentric import R, ego_targets  # noqa: E402

K = 2 * R + 1


def template(maze, cell):
    """(K*K*2,) +1 open / -1 wall / nan off-grid, for the passage E (d=0) / S (d=1) of each offset cell.
    (Coding off-grid passages as walls instead was also tried; it matches less well, see README.)"""
    r0, c0 = divmod(cell, maze.W)
    t = np.full((K, K, 2), np.nan)
    for i in range(K):
        for j in range(K):
            r, c = r0 + i - R, c0 + j - R
            if not (0 <= r < maze.H and 0 <= c < maze.W):
                continue
            for d in range(2):
                if (d == 0 and c + 1 < maze.W) or (d == 1 and r + 1 < maze.H):
                    t[i, j, d] = 1.0 if maze.open[r * maze.W + c, 1 if d == 0 else 2] else -1.0
    return t.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--places", default="results/place_cells.json")
    ap.add_argument("--out", default="results/template_matching.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    nL = model.cfg.n_layer
    data = make_dataset(maze, 3000, ck["args"]["T"], 0.0, seed=515)
    _, E = ego_targets(data)
    X = E[:, 1:].reshape(-1, E.shape[-1]).astype(float)          # passage features, t >= 1
    keep = X.std(0) > 0
    act = collect(model, data["tokens"], [f"L{nL - 1}.mlp_post"])[f"L{nL - 1}.mlp_post"][:, 1:]
    act = act.reshape(-1, act.shape[-1]).numpy()

    with open(args.places) as f:
        PC = json.load(f)
    # all place neurons in the last block, with preferred cells (recompute preference from the stored examples list)
    ex = PC["layers"][str(nL - 1)]["examples"]
    neurons = [(e["neuron"], e["pref"]) for e in ex]

    Xk = X[:, keep]
    Xc = Xk - Xk.mean(0)
    A = Xc.T @ Xc + 1e-2 * len(Xc) * np.eye(Xc.shape[1])
    templates = np.stack([template(maze, c) for c in range(maze.N)])[:, keep]     # (N, F)
    rows = []
    for n, pref in neurons:
        y = act[:, n] - act[:, n].mean()
        beta = np.linalg.solve(A, Xc.T @ y)
        scores = []
        for c in range(maze.N):
            m = ~np.isnan(templates[c])
            scores.append(np.corrcoef(beta[m], templates[c][m])[0, 1] if m.sum() > 5 else np.nan)
        scores = np.array(scores)
        own = scores[pref]
        others = np.delete(scores, pref)
        rows.append({"neuron": n, "pref": pref, "own": float(own),
                     "others_mean": float(np.nanmean(others)),
                     "rank_of_own": int((np.nan_to_num(others, nan=-9) >= own).sum()) + 1,
                     "beta": beta.tolist()})
    # ---- causal receptive fields: write each passage feature into the residual stream after
    # block 3 (newest token only) along the egocentric probe direction, and record how much
    # each place neuron's activation changes. This manipulates the feature instead of
    # observing correlations, so "the neuron is just a position indicator" cannot produce it.
    Ae = collect(model, data["tokens"], [f"resid_{nL - 1}"])[f"resid_{nL - 1}"]
    D = Ae.shape[-1]
    ego_probe = fit_multilabel_probe(Ae.reshape(-1, D), torch.tensor(E).reshape(-1, E.shape[-1]), device=args.device).cpu()
    Wego = ego_probe.lin.weight.detach()
    rng = np.random.default_rng(1)
    base = []
    P = data["belief"]
    for b in rng.permutation(len(data["tokens"])):
        t = int(rng.integers(6, 40))
        if (P[b, t] > 1e-6).sum() >= 2:         # uncertain moments, where matching matters
            base.append((b, t))
        if len(base) >= 256:
            break
    idx = [n for n, _ in neurons]
    feats = np.where(keep)[0]
    causal = np.zeros((len(idx), len(feats)))
    by_len = {}
    for b, t in base:
        by_len.setdefault(t, []).append(b)
    alpha = 8.0
    with torch.no_grad():
        for t, bs in by_len.items():
            tok = torch.tensor(data["tokens"][bs, :t + 1], device=args.device)
            for fi, f in enumerate(feats):
                w = (Wego[f] / Wego[f].norm()).to(args.device)
                outs = []
                for sign in (0.0, 1.0):
                    cache = {}
                    def hook(l, x, sign=sign):
                        if l == nL - 1 and sign:
                            x = x.clone()
                            x[:, -1] += alpha * w
                        return x
                    model(tok, cache=cache, resid_hook=hook)
                    outs.append(cache[f"L{nL - 1}.mlp_post"][:, -1, idx].cpu().numpy())
                causal[:, fi] += (outs[1] - outs[0]).sum(0)
    causal /= len(base)
    crows = []
    for k, (n, pref) in enumerate(neurons):
        sc = []
        for c in range(maze.N):
            m = ~np.isnan(templates[c])
            sc.append(np.corrcoef(causal[k][m], templates[c][m])[0, 1] if m.sum() > 5 else np.nan)
        sc = np.array(sc)
        others = np.delete(sc, pref)
        crows.append({"own": float(sc[pref]), "others_mean": float(np.nanmean(others)),
                      "rank_of_own": int((np.nan_to_num(others, nan=-9) >= sc[pref]).sum()) + 1})
        rows[k]["causal_beta"] = causal[k].tolist()
    cown = np.array([r["own"] for r in crows]); coth = np.array([r["others_mean"] for r in crows])
    cranks = np.array([r["rank_of_own"] for r in crows])

    own = np.array([r["own"] for r in rows])
    oth = np.array([r["others_mean"] for r in rows])
    ranks = np.array([r["rank_of_own"] for r in rows])
    out = {"n": len(rows), "own_mean": float(own.mean()), "others_mean": float(oth.mean()),
           "frac_own_best": float((ranks == 1).mean()), "median_rank": float(np.median(ranks)),
           "causal": {"own_mean": float(cown.mean()), "others_mean": float(coth.mean()),
                      "frac_own_best": float((cranks == 1).mean()), "median_rank": float(np.median(cranks)),
                      "alpha": alpha, "n_base": len(base)},
           "keep": keep.tolist(), "rows": rows, "maze": maze.to_dict()}
    print(f"{len(rows)} place neurons: corr(receptive field, own cell's wall template) = {own.mean():.2f} "
          f"vs other cells {oth.mean():.2f}; own cell is the best-matching of 64 for {out['frac_own_best']:.0%} "
          f"(median rank {out['median_rank']:.0f})")
    print(f"CAUSAL receptive fields: corr with own template {cown.mean():.2f} vs other cells {coth.mean():.2f}; "
          f"own cell best of 64 for {(cranks == 1).mean():.0%} (median rank {np.median(cranks):.0f})")
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
