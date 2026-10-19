"""Experiment 2 — is the decoded map *used*? Causal edits, map extraction, persistence.

Setup: walks of length t where the exact posterior is certain the walker is at
cell c. We edit the residual stream *at the last position only*, entering
block l, so that it should mean "you are at c'" (a random cell whose open sides
differ from c's), and check whether the model's next-move prediction now
follows the walls of c'. A prediction counts as correct when the set of moves
given > 10% probability equals the set of open sides of c'.

Edits compared at each depth:
  full_patch   replace the whole vector with one from another walk that ends at c'
               (upper bound; tells us *where* location information lives)
  mean_diff    x + mean[c'] - mean[c]      (class-conditional means of activations)
  probe_dir    x + α·|mean_diff|·unit(W[c'] - W[c])   (direction from the belief probe)
  random       random direction, same norm as probe_dir at that α (control)
  probe_pinv   minimum-norm edit making the probe's logits swap c and c' (fails; kept as a lesson)

Then:
  map extraction  steer to every cell in turn and record which moves the model
                  allows -> redraw the maze from the model alone.
  persistence     steer at step t, append a move that is legal at c', and ask whether
                  the prediction at t+1 follows the walls of next(c', move).

    python experiments/intervene.py --run runs/main
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
from cartographer.analysis import collect, fit_belief_probe, make_dataset  # noqa: E402
from cartographer.env import Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402

ALPHAS = [0.5, 1, 2, 4, 8]


@torch.no_grad()
def predict(model, tokens, layer, new_last, device, pos=-1, read=-1):
    """Replace resid entering block `layer` at position `pos` with new_last (B,D); return P(next move) at `read`."""
    tok = torch.as_tensor(tokens, device=device)
    v = new_last.to(device)

    def hook(l, x):
        if l == layer:
            x = x.clone()
            x[:, pos] = v
        return x

    return F.softmax(model(tok, resid_hook=hook)[:, read].float(), -1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--map_alpha", type=float, default=2.0)
    ap.add_argument("--out", default="results/intervene.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = args.device

    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(dev)
    maze = Maze.from_dict(ck["maze"])
    legal = torch.tensor(maze.open)
    nxt = torch.tensor(maze.nxt())
    nL = model.cfg.n_layer

    train = make_dataset(maze, 3000, ck["args"]["T"], 0.35, seed=101)
    test = make_dataset(maze, args.n, args.t, 0.35, seed=303)
    keep = test["belief"][:, -1].max(-1) > 0.99
    tokens = test["tokens"][keep]
    cell = torch.tensor(test["cell"][keep, -1])
    B = len(tokens)

    # targets: random cell with a different set of open sides; donors for full patching
    tgt = torch.empty(B, dtype=torch.long)
    for i in range(B):
        while True:
            c2 = int(rng.integers(maze.N))
            if not (maze.open[c2] == maze.open[int(cell[i])]).all():
                tgt[i] = c2
                break
    donor = torch.empty(B, dtype=torch.long)
    by_cell = {c: torch.where(cell == c)[0] for c in range(maze.N)}
    for i in range(B):
        pool = by_cell[int(tgt[i])]
        donor[i] = pool[int(rng.integers(len(pool)))] if len(pool) else -1
    has_donor = donor >= 0

    def acc(p, target, mask=None):
        ok = ((p > 0.1) == legal[target]).all(-1)
        return (ok[mask] if mask is not None else ok).float().mean().item()

    keys = [f"resid_{l}" for l in range(nL + 1)]
    A_test = collect(model, tokens, keys)
    A_train = collect(model, train["tokens"], keys)
    P_train = torch.tensor(train["belief"]).reshape(-1, maze.N)
    loc_train = P_train.max(-1).values > 0.99
    cell_train = torch.tensor(train["cell"]).reshape(-1)

    res = {"n": B, "t": args.t, "alphas": ALPHAS, "layers": {}}
    probes = {}
    for l in range(nL + 1):
        k = keys[l]
        X = A_test[k][:, -1]
        Xtr = A_train[k].reshape(-1, X.shape[1])
        probe = fit_belief_probe(Xtr, P_train, device=dev).cpu()
        W, b = probe.lin.weight.detach(), probe.lin.bias.detach()
        probes[l] = (W, b)
        means = torch.stack([Xtr[loc_train & (cell_train == c)].mean(0) for c in range(maze.N)])
        md = means[tgt] - means[cell]
        mdn = md.norm(dim=-1, keepdim=True)
        pdir = W[tgt] - W[cell]
        pdir = pdir / pdir.norm(dim=-1, keepdim=True)
        r = {
            "none_vs_original": acc(predict(model, tokens, l, X, dev), cell),
            "full_patch": acc(predict(model, tokens, l, X[donor.clamp_min(0)], dev), tgt, has_donor),
            "mean_diff": acc(predict(model, tokens, l, X + md, dev), tgt),
            "probe_dir": {}, "random": {},
            "resid_norm": X.norm(dim=-1).mean().item(), "mean_diff_norm": mdn.mean().item(),
        }
        z = X @ W.T + b
        ar = torch.arange(B)
        z2 = z.clone()
        z2[ar, cell], z2[ar, tgt] = z[ar, tgt], z[ar, cell]
        r["probe_pinv"] = acc(predict(model, tokens, l, X + (z2 - z) @ torch.linalg.pinv(W).T, dev), tgt)
        for a in ALPHAS:
            r["probe_dir"][a] = acc(predict(model, tokens, l, X + a * mdn * pdir, dev), tgt)
            rnd = torch.randn_like(X)
            rnd = rnd / rnd.norm(dim=-1, keepdim=True) * a * mdn
            r["random"][a] = acc(predict(model, tokens, l, X + rnd, dev), tgt)
        res["layers"][l] = r
        print(f"entering block {l}: patch {r['full_patch']:.3f} | mean-diff {r['mean_diff']:.3f} | "
              f"probe-dir " + " ".join(f"{a}:{v:.2f}" for a, v in r["probe_dir"].items()) +
              f" | random@2 {r['random'][2]:.3f} | pinv {r['probe_pinv']:.3f}", flush=True)

    # ---------------------------------------------------------------- map extraction
    best = max(range(nL + 1), key=lambda l: res["layers"][l]["probe_dir"][args.map_alpha])
    W, _ = probes[best]
    X = A_test[keys[best]][:, -1]
    Xtr = A_train[keys[best]].reshape(-1, X.shape[1])
    means = torch.stack([Xtr[loc_train & (cell_train == c)].mean(0) for c in range(maze.N)])
    sub = torch.arange(min(B, 400))
    recon = np.zeros((maze.N, 4))
    for c2 in range(maze.N):
        t2 = torch.full((len(sub),), c2)
        src = cell[sub]
        pdir = W[t2] - W[src]
        pdir = pdir / pdir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        mdn = (means[t2] - means[src]).norm(dim=-1, keepdim=True)
        p = predict(model, tokens[sub.numpy()], best, X[sub] + args.map_alpha * mdn * pdir, dev)
        p[src == c2] = float("nan")
        recon[c2] = np.nanmean(p.numpy(), 0)
    pred_open = recon > 0.1
    inter = hit = 0
    for c in range(maze.N):
        r_, q = divmod(c, maze.W)
        for m, ok in ((1, q + 1 < maze.W), (2, r_ + 1 < maze.H)):
            if ok:
                inter += 1
                # a passage exists iff both sides agree it is open; use the mean of both views
                nb = c + (1 if m == 1 else maze.W)
                vote = 0.5 * (recon[c, m] + recon[nb, (m + 2) % 4]) > 0.1
                hit += int(vote == maze.open[c, m])
    res["map"] = {"layer": best, "alpha": args.map_alpha, "recon": recon.tolist(),
                  "interior_correct": hit, "interior_total": inter,
                  "cell_sides_correct": float((pred_open == maze.open).mean())}
    print(f"map extraction (entering block {best}, alpha {args.map_alpha}): {hit}/{inter} interior walls correct", flush=True)

    # ---------------------------------------------------------------- persistence
    # steer at the last position, then append one move legal at the *target* cell
    pers = {}
    for l in range(nL + 1):
        W, _ = probes[l]
        X = A_test[keys[l]][:, -1]
        Xtr = A_train[keys[l]].reshape(-1, X.shape[1])
        means = torch.stack([Xtr[loc_train & (cell_train == c)].mean(0) for c in range(maze.N)])
        pdir = W[tgt] - W[cell]
        pdir = pdir / pdir.norm(dim=-1, keepdim=True)
        mdn = (means[tgt] - means[cell]).norm(dim=-1, keepdim=True)
        mv = torch.tensor([int(rng.choice(np.where(maze.open[int(c)])[0])) for c in tgt])
        tok2 = np.concatenate([tokens, mv.numpy()[:, None]], 1)
        after = nxt[tgt, mv]
        p = predict(model, tok2, l, X + args.map_alpha * mdn * pdir, dev, pos=-2, read=-1)
        p0 = predict(model, tok2, l, X, dev, pos=-2, read=-1)
        pers[l] = {"steered": acc(p, after), "unsteered": acc(p0, after)}
        print(f"persistence entering block {l}: next-step follows steered map {pers[l]['steered']:.3f} "
              f"(unsteered {pers[l]['unsteered']:.3f})", flush=True)
    res["persistence"] = pers
    res["maze"] = maze.to_dict()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
