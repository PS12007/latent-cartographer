"""Experiment 5 — how is the location computed?

Hypothesis (a parallel algorithm a transformer can implement in a few layers):
  1. dead reckoning: each token computes its displacement since the start
     (one attention layer averaging move vectors, rescaled by position);
  2. local evidence: token k works out which start cells its *own* move m_k
     rules out — start s is excluded if moving m_k from s + disp(k-1) hits a wall.
     This needs only (disp(k-1), m_k), so it can be done in place by an MLP;
  3. aggregation: later attention averages these exclusion masks over the history;
     a start survives if nobody excluded it. Shifting the survivors by disp(t)
     gives the current-cell belief.

This script measures
  * per-head attention statistics (how uniform; how much on the start token),
  * linear decodability of the *single-step* exclusion mask at every layer,
  * a nonlinear (MLP) probe for the *cumulative* consistent-start set, to test
    whether that information is present but not linearly readable.

    python experiments/mechanism.py --run runs/main
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
from cartographer.analysis import collect, fit_multilabel_probe, make_dataset  # noqa: E402
from cartographer.env import Maze  # noqa: E402
from cartographer.model import GPT, GPTConfig, load_model  # noqa: E402
from probes import balanced_acc, stream_points  # noqa: E402


def local_exclusion(maze, gt):
    """excl[b, k, s] = True if move m_k (the k-th token) is illegal from s + disp(k-1).
    Defined for k >= 1; row 0 is all False. Only meaningful for unknown-start walks."""
    tok, disp = gt["tokens"], gt["disp"]
    B, S = tok.shape
    r0, c0 = np.divmod(np.arange(maze.N), maze.W)
    excl = np.zeros((B, S, maze.N), dtype=bool)
    for k in range(1, S):
        r = r0[None] + disp[:, k - 1, 0:1]
        c = c0[None] + disp[:, k - 1, 1:2]
        inside = (r >= 0) & (r < maze.H) & (c >= 0) & (c < maze.W)
        cell = np.clip(r, 0, maze.H - 1) * maze.W + np.clip(c, 0, maze.W - 1)
        m = np.broadcast_to(tok[:, k:k + 1], cell.shape)
        excl[:, k] = inside & ~maze.open[cell, m]
    return excl


def attention_stats(model, tokens, device):
    A = collect(model, tokens, [f"L{l}.attn" for l in range(model.cfg.n_layer)])
    S = tokens.shape[1]
    out = {}
    for l in range(model.cfg.n_layer):
        a = A[f"L{l}.attn"]                          # (B,H,S,S)
        for h in range(a.shape[1]):
            rows = a[:, h, 8:]                        # queries at t >= 8
            t = torch.arange(8, S).float()
            ent = -(rows.clamp_min(1e-12).log() * rows).sum(-1)       # (B, S-8)
            norm_ent = (ent / torch.log(t + 1)).mean().item()          # 1 = uniform over the prefix
            to_start = rows[..., 0].mean().item()
            to_self = torch.diagonal(a[:, h], dim1=-2, dim2=-1)[:, 8:].mean().item()
            # how much of the attention lands on the previous token
            to_prev = torch.diagonal(a[:, h], offset=-1, dim1=-2, dim2=-1)[:, 7:].mean().item()
            out[f"L{l}H{h}"] = {"norm_entropy": norm_ent, "to_start": to_start, "to_self": to_self, "to_prev": to_prev}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/mechanism.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(dev)
    maze = Maze.from_dict(ck["maze"])
    T = ck["args"]["T"]
    # unknown-start walks only: that is where localisation happens
    train = make_dataset(maze, 4000, T, 0.0, seed=111)
    test = make_dataset(maze, 800, T, 0.0, seed=222)

    res = {"attention": attention_stats(model, test["tokens"][:400], dev)}
    for k, v in res["attention"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})

    E_tr = torch.tensor(local_exclusion(maze, train))[:, 1:].reshape(-1, maze.N)
    E_te = torch.tensor(local_exclusion(maze, test))[:, 1:].reshape(-1, maze.N)
    C_tr = torch.tensor(train["consistent"])[:, 1:].reshape(-1, maze.N)
    C_te = torch.tensor(test["consistent"])[:, 1:].reshape(-1, maze.N)
    print(f"single-step masks exclude {E_te.float().sum(1).mean():.1f} starts on average; "
          f"consistent set averages {C_te.float().sum(1).mean():.1f}")

    def probe_all(m, tag):
        pts = stream_points(m.cfg.n_layer)
        keys = [k for k, _ in pts]
        A_tr = collect(m, train["tokens"], keys)
        A_te = collect(m, test["tokens"], keys)
        rows = {}
        for k, label in pts:
            X_tr = A_tr[k][:, 1:].reshape(-1, A_tr[k].shape[-1])
            X_te = A_te[k][:, 1:].reshape(-1, A_te[k].shape[-1])
            pe = fit_multilabel_probe(X_tr, E_tr, device=dev, steps=1500)
            with torch.no_grad():
                le = pe(X_te.to(dev)).cpu()
            # nonlinear probe for the cumulative consistent set
            mlp = torch.nn.Sequential(torch.nn.Linear(X_tr.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, maze.N)).to(dev)
            opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3)
            Xd, Yd = X_tr.to(dev), C_tr.float().to(dev)
            g = torch.Generator().manual_seed(0)
            for _ in range(2500):
                i = torch.randint(len(Xd), (8192,), generator=g)
                loss = F.binary_cross_entropy_with_logits(mlp(Xd[i]), Yd[i])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                lc = mlp(X_te.to(dev)).cpu()
            rows[k] = {"label": label, "excl_bacc": balanced_acc(E_te, le), "consistent_mlp_bacc": balanced_acc(C_te, lc)}
            print(f"[{tag}] {label:10s} single-step exclusion bacc {rows[k]['excl_bacc']:.3f} | "
                  f"cumulative consistent (MLP probe) bacc {rows[k]['consistent_mlp_bacc']:.3f}", flush=True)
        return rows

    res["trained"] = probe_all(model, "trained")
    torch.manual_seed(0)
    res["random"] = probe_all(GPT(GPTConfig(**ck["config"])).to(dev).eval(), "random")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
