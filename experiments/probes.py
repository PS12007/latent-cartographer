"""Experiment 1 — what does each layer know?

Fits linear probes at every point of the residual stream for the three latent
variables of the "eliminate impossible starts" algorithm:

    displacement (d_row, d_col)   ->  consistent start set  ->  current-cell belief

and compares against baselines (time-only prior, the model's own output logits,
and a randomly initialised network of the same shape).

    python experiments/probes.py --run runs/main
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
from cartographer.analysis import (collect, fit_belief_probe, fit_multilabel_probe,  # noqa: E402
                                   fit_regression_probe, kl, make_dataset, r2)
from cartographer.env import Maze  # noqa: E402
from cartographer.model import GPT, GPTConfig, load_model  # noqa: E402


def stream_points(n_layer):
    pts = [("resid_0", "embed")]
    for l in range(n_layer):
        pts.append((f"L{l}.resid_mid", f"L{l} attn"))
        pts.append((f"resid_{l + 1}", f"L{l} mlp"))
    return pts


def balanced_acc(Y, logits):
    pred = logits > 0
    Y = Y.bool()
    tpr = (pred & Y).sum() / Y.sum().clamp_min(1)
    tnr = (~pred & ~Y).sum() / (~Y).sum().clamp_min(1)
    return ((tpr + tnr) / 2).item()


def run_probes(model, maze, train, test, device, steps=1500):
    """Returns dict point_name -> metrics."""
    L = model.cfg.n_layer
    pts = stream_points(L)
    keys = [k for k, _ in pts]
    A_tr = collect(model, train["tokens"], keys)
    A_te = collect(model, test["tokens"], keys)

    def flat(a):
        return a.reshape(-1, a.shape[-1])

    # targets
    D_tr = torch.tensor(train["disp"]).reshape(-1, 2).float()
    D_te = torch.tensor(test["disp"]).reshape(-1, 2).float()
    C_tr = torch.tensor(train["consistent"]).reshape(-1, maze.N)
    C_te = torch.tensor(test["consistent"]).reshape(-1, maze.N)
    P_tr = torch.tensor(train["belief"]).reshape(-1, maze.N)
    P_te = torch.tensor(test["belief"]).reshape(-1, maze.N)
    cell_te = torch.tensor(test["cell"]).reshape(-1)
    localized_te = P_te.max(-1).values > 0.99
    uncertain_te = ~localized_te
    t_idx = torch.arange(train["tokens"].shape[1]).repeat(len(test["tokens"]))

    out = {}
    for key, label in pts:
        X_tr, X_te = flat(A_tr[key]), flat(A_te[key])
        W, b = fit_regression_probe(X_tr, D_tr)
        disp_r2 = r2(D_te, X_te @ W + b)
        cp = fit_multilabel_probe(X_tr, C_tr, device=device, steps=steps)
        with torch.no_grad():
            c_logits = cp(X_te.to(device)).cpu()
        bp = fit_belief_probe(X_tr, P_tr, device=device, steps=steps)
        with torch.no_grad():
            logq = F.log_softmax(bp(X_te.to(device)).cpu(), -1)
        out[key] = {
            "label": label,
            "disp_r2": disp_r2,
            "consistent_bacc": balanced_acc(C_te, c_logits),
            "belief_kl": kl(P_te, logq),
            "belief_kl_uncertain": kl(P_te[uncertain_te], logq[uncertain_te]),
            "cell_acc_localized": (logq[localized_te].argmax(-1) == cell_te[localized_te]).float().mean().item(),
        }
        print(f"{label:10s} disp R2 {disp_r2:.3f} | starts bacc {out[key]['consistent_bacc']:.3f} | "
              f"belief KL {out[key]['belief_kl']:.3f} (uncertain {out[key]['belief_kl_uncertain']:.3f}) | "
              f"cell acc {out[key]['cell_acc_localized']:.3f}", flush=True)

    # Baseline: best time-only predictor (the average belief at each position)
    mean_by_t = torch.tensor(train["belief"]).mean(0)               # (S, N)
    logq_t = mean_by_t[t_idx].clamp_min(1e-8).log()
    # Baseline: belief decoded from the model's own 4-way output distribution
    lg_tr, lg_te = flat(A_tr["logits"]), flat(A_te["logits"])
    tfeat = lambda n: F.one_hot(torch.arange(train["tokens"].shape[1]).repeat(n), train["tokens"].shape[1]).float()  # noqa: E731
    Xo_tr = torch.cat([F.log_softmax(lg_tr, -1), tfeat(len(train["tokens"]))], -1)
    Xo_te = torch.cat([F.log_softmax(lg_te, -1), tfeat(len(test["tokens"]))], -1)
    hidden = torch.nn.Sequential(torch.nn.Linear(Xo_tr.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, maze.N))
    hidden.to(device)
    opt = torch.optim.AdamW(hidden.parameters(), lr=3e-3)
    Xo_d, P_d = Xo_tr.to(device), P_tr.to(device)
    g = torch.Generator().manual_seed(0)
    for _ in range(3000):
        i = torch.randint(len(Xo_d), (8192,), generator=g)
        loss = -(P_d[i] * F.log_softmax(hidden(Xo_d[i]), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logq_o = F.log_softmax(hidden(Xo_te.to(device)).cpu(), -1)
    baselines = {
        "time_only_kl": kl(P_te, logq_t),
        "time_only_kl_uncertain": kl(P_te[uncertain_te], logq_t[uncertain_te]),
        "output_mlp_kl": kl(P_te, logq_o),
        "output_mlp_kl_uncertain": kl(P_te[uncertain_te], logq_o[uncertain_te]),
    }
    return out, baselines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="results/probes.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    T = ck["args"]["T"]
    train = make_dataset(maze, args.n_train, T, 0.35, seed=101)
    test = make_dataset(maze, args.n_test, T, 0.35, seed=202)

    print("== trained model")
    trained, baselines = run_probes(model, maze, train, test, args.device, args.steps)
    print("baselines:", json.dumps(baselines, indent=1))
    print("== random-init control (same architecture)")
    torch.manual_seed(0)
    rand = GPT(GPTConfig(**ck["config"])).to(args.device).eval()
    random_ctrl, _ = run_probes(rand, maze, train, test, args.device, args.steps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"trained": trained, "random": random_ctrl, "baselines": baselines,
                   "n_layer": model.cfg.n_layer}, f, indent=1)


if __name__ == "__main__":
    main()
