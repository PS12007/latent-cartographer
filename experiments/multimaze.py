"""Experiment 14 — four mazes in one network: remapping and context switching.

One model is trained on walks from 4 different 8x8 mazes. Each sequence opens
with a maze token, then the usual start token, then moves:

    [MAZE_k, start, m_1, ..., m_64]

Questions
  1. Does a single 0.8M-parameter network learn all four maps (KL to the
     per-maze Bayes optimum)?
  2. Remapping: do block-4 place neurons keep their field across mazes, move it,
     or switch off? (Hippocampal place cells "globally remap" between rooms.)
  3. Context switching: with a known start, the position (start + displacement)
     does not depend on the maze. Swap only the maze token: do the predicted
     legal moves follow the *new* maze's walls at the same cell?

    python experiments/multimaze.py --out runs/multi4            # train + analyse
    python experiments/multimaze.py --out runs/multi4 --analyse  # analyse only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.env import BayesFilter, make_maze, sample_walks  # noqa: E402
from cartographer.model import GPT, GPTConfig  # noqa: E402

N_MAZES, SEEDS, T = 4, [10, 11, 12, 13], 64


def maze_token(k, maze):
    return maze.vocab_size + k            # after moves, UNK and the 64 start tokens


def batch(mazes, B, rng, p_known=0.35, k=None):
    ks = rng.integers(len(mazes), size=B) if k is None else np.full(B, k)
    tok = np.empty((B, T + 2), dtype=np.int64)
    cells = np.empty((B, T + 1), dtype=np.int64)
    known = np.empty(B, dtype=bool)
    for j in range(len(mazes)):
        idx = np.where(ks == j)[0]
        if len(idx) == 0:
            continue
        t, c, kn = sample_walks(mazes[j], len(idx), T, p_known, rng)
        tok[idx, 0] = maze_token(j, mazes[j])
        tok[idx, 1:] = t
        cells[idx], known[idx] = c, kn
    return tok, cells, known, ks


@torch.no_grad()
def kl_per_maze(model, mazes, dev, n=1024):
    out = {}
    rng = np.random.default_rng(999)
    for j, mz in enumerate(mazes):
        tok, _, known, _ = batch(mazes, n, rng, p_known=0.5, k=j)
        _, pred = BayesFilter(mz).run(torch.tensor(tok[:, 1:]))
        logp = F.log_softmax(model(torch.tensor(tok[:, :-1], device=dev)), -1)[:, 1:].cpu()
        p = pred[:, :-1].float()
        kl = (p * (p.clamp_min(1e-30).log() - logp)).sum(-1)
        kn = torch.tensor(known)
        out[j] = {"known": kl[kn].mean().item(), "unknown": kl[~kn].mean().item()}
    return out


def train(args, mazes, dev):
    cfg = GPTConfig(vocab_size=mazes[0].vocab_size + N_MAZES, block_size=T + 1, n_layer=4, n_head=4, d_model=128, d_mlp=512)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.05)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(args.steps + 1):
        lr = 1e-3 * min(1, (step + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        tok, _, _, _ = batch(mazes, 256, rng)
        x = torch.tensor(tok[:, :-1], device=dev)
        y = torch.tensor(tok[:, 2:], device=dev)            # position i>=1 predicts move i+1
        loss = F.cross_entropy(model(x)[:, 1:].reshape(-1, 4), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5000 == 0:
            model.eval()
            print(f"step {step} loss {loss.item():.4f} KL {kl_per_maze(model, mazes, dev)} {time.time() - t0:.0f}s", flush=True)
            model.train()
    model.eval()
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "mazes": [m.to_dict() for m in mazes],
                "args": {"T": T, "steps": args.steps}}, os.path.join(args.out, "model.pt"))
    return model


def eta2(x, g, n):
    tot = ((x - x.mean(0)) ** 2).sum(0)
    m = np.zeros((n, x.shape[1])); cnt = np.bincount(g, minlength=n)
    np.add.at(m, g, x); m /= np.maximum(cnt, 1)[:, None]
    return (cnt[:, None] * (m - x.mean(0)) ** 2).sum(0) / np.maximum(tot, 1e-12), m


@torch.no_grad()
def analyse(model, mazes, dev):
    res = {"kl": kl_per_maze(model, mazes, dev)}
    nL = model.cfg.n_layer
    rng = np.random.default_rng(7)
    ratemaps, etas = [], []
    for j, mz in enumerate(mazes):
        tok, cells, known, _ = batch(mazes, 2000, rng, p_known=1.0, k=j)     # known start: always localised
        cache = {}
        acts = []
        for i in range(0, len(tok), 500):
            cache = {}
            model(torch.tensor(tok[i:i + 500, :-1], device=dev), cache=cache)
            acts.append(cache[f"L{nL - 1}.mlp_post"][:, 1:].cpu().numpy())
        X = np.concatenate(acts).reshape(-1, model.cfg.d_mlp)
        c = cells[:, :-1].reshape(-1)
        e, m = eta2(X, c, mz.N)
        ratemaps.append(m); etas.append(e)
    ratemaps, etas = np.stack(ratemaps), np.stack(etas)         # (4, 64, D), (4, D)
    lo, hi = ratemaps.min(1, keepdims=True), ratemaps.max(1, keepdims=True)
    field = ((ratemaps - lo) > 0.5 * (hi - lo)).sum(1)               # (4, D) cells in the field
    place = (etas > 0.5) & (field <= 3)                               # compact place field in that maze
    pref = ratemaps.argmax(1)                                     # (4, D)
    # remapping between every pair of mazes, for neurons tuned in both
    pairs = []
    for a in range(N_MAZES):
        for b in range(a + 1, N_MAZES):
            both = np.where(place[a] & place[b])[0]
            if len(both) == 0:
                continue
            r = [np.corrcoef(ratemaps[a][:, n], ratemaps[b][:, n])[0, 1] for n in both]
            same = (pref[a, both] == pref[b, both]).mean()
            pairs.append({"a": a, "b": b, "n_both": int(len(both)), "field_corr_mean": float(np.mean(r)),
                          "same_pref_cell": float(same)})
    # split-half reliability within one maze, as the ceiling for "same field"
    tok, cells, _, _ = batch(mazes, 4000, np.random.default_rng(8), p_known=1.0, k=0)
    X = []
    for i in range(0, len(tok), 500):
        cache = {}
        model(torch.tensor(tok[i:i + 500, :-1], device=dev), cache=cache)
        X.append(cache[f"L{nL - 1}.mlp_post"][:, 1:].cpu().numpy())
    X = np.concatenate(X)
    half = len(X) // 2
    _, m1 = eta2(X[:half].reshape(-1, X.shape[-1]), cells[:half, :-1].reshape(-1), 64)
    _, m2 = eta2(X[half:].reshape(-1, X.shape[-1]), cells[half:, :-1].reshape(-1), 64)
    rel = [np.corrcoef(m1[:, n], m2[:, n])[0, 1] for n in np.where(place[0])[0]]
    res["remapping"] = {"n_place_per_maze": place.sum(1).tolist(),
                        "n_place_in_k_mazes": np.bincount(place.sum(0), minlength=N_MAZES + 1).tolist(),
                        "pairs": pairs, "within_maze_split_half_corr": float(np.mean(rel)),
                        "chance_same_pref": 1 / 64}
    print("place cells per maze", place.sum(1).tolist(), "| tuned in k mazes", res["remapping"]["n_place_in_k_mazes"])
    for p in pairs:
        print(f"  maze {p['a']} vs {p['b']}: {p['n_both']} shared place cells, field corr {p['field_corr_mean']:.2f}, "
              f"same preferred cell {p['same_pref_cell']:.0%}")
    print(f"  within-maze split-half field corr {res['remapping']['within_maze_split_half_corr']:.2f}")

    # context switching: known start, swap only the maze token
    legal = [torch.tensor(m.open) for m in mazes]
    sw = []
    for a in range(N_MAZES):
        tok, cells, _, _ = batch(mazes, 1000, np.random.default_rng(20 + a), p_known=1.0, k=a)
        for b in range(N_MAZES):
            if a == b:
                continue
            for t in (8, 24, 48):
                tt = tok[:, :t + 2].copy()
                tt[:, 0] = maze_token(b, mazes[b])
                p = F.softmax(model(torch.tensor(tt, device=dev))[:, -1], -1).cpu()
                c = torch.tensor(cells[:, t])
                differ = ~(legal[a][c] == legal[b][c]).all(-1)
                okB = ((p > 0.1) == legal[b][c]).all(-1)
                okA = ((p > 0.1) == legal[a][c]).all(-1)
                sw.append({"a": a, "b": b, "t": t, "follows_new": okB[differ].float().mean().item(),
                           "follows_old": okA[differ].float().mean().item()})
    res["swap"] = sw
    for t in (8, 24, 48):
        r = [s for s in sw if s["t"] == t]
        print(f"swap maze token at move {t}: predictions follow the NEW maze's walls {np.mean([s['follows_new'] for s in r]):.0%}, "
              f"old maze {np.mean([s['follows_old'] for s in r]):.0%}")
    res["mazes"] = [m.to_dict() for m in mazes]
    res["ratemaps_example"] = {}
    multi = np.where(place.sum(0) == N_MAZES)[0]
    for n in multi[:6]:
        res["ratemaps_example"][int(n)] = ratemaps[:, :, n].tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/multi4")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    mazes = [make_maze(8, 8, 0.35, s) for s in SEEDS]
    if args.analyse:
        ck = torch.load(os.path.join(args.out, "model.pt"), map_location="cpu", weights_only=False)
        model = GPT(GPTConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        model.to(args.device).eval()
    else:
        model = train(args, mazes, args.device)
    res = analyse(model, mazes, args.device)
    with open("results/multimaze.json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
