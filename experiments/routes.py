"""Experiment 17 — does the map survive goal-directed data?

Vafa et al. (2024) found that transformers trained on real taxi routes predict
the next turn well yet hold an incoherent map of the city. Uniform random walks
are friendly data: every legal move from every cell is shown, equally often.
Here the same 0.8M-parameter model is trained twice on the main maze, with the
start always given, on two kinds of data:

  walks   uniform random walks (as in the rest of the project)
  depots  the same, but every goal is one of two fixed depots (opposite corners):
          hub-and-spoke traffic, so many moves are rare or never seen.
  mixed   depot traffic with 10% of sequences replaced by random walks.
  routes  chained shortest paths: the walker heads for a hidden random goal along
          a shortest path (ties broken at random); on arrival it picks a new goal.
          Like taxi data: purposeful, and most moves at a cell are never taken
          from most histories.

Tests (all policy-agnostic, so the two models are judged by the same yardstick)
  1. illegal mass   probability the model puts on walls at the true cell, on its own
                    data and on the *other* kind of data (routes model on random
                    walks = the detour test: does it still know the walls?).
  2. cell probe     softmax probe for the current cell at every depth, fitted on the
                    model's own data, tested on both kinds of data.
  3. steering       probe-direction edit to a random other cell c' after block l;
                    mass on c''s walls afterwards (lower = the model moved to c').
  4. map extraction steer to every cell, average the predicted move distribution
                    over many histories, and score how well it separates open sides
                    from walls (AUC over interior sides) and count interior edges right,
                    scored exactly as in intervene.py.

    python experiments/routes.py              # train both + analyse
    python experiments/routes.py --analyse    # analyse saved models
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
from cartographer.analysis import collect, fit_belief_probe  # noqa: E402
from cartographer.env import START0, Maze, make_maze, sample_walks  # noqa: E402
from cartographer.model import GPT, GPTConfig, load_model  # noqa: E402

T = 64
POLICIES = ["walks", "routes", "depots", "mixed"]
DEPOTS = [0, 63]                                      # two opposite corners


def distances(maze):
    nxt = maze.nxt()
    D = np.full((maze.N, maze.N), 10 ** 6, dtype=np.int64)
    for s in range(maze.N):
        D[s, s] = 0
        frontier = [s]
        while frontier:
            new = []
            for a in frontier:
                for m in range(4):
                    if maze.open[a, m] and D[s, nxt[a, m]] > D[s, a] + 1:
                        D[s, nxt[a, m]] = D[s, a] + 1
                        new.append(nxt[a, m])
            frontier = new
    return D


def sample_routes(maze, B, T, rng, D=None, goals=None):
    """Chained shortest paths to hidden goals (random cells, or drawn from `goals`). Returns tokens (B,T+1), cells (B,T+1)."""
    D = distances(maze) if D is None else D
    nxt = maze.nxt()
    cur = rng.integers(maze.N, size=B)
    pick = (lambda n: rng.integers(maze.N, size=n)) if goals is None else (lambda n: rng.choice(goals, size=n))
    goal = pick(B)
    cells = np.empty((B, T + 1), dtype=np.int64)
    moves = np.empty((B, T), dtype=np.int64)
    cells[:, 0] = cur
    ar = np.arange(B)
    for t in range(T):
        while True:                                   # new goal on arrival
            done = goal == cur
            if not done.any():
                break
            goal[done] = pick(done.sum())
        nb = nxt[cur]                                 # (B,4)
        ok = maze.open[cur] & (D[nb, goal[:, None]] == D[cur, goal][:, None] - 1)
        score = np.where(ok, rng.random((B, 4)), -1)
        m = score.argmax(1)
        moves[:, t] = m
        cur = nb[ar, m]
        cells[:, t + 1] = cur
    tokens = np.concatenate([(START0 + cells[:, :1]), moves], 1)
    return tokens, cells


def sample(policy, maze, B, rng, D=None, T=T):
    if policy == "walks":
        tok, cells, _ = sample_walks(maze, B, T, 1.0, rng)
        return tok, cells
    if policy == "mixed":                             # depot traffic with 10% random walks mixed in
        tok, cells = sample_routes(maze, B, T, rng, D, goals=DEPOTS)
        k = rng.random(B) < 0.1
        if k.any():
            tok[k], cells[k] = sample("walks", maze, int(k.sum()), rng, D, T)
        return tok, cells
    return sample_routes(maze, B, T, rng, D, goals=DEPOTS if policy == "depots" else None)


def coverage(policy, maze, D, n=20000):
    """How often each legal (cell, move) appears in training data, relative to uniform walks."""
    tok, cells = sample(policy, maze, n, np.random.default_rng(77), D)
    cnt = np.zeros((maze.N, 4))
    np.add.at(cnt, (cells[:, :-1].reshape(-1), tok[:, 1:].reshape(-1)), 1)
    f = cnt[maze.open] / cnt.sum()
    u = 1 / maze.open.sum()
    return {"never_seen": int((f == 0).sum()), "below_10pct_of_uniform": int((f < 0.1 * u).sum()),
            "n_legal_moves": int(maze.open.sum()), "counts": cnt.tolist()}


def train(policy, maze, D, out, steps, dev):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=maze.vocab_size, block_size=T + 1, n_layer=4, n_head=4, d_model=128, d_mlp=512)
    model = GPT(cfg).to(dev)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.98))
    rng = np.random.default_rng(1)
    t0 = time.time()
    for step in range(steps):
        lr = 1e-3 * min(1, (step + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        tok, _ = sample(policy, maze, 256, rng, D)
        tok = torch.from_numpy(tok).to(dev)
        loss = F.cross_entropy(model(tok[:, :-1]).reshape(-1, 4), tok[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2500 == 0:
            print(f"[{policy}] step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    os.makedirs(out, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "maze": maze.to_dict(),
                "args": {"T": T, "steps": steps, "policy": policy}, "step": steps}, os.path.join(out, "model.pt"))
    return model


@torch.no_grad()
def probs(model, tok, dev, hook=None, bs=1000):
    if hook is not None:
        bs = len(tok)                                 # hooks carry one edit per sequence
    out = []
    for i in range(0, len(tok), bs):
        x = torch.as_tensor(tok[i:i + bs], device=dev)
        out.append(F.softmax(model(x, resid_hook=hook).float(), -1).cpu())
    return torch.cat(out)


def illegal_mass(p, cells, legal):
    """p (B,S,4) predictions after each prefix; cells (B,S) true cell. Mean mass on walls."""
    wall = ~legal[torch.as_tensor(cells)]
    m = (p * wall).sum(-1)
    return m.mean().item(), (m > 0.05).float().mean().item()


def analyse(models, maze, D, dev, n_map=300, t_steer=40):
    torch.manual_seed(0)
    legal = torch.tensor(maze.open)
    nL = 4
    data = {}
    for j, pol in enumerate(POLICIES):
        rng = np.random.default_rng(500 + j)
        data[pol] = {"train": sample(pol, maze, 1500, rng, D), "test": sample(pol, maze, 1500, rng, D)}
    res = {}
    for pol, model in models.items():
        r = {"illegal": {}, "probe": {}, "steer": {}}
        # 1. illegal mass on own and other data
        for dpol in POLICIES:
            tok, cells = data[dpol]["test"]
            p = probs(model, tok[:, :-1], dev)
            mass, frac = illegal_mass(p, cells[:, :-1], legal)
            r["illegal"][dpol] = {"mass": mass, "frac_over_5pct": frac}
        # 2. cell probes at every depth (fit on own data)
        keys = [f"resid_{l}" for l in range(nL + 1)]
        tok_tr, cells_tr = data[pol]["train"]
        A_tr = collect(model, tok_tr[:, :-1], keys)
        Y = F.one_hot(torch.tensor(cells_tr[:, :-1]).reshape(-1), maze.N).float()
        A_te = {d: collect(model, data[d]["test"][0][:, :-1], keys) for d in POLICIES}
        probes = {}
        for l in range(nL + 1):
            k = keys[l]
            Xtr = A_tr[k].reshape(-1, A_tr[k].shape[-1])
            pr = fit_belief_probe(Xtr, Y, device=dev).cpu()
            probes[l] = pr
            means = torch.stack([Xtr[Y[:, c] > 0].mean(0) for c in range(maze.N)])
            accs = {}
            for d in POLICIES:
                X = A_te[d][k].reshape(-1, Xtr.shape[-1])
                c = torch.tensor(data[d]["test"][1][:, :-1]).reshape(-1)
                with torch.no_grad():
                    accs[d] = (pr(X).argmax(-1) == c).float().mean().item()
            r["probe"][l] = accs
            # 3. probe-direction steering at the newest token, own data, prefixes of length t_steer
            if l in (1, 2, 3):
                tok = data[pol]["test"][0][:, :t_steer + 1]
                cell = torch.tensor(data[pol]["test"][1][:, t_steer])
                X = A_te[pol][k][:, t_steer]
                g = torch.Generator().manual_seed(l)
                tgt = (cell + torch.randint(1, maze.N, (len(tok),), generator=g)) % maze.N   # any other cell
                W = pr.lin.weight.detach()
                md = (means[tgt] - means[cell]).norm(dim=-1, keepdim=True)
                pdir = W[tgt] - W[cell]
                pdir = pdir / pdir.norm(dim=-1, keepdim=True)
                rnd = torch.randn(X.shape, generator=g)
                rnd = rnd / rnd.norm(dim=-1, keepdim=True)
                sr = {}
                for name, vec in [("none", X), ("probe_dir", X + 2 * md * pdir), ("random", X + 2 * md * rnd)]:
                    def hook(li, x, vec=vec, l=l):
                        if li == l:
                            x = x.clone()
                            x[:, -1] = vec.to(x.device)
                        return x
                    p = probs(model, tok, dev, hook)[:, -1]
                    sr[name] = {"mass_on_target_walls": (p * ~legal[tgt]).sum(-1).mean().item(),
                                "mass_on_true_walls": (p * ~legal[cell]).sum(-1).mean().item()}
                r["steer"][l] = sr
        # 4. map extraction: steer every cell over n_map histories at the best block
        best = min(r["steer"], key=lambda l: r["steer"][l]["probe_dir"]["mass_on_target_walls"])
        k = keys[best]
        W = probes[best].lin.weight.detach()
        Xtr = A_tr[k].reshape(-1, A_tr[k].shape[-1])
        means = torch.stack([Xtr[Y[:, c] > 0].mean(0) for c in range(maze.N)])
        tok = data[pol]["test"][0][:n_map, :t_steer + 1]
        cell = torch.tensor(data[pol]["test"][1][:n_map, t_steer])
        X = A_te[pol][k][:n_map, t_steer]
        P = torch.zeros(maze.N, 4)
        for c2 in range(maze.N):
            md = (means[c2] - means[cell]).norm(dim=-1, keepdim=True)
            pdir = W[c2] - W[cell]
            ok = cell != c2
            pdir = pdir / pdir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            vec = torch.where(ok[:, None], X + 2 * md * pdir, X)

            def hook(li, x, vec=vec):
                if li == best:
                    x = x.clone()
                    x[:, -1] = vec.to(x.device)
                return x
            P[c2] = probs(model, tok, dev, hook)[:, -1][ok].mean(0)
        rr, cc = np.divmod(np.arange(maze.N), maze.W)
        interior = np.stack([rr > 0, cc < maze.W - 1, rr < maze.H - 1, cc > 0], 1)
        s, y = P.numpy()[interior], maze.open[interior]
        auc = float((s[y][:, None] > s[~y][None, :]).mean() + 0.5 * (s[y][:, None] == s[~y][None, :]).mean())
        # same scoring as intervene.py: each interior edge is open iff the mean of both views > 0.1
        Pn, hit, inter = P.numpy(), 0, 0
        for c in range(maze.N):
            r_, q = divmod(c, maze.W)
            for mv, okk in ((1, q + 1 < maze.W), (2, r_ + 1 < maze.H)):
                if okk:
                    nb = c + (1 if mv == 1 else maze.W)
                    inter += 1
                    hit += int((0.5 * (Pn[c, mv] + Pn[nb, (mv + 2) % 4]) > 0.1) == maze.open[c, mv])
        r["map"] = {"block": int(best), "auc_interior_sides": auc, "edges_correct": hit, "n_edges": inter, "P": P.tolist()}
        res[pol] = r
        print(f"\n== model trained on {pol} ==")
        for d in POLICIES:
            print(f"  on {d:6s}: mass on walls {r['illegal'][d]['mass']:.4f} | positions with >5% on a wall "
                  f"{r['illegal'][d]['frac_over_5pct']:.1%} | cell probe by depth "
                  + " ".join(f"{r['probe'][l][d]:.2f}" for l in range(nL + 1)))
        for l, sr in r["steer"].items():
            print(f"  steer after block {l}: mass on target walls none {sr['none']['mass_on_target_walls']:.3f} "
                  f"-> probe-dir {sr['probe_dir']['mass_on_target_walls']:.3f} (random {sr['random']['mass_on_target_walls']:.3f})")
        m = r["map"]
        print(f"  map from steering (block {best}): AUC open-vs-wall {m['auc_interior_sides']:.3f} | "
              f"interior edges correct {m['edges_correct']}/{m['n_edges']}", flush=True)
    return res


@torch.no_grad()
def detour_split(model, maze, counts, dev, n=3000):
    """Random walks scored against the training traffic: wall mass before vs after the walk
    first takes a move that was rare (<10% of uniform) in this model's training data."""
    legal = torch.tensor(maze.open)
    cnt = np.array(counts)
    f = cnt / cnt.sum()
    rare = maze.open & (f < 0.1 / maze.open.sum())
    tok, cells, _ = sample_walks(maze, n, T, 1.0, np.random.default_rng(91))
    took_rare = rare[cells[:, :-1], tok[:, 1:]]                       # (n, T): move t+1 was rare
    first = np.where(took_rare.any(1), took_rare.argmax(1), T)        # index of first rare move
    p = probs(model, tok[:, :-1], dev)                                 # prediction after prefix t
    m = (p * ~legal[torch.as_tensor(cells[:, :-1])]).sum(-1).numpy()   # (n, T)
    t = np.arange(T)[None]
    before = t <= first[:, None]                                       # prefix still on familiar moves
    since = t - first[:, None] - 1                                     # moves since the first rare one
    by_since = [float(m[since == k].mean()) for k in range(0, 12)]
    return {"mass_before_first_rare": float(m[before].mean()), "mass_after_first_rare": float(m[~before].mean()),
            "mass_by_moves_since_rare": by_since, "median_first_rare_step": float(np.median(first))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/routes")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--results", default="results/routes.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    maze = make_maze(8, 8, 0.35, 0)                    # the main maze
    D = distances(maze)
    models = {}
    for pol in POLICIES:
        path = os.path.join(args.out, pol)
        if args.analyse or os.path.exists(os.path.join(path, "model.pt")):
            models[pol] = load_model(os.path.join(path, "model.pt"))[0].to(args.device).eval()
        else:
            models[pol] = train(pol, maze, D, path, args.steps, args.device)
    res = analyse(models, maze, D, args.device)
    for pol in POLICIES:
        res[pol]["coverage"] = coverage(pol, maze, D)
        c = res[pol]["coverage"]
        print(f"{pol}: legal moves never seen {c['never_seen']}/{c['n_legal_moves']}, rare (<10% of uniform) {c['below_10pct_of_uniform']}")
        res[pol]["detour_split"] = d = detour_split(models[pol], maze, c["counts"], args.device) if c["below_10pct_of_uniform"] else None
        if d:
            print(f"  {pol} on random walks: mass on walls {d['mass_before_first_rare']:.4f} before the first rare move, "
                  f"{d['mass_after_first_rare']:.3f} after (median first rare move at step {d['median_first_rare_step']:.0f}); "
                  f"by moves since: " + " ".join(f"{v:.2f}" for v in d["mass_by_moves_since_rare"][:6]))
    res["depots"]["depot_cells"] = DEPOTS
    res["mean_route_len"] = float(D[D < 10 ** 6].mean())
    with open(args.results, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
