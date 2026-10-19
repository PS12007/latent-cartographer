"""Ground-truth latent variables, activation collection, and linear probes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .env import DC, DR, START0, BayesFilter, Maze, sample_walks


def ground_truth(maze: Maze, tokens: np.ndarray, cells: np.ndarray, known: np.ndarray):
    """Every latent quantity we want to look for inside the model.

    For a sequence [start, m_1..m_T], at position t (after reading m_1..m_t):
      cell        true current cell                         (B,S)
      disp        (d_row, d_col) displacement since start     (B,S,2)
      consistent  which start cells s are compatible with the moves so far,
                  i.e. the walk from s never crosses a wall   (B,S,N) bool
                  (for known-start sequences only the given start is consistent)
      belief      exact Bayesian posterior over current cell  (B,S,N)
      legal       legal moves at the true current cell        (B,S,4)
    """
    B, S = tokens.shape
    N = maze.N
    moves = tokens[:, 1:]
    disp = np.zeros((B, S, 2), dtype=np.int64)
    disp[:, 1:, 0] = np.cumsum(DR[moves], axis=1)
    disp[:, 1:, 1] = np.cumsum(DC[moves], axis=1)

    nxt = maze.nxt()
    pos = np.tile(np.arange(N), (B, 1))            # hypothetical position for each start
    ok = np.ones((B, N), dtype=bool)
    ok[known] = False
    ok[known, cells[known, 0]] = True
    consistent = np.empty((B, S, N), dtype=bool)
    consistent[:, 0] = ok
    for t in range(1, S):
        m = np.broadcast_to(moves[:, t - 1][:, None], pos.shape)
        ok &= maze.open[pos, m]
        pos = nxt[pos, m]
        consistent[:, t] = ok

    bf = BayesFilter(maze)
    beliefs, _ = bf.run(torch.tensor(tokens))
    return {
        "cell": cells,
        "disp": disp,
        "consistent": consistent,
        "belief": beliefs.float().numpy(),
        "legal": maze.open[cells],
        "known": known,
    }


def make_dataset(maze: Maze, n: int, T: int, p_known: float, seed: int):
    rng = np.random.default_rng(seed)
    tokens, cells, known = sample_walks(maze, n, T, p_known, rng)
    gt = ground_truth(maze, tokens, cells, known)
    gt["tokens"] = tokens
    return gt


@torch.no_grad()
def collect(model, tokens: np.ndarray, keys=None, batch=512):
    """Run the model and return {key: tensor} of cached activations (CPU, float32)."""
    dev = next(model.parameters()).device
    out: dict[str, list] = {}
    logits_all = []
    for i in range(0, len(tokens), batch):
        cache = {}
        logits = model(torch.tensor(tokens[i:i + batch], device=dev), cache=cache)
        logits_all.append(logits.float().cpu())
        for k, v in cache.items():
            if keys is None or k in keys:
                out.setdefault(k, []).append(v.float().cpu())
    res = {k: torch.cat(v) for k, v in out.items()}
    res["logits"] = torch.cat(logits_all)
    return res


# ----------------------------------------------------------------------------- probes

class LinearProbe(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.lin(x)


def _fit(probe, X, loss_fn, steps=1500, lr=3e-3, wd=1e-4, batch=8192, device="cpu"):
    probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = X.shape[0]
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(n, (min(batch, n),), generator=g)
        loss = loss_fn(probe, idx)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return probe


def fit_belief_probe(X, P, device="cpu", **kw):
    """Softmax-regression probe X (n,d) -> distribution over cells, trained to
    minimise KL(P || probe), i.e. soft-label cross-entropy."""
    X, P = X.to(device), P.to(device)
    probe = LinearProbe(X.shape[1], P.shape[1])
    return _fit(probe, X, lambda pr, i: -(P[i] * F.log_softmax(pr(X[i]), -1)).sum(-1).mean(), device=device, **kw)


def fit_multilabel_probe(X, Y, device="cpu", **kw):
    """Independent logistic probes, X (n,d) -> Y (n,k) in {0,1}."""
    X, Y = X.to(device), Y.float().to(device)
    probe = LinearProbe(X.shape[1], Y.shape[1])
    return _fit(probe, X, lambda pr, i: F.binary_cross_entropy_with_logits(pr(X[i]), Y[i]), device=device, **kw)


def fit_regression_probe(X, Y, ridge=1e-3):
    """Closed-form ridge regression; returns (W, b) with Y ≈ X W + b."""
    X = X.double()
    Y = Y.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc + ridge * len(X) * torch.eye(X.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(A, Xc.T @ Yc)
    b = my - mx @ W
    return W.float(), b.float()


def kl(P, logQ):
    """Mean KL(P || Q) for P (n,k) probabilities and logQ (n,k) log-probabilities."""
    return (P * (P.clamp_min(1e-30).log() - logQ)).sum(-1).mean().item()


def r2(Y, Yhat):
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).mean().item()
