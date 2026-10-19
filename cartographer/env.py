"""Maze world, random-walk data, and the exact Bayesian filter.

The transformer only ever sees *move tokens* (N/E/S/W) plus a start token that
says either "you start at cell c" or "start unknown". It is never shown the
maze, coordinates, or walls. Everything it knows about the world has to be
inferred from which move sequences occur.

Because the walker picks uniformly among legal moves, the Bayes-optimal next-move
distribution is computable exactly with a discrete filter over cells. That gives
us ground truth for (a) the best achievable loss and (b) the belief state the
model *should* be tracking internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MOVES = "NESW"
DR = np.array([-1, 0, 1, 0])  # row delta per move
DC = np.array([0, 1, 0, -1])  # col delta per move
N_MOVES = 4
UNK = 4          # token: start position unknown
START0 = 5       # tokens START0 + c: start at cell c


@dataclass
class Maze:
    H: int
    W: int
    open: np.ndarray  # (N, 4) bool, open[c, m] = move m from cell c is legal
    seed: int = 0

    @property
    def N(self) -> int:
        return self.H * self.W

    @property
    def vocab_size(self) -> int:
        return START0 + self.N

    def rc(self, c):
        return np.divmod(c, self.W)

    def nxt(self) -> np.ndarray:
        """(N, 4) next-cell table; illegal moves map to the same cell."""
        r, c = np.divmod(np.arange(self.N), self.W)
        out = np.empty((self.N, 4), dtype=np.int64)
        for m in range(4):
            rr = np.clip(r + DR[m], 0, self.H - 1)
            cc = np.clip(c + DC[m], 0, self.W - 1)
            out[:, m] = np.where(self.open[:, m], rr * self.W + cc, np.arange(self.N))
        return out

    def degree(self) -> np.ndarray:
        return self.open.sum(1)

    def to_dict(self) -> dict:
        return {"H": self.H, "W": self.W, "seed": self.seed,
                "open": self.open.astype(int).tolist()}

    @staticmethod
    def from_dict(d: dict) -> "Maze":
        return Maze(d["H"], d["W"], np.array(d["open"], dtype=bool), d.get("seed", 0))

    def ascii(self, values=None) -> str:
        """Render the maze as ASCII; optional per-cell single chars."""
        lines = ["+" + "---+" * self.W]
        for r in range(self.H):
            row, floor = "|", "+"
            for c in range(self.W):
                i = r * self.W + c
                ch = " " if values is None else str(values[i])[:1]
                row += f" {ch} " + (" " if self.open[i, 1] else "|")
                floor += ("   " if self.open[i, 2] else "---") + "+"
            lines += [row, floor]
        return "\n".join(lines)


def make_maze(H: int = 8, W: int = 8, braid: float = 0.35, seed: int = 0) -> Maze:
    """Randomized-DFS perfect maze, then knock out a fraction of the remaining
    interior walls so the maze has loops (a perfect maze is a tree, which makes
    walks get stuck in dead ends and makes localisation unusually easy)."""
    rng = np.random.default_rng(seed)
    N = H * W
    op = np.zeros((N, 4), dtype=bool)
    seen = np.zeros(N, dtype=bool)
    stack = [int(rng.integers(N))]
    seen[stack[0]] = True
    while stack:
        cur = stack[-1]
        r, c = divmod(cur, W)
        opts = []
        for m in range(4):
            rr, cc = r + DR[m], c + DC[m]
            if 0 <= rr < H and 0 <= cc < W and not seen[rr * W + cc]:
                opts.append(m)
        if not opts:
            stack.pop()
            continue
        m = opts[rng.integers(len(opts))]
        nb = (r + DR[m]) * W + (c + DC[m])
        op[cur, m] = True
        op[nb, (m + 2) % 4] = True
        seen[nb] = True
        stack.append(nb)
    # braid: remove some remaining interior walls (only consider E and S to avoid double counting)
    walls = []
    for cell in range(N):
        r, c = divmod(cell, W)
        if c + 1 < W and not op[cell, 1]:
            walls.append((cell, 1))
        if r + 1 < H and not op[cell, 2]:
            walls.append((cell, 2))
    rng.shuffle(walls)
    for cell, m in walls[: int(round(braid * len(walls)))]:
        nb = cell + (1 if m == 1 else W)
        op[cell, m] = True
        op[nb, (m + 2) % 4] = True
    return Maze(H, W, op, seed)


def sample_walks(maze: Maze, B: int, T: int, p_known: float, rng: np.random.Generator):
    """Sample B uniform random walks of T moves.

    Returns
      tokens: (B, T+1) int64 — [start_token, m_1, ..., m_T]
      cells:  (B, T+1) int64 — true cell after each prefix (cells[:,0] = start)
      known:  (B,) bool
    """
    nxt = maze.nxt()
    cells = np.empty((B, T + 1), dtype=np.int64)
    moves = np.empty((B, T), dtype=np.int64)
    cur = rng.integers(maze.N, size=B)
    cells[:, 0] = cur
    for t in range(T):
        legal = maze.open[cur]
        m = np.argmax(rng.random((B, 4)) * legal, axis=1)
        moves[:, t] = m
        cur = nxt[cur, m]
        cells[:, t + 1] = cur
    known = rng.random(B) < p_known
    start_tok = np.where(known, START0 + cells[:, 0], UNK)
    tokens = np.concatenate([start_tok[:, None], moves], axis=1)
    return tokens, cells, known


class BayesFilter:
    """Exact posterior over the walker's cell given the token prefix."""

    def __init__(self, maze: Maze, device="cpu"):
        self.maze = maze
        N = maze.N
        deg = maze.degree().astype(np.float64)
        L = maze.open / deg[:, None]  # L[c, m] = P(move m | cell c)
        nxt = maze.nxt()
        Tm = np.zeros((4, N, N))
        for m in range(4):
            for c in range(N):
                if maze.open[c, m]:
                    Tm[m, c, nxt[c, m]] = L[c, m]
        self.L = torch.tensor(L, dtype=torch.float64, device=device)
        self.Tm = torch.tensor(Tm, dtype=torch.float64, device=device)

    @torch.no_grad()
    def run(self, tokens: torch.Tensor):
        """tokens (B, S) -> beliefs (B, S, N), predictive (B, S, 4).

        beliefs[:, t] is the posterior over the current cell after reading
        tokens[:, :t+1]; predictive[:, t] is P(next move | prefix) — exactly the
        distribution the LM should output at position t."""
        B, S = tokens.shape
        N = self.maze.N
        tok = tokens.to(self.L.device)
        b = torch.full((B, N), 1.0 / N, dtype=torch.float64, device=self.L.device)
        known = tok[:, 0] >= START0
        if known.any():
            onehot = torch.zeros(int(known.sum()), N, dtype=torch.float64, device=self.L.device)
            onehot[torch.arange(onehot.shape[0]), tok[known, 0] - START0] = 1.0
            b[known] = onehot
        beliefs = torch.empty(B, S, N, dtype=torch.float64, device=self.L.device)
        beliefs[:, 0] = b
        for t in range(1, S):
            m = tok[:, t]
            b = torch.bmm(b.unsqueeze(1), self.Tm[m]).squeeze(1)
            b = b / b.sum(1, keepdim=True).clamp_min(1e-300)
            beliefs[:, t] = b
        pred = beliefs @ self.L
        return beliefs, pred


def belief_entropy(beliefs: torch.Tensor) -> torch.Tensor:
    p = beliefs.clamp_min(1e-30)
    return -(beliefs * p.log()).sum(-1)
