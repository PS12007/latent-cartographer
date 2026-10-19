import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from egocentric import R, ego_targets  # noqa: E402

from cartographer.analysis import make_dataset
from cartographer.env import make_maze
from cartographer.model import GPT, GPTConfig


def test_ego_targets_basic_properties():
    maze = make_maze(seed=0)
    d = make_dataset(maze, 40, 30, 0.0, seed=1)
    V, E = ego_targets(d)
    K = 2 * R + 1
    V = V.reshape(40, 31, K, K)
    E = E.reshape(40, 31, K, K, 2)
    assert V[:, :, R, R].all(), "the current cell is always visited"
    assert not E[:, 0].any(), "no passages walked before the first move"
    # after one move exactly one passage adjacent to the centre has been walked
    for b in range(40):
        m = d["tokens"][b, 1]
        e = E[b, 1]
        if m == 1:   # moved east: passage east of the previous cell, now at offset (0,-1)
            assert e[R, R - 1, 0]
        elif m == 3:  # moved west: passage east of the current cell
            assert e[R, R, 0]
        elif m == 2:  # moved south: passage south of the previous cell (offset (-1,0))
            assert e[R - 1, R, 1]
        else:         # moved north: passage south of the current cell
            assert e[R, R, 1]
        assert e.sum() == 1


def test_walked_passages_are_open_in_the_maze():
    maze = make_maze(seed=2)
    d = make_dataset(maze, 30, 40, 1.0, seed=3)   # known start: we can place the sketch in the maze
    _, E = ego_targets(d)
    K = 2 * R + 1
    E = E.reshape(30, 41, K, K, 2)
    for b in range(30):
        for t in range(0, 41, 5):
            r0, c0 = divmod(int(d["cell"][b, t]), maze.W)
            for i, j, k in zip(*np.where(E[b, t])):
                r, c = r0 + i - R, c0 + j - R
                assert maze.open[r * maze.W + c, 1 if k == 0 else 2]


def test_head_hook_identity_and_mean_ablation_shape():
    cfg = GPTConfig(vocab_size=69, block_size=9, n_layer=2, d_model=32, d_mlp=64)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 69, (2, 9))
    base = model(idx)
    model.blocks[0].attn.head_hook = lambda y: y
    torch.testing.assert_close(model(idx), base)
    model.blocks[0].attn.head_hook = lambda y: torch.zeros_like(y)
    assert not torch.allclose(model(idx), base)
    model.blocks[0].attn.head_hook = None
