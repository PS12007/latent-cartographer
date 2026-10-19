"""Check: do the place-indexed heads' queries and keys linearly encode displacement? (python experiments/qk_displacement.py)"""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from cartographer.analysis import collect, make_dataset, fit_regression_probe, r2
from cartographer.env import Maze
from cartographer.model import load_model
model, ck = load_model('runs/main/model.pt'); maze = Maze.from_dict(ck['maze'])
d = make_dataset(maze, 1500, 64, 0.0, seed=41)
A = collect(model, d['tokens'], ['resid_1', 'resid_2', 'resid_0'])
D = torch.tensor(d['disp']).reshape(-1, 2).float()
n = len(D) // 2
for l, h in [(0, 1), (1, 1), (2, 1), (1, 0)]:
    blk = model.blocks[l]
    with torch.no_grad():
        x = blk.ln1(A[f'resid_{l}'].reshape(-1, 128))
        q, k, v = blk.attn.qkv(x).split(128, -1)
    dh = 32
    for name, z in (('query', q[:, h*dh:(h+1)*dh]), ('key', k[:, h*dh:(h+1)*dh])):
        W, b = fit_regression_probe(z[:n], D[:n])
        print(f'block {l+1} head {h} {name}: displacement R2 {r2(D[n:], z[n:] @ W + b):.2f}')
