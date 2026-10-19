"""Experiment 3 — when does the map appear during training?

Re-runs the layer-wise probes on every training snapshot.

    python experiments/dynamics.py --run runs/main
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.analysis import make_dataset  # noqa: E402
from cartographer.env import BayesFilter, Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402
from cartographer.train import evaluate  # noqa: E402
from probes import run_probes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="results/dynamics.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    snaps = sorted(glob.glob(os.path.join(args.run, "snapshots", "step*.pt")))
    _, ck = load_model(snaps[0])
    maze = Maze.from_dict(ck["maze"])
    T = ck["args"]["T"]
    train = make_dataset(maze, 2500, T, 0.35, seed=101)
    test = make_dataset(maze, 600, T, 0.35, seed=202)
    bf = BayesFilter(maze, device=args.device)
    rows = []
    for p in snaps:
        model, ck = load_model(p)
        model.to(args.device)
        print("==", os.path.basename(p), flush=True)
        ev = evaluate(model, maze, bf, T=T)
        pr, _ = run_probes(model, maze, train, test, args.device, steps=800)
        rows.append({"step": ck["step"], "kl_known": ev["kl_known"], "kl_unknown": ev["kl_unknown"], "probes": pr})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
