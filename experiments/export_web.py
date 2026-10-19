"""Export model weights + per-layer belief probes for the in-browser demo.

Writes docs/assets/model.json (weights as base64 float32) and a set of
reference activations used by docs/js/selftest to verify that the JavaScript
forward pass matches PyTorch.

    python experiments/export_web.py --run runs/main
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.analysis import collect, fit_belief_probe, fit_multilabel_probe, make_dataset  # noqa: E402
from cartographer.env import Maze  # noqa: E402
from cartographer.model import load_model  # noqa: E402


def b64(t: torch.Tensor) -> dict:
    a = t.detach().float().cpu().numpy().astype("<f4")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="docs/assets/model.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, ck = load_model(os.path.join(args.run, "model.pt"))
    model.to(args.device)
    maze = Maze.from_dict(ck["maze"])
    nL = model.cfg.n_layer
    sd = {k: v for k, v in model.state_dict().items()}

    # belief probes at every residual stream point (embed, after each block)
    train = make_dataset(maze, 4000, ck["args"]["T"], 0.35, seed=101)
    keys = [f"resid_{l}" for l in range(nL + 1)]
    acts = collect(model, train["tokens"], keys)
    P = torch.tensor(train["belief"]).reshape(-1, maze.N)
    probes = {}
    loc = P.max(-1).values > 0.99
    cell = torch.tensor(train["cell"]).reshape(-1)
    for k in keys:
        X = acts[k].reshape(-1, acts[k].shape[-1])
        pr = fit_belief_probe(X, P, device=args.device).cpu()
        # class-conditional means (localised states) set the natural scale of a steering edit
        means = torch.stack([X[loc & (cell == c)].mean(0) for c in range(maze.N)])
        probes[k] = {"W": b64(pr.lin.weight), "b": b64(pr.lin.bias), "means": b64(means)}
        print("probe", k, flush=True)

    # egocentric-map probes (visited cells + walked passages in a 7x7 window around the walker)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from egocentric import R as EGO_R, ego_targets
    ego_data = make_dataset(maze, 3000, ck["args"]["T"], 0.0, seed=111)
    Vt, Et = ego_targets(ego_data)
    Xe = collect(model, ego_data["tokens"], ["resid_3"])["resid_3"]
    Xe = Xe.reshape(-1, Xe.shape[-1])
    ego = {"R": EGO_R, "layer": 3}
    for name, Y in (("visited", Vt), ("passages", Et)):
        pr = fit_multilabel_probe(Xe, torch.tensor(Y).reshape(-1, Y.shape[-1]), device=args.device).cpu()
        ego[name] = {"W": b64(pr.lin.weight), "b": b64(pr.lin.bias)}
    print("ego probes", flush=True)

    # per-position mean output of every head, for mean-ablation in the demo
    head_means = {}
    with torch.no_grad():
        sums = None
        toks = torch.tensor(train["tokens"], device=args.device)
        for i in range(0, len(toks), 500):
            c = {}
            model(toks[i:i + 500], cache=c)
            v = [c[f"L{l}.head_out"].sum(0) for l in range(nL)]
            sums = v if sums is None else [a + b for a, b in zip(sums, v)]
        for l in range(nL):
            head_means[l] = b64(sums[l] / len(toks))          # (H, S, dh)

    # reference check for the JS implementation
    ref_tokens = train["tokens"][0, :20].tolist()
    cache = {}
    with torch.no_grad():
        logits = model(torch.tensor([ref_tokens], device=args.device), cache=cache)
    ref = {"tokens": ref_tokens,
           "logits": logits[0].cpu().tolist(),
           "probe_last": {k: F.softmax(torch.tensor(np.frombuffer(base64.b64decode(probes[k]["W"]["b64"]), dtype="<f4")
                                                    .reshape(maze.N, -1).copy()) @ cache[k][0, -1].cpu()
                                       + torch.tensor(np.frombuffer(base64.b64decode(probes[k]["b"]["b64"]), dtype="<f4").copy()),
                                       -1).tolist() for k in keys}}

    # reference for head knock-out: block 1 head 1 replaced by its mean
    mu0 = sums[0] / len(toks)
    def ko(y):
        y = y.clone()
        y[:, 1] = mu0[1, : y.shape[2]]
        return y
    model.blocks[0].attn.head_hook = ko
    with torch.no_grad():
        ref["ablate_0_1_logits"] = model(torch.tensor([ref_tokens], device=args.device))[0].cpu().tolist()
    model.blocks[0].attn.head_hook = None

    out = {"config": ck["config"], "maze": maze.to_dict(),
           "weights": {k: b64(v) for k, v in sd.items()},
           "probes": probes, "ego": ego, "head_means": head_means, "reference": ref}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out, os.path.getsize(args.out) / 1e6, "MB")


if __name__ == "__main__":
    main()
