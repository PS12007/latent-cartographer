"""Print a markdown robustness table across mazes from runs/*/train_log.json and results/*.json.

    python experiments/summarize.py
"""
from __future__ import annotations

import json
import os

RUNS = [("main", "8×8, seed 0", "runs/main", "results/intervene.json", "results/probes.json"),
        ("maze1", "8×8, seed 1", "runs/maze1", "results/intervene_maze1.json", "results/probes_maze1.json"),
        ("maze2", "8×8, seed 2", "runs/maze2", "results/intervene_maze2.json", "results/probes_maze2.json"),
        ("maze3", "8×8, seed 3", "runs/maze3", "results/intervene_maze3.json", "results/probes_maze3.json"),
        ("big12", "12×12, seed 0", "runs/big12", "results/intervene_big12.json", "results/probes_big12.json")]


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    print("| maze | KL to optimal (given / unknown start) | best probe cell acc. | steering after block 3 (random) | walls recovered | persistence (steered / not) |")
    print("|---|---|---|---|---|---|")
    for _, name, run, iv, pr in RUNS:
        if not (os.path.exists(f"{run}/train_log.json") and os.path.exists(iv) and os.path.exists(pr)):
            continue
        log = load(f"{run}/train_log.json")[-1]
        I, P = load(iv), load(pr)
        n_layer = P["n_layer"]
        best_cell = max(v["cell_acc_localized"] for v in P["trained"].values())
        l3 = I["layers"][str(n_layer - 1)]
        m = I["map"]
        pers = I["persistence"][str(n_layer - 1)]
        print(f"| {name} | {log['kl_known']:.4f} / {log['kl_unknown']:.4f} | {best_cell:.1%} | "
              f"{l3['probe_dir']['2']:.0%} ({l3['random']['2']:.0%}) | {m['interior_correct']}/{m['interior_total']} | "
              f"{pers['steered']:.0%} / {pers['unsteered']:.0%} |")


if __name__ == "__main__":
    main()
