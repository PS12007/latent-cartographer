#!/usr/bin/env bash
# Robustness — repeat the main pipeline on different mazes (and model init seeds).
PY=${PY:-python}
for S in 1 2 3; do
  $PY -m cartographer.train --out runs/maze$S --maze_seed $S --seed $S --steps 20000 --eval_every 2000 --snapshots 2
  $PY experiments/intervene.py --run runs/maze$S --out results/intervene_maze$S.json
  $PY experiments/probes.py --run runs/maze$S --out results/probes_maze$S.json
done
