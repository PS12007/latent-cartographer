#!/usr/bin/env bash
# Experiment 4 — how much depth does belief tracking need?
# Same data, width and budget; only n_layer changes. (4 layers = runs/main)
PY=${PY:-python}
for L in 1 2 3 6; do
  $PY -m cartographer.train --out runs/depth$L --n_layer $L --steps 20000 --eval_every 2000 --snapshots 2
done
