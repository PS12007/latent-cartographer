# Latent Cartographer

**A transformer that has only ever read `N E S W` builds a map of a maze it has never seen, tracks a Bayesian belief over where it is, and lets you redraw the whole maze from its activations.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PS12007/latent-cartographer/blob/main/notebooks/walkthrough.ipynb) [![tests](https://github.com/PS12007/latent-cartographer/actions/workflows/tests.yml/badge.svg)](https://github.com/PS12007/latent-cartographer/actions/workflows/tests.yml)

**[▶ Live demo](https://ps12007.github.io/latent-cartographer/)**: the model runs in your browser (plain JavaScript, no server). Walk the maze, watch its internal belief collapse onto your position, click a cell to *steer* what it believes, or press **Extract the map** to rebuild the maze from the network alone.

![Belief over a walk: exact Bayes vs decoded from the model](figures/beliefs.png)

**How it localises**, as established below with probes and causal edits:

![Pipeline: dead reckoning, egocentric sketch, match against the maze in the weights](figures/pipeline.svg)

---

## The setup

* **World.** A fixed 8×8 maze with loops (randomised DFS, then 35% of the remaining walls removed). 
* **Data.** Uniform random walks of 64 moves. Each sequence starts with either `@cell` (start given, 35% of the time) or `?` (start unknown), followed by move tokens only.
* **Model.** A 0.8M-parameter GPT (4 blocks, d=128, 4 heads), trained with ordinary next-token cross-entropy for 20k steps (≈330M tokens, 7.5 min on a laptop RTX 5060).
* **Ground truth.** Because the walker picks uniformly among open directions, the best possible predictor is an **exact Bayesian filter** over the 64 cells ([`cartographer/env.py`](cartographer/env.py), checked against brute-force enumeration in the tests). This gives the exact optimal loss, *and* the exact belief state the model ought to be tracking.

The model is never told the maze, the coordinates, or that walls exist. To predict well it has to infer which moves are legal, which means knowing where it is, which needs a map nobody gave it. When the start is unknown it also has to **localise**: rule out every place it could not be, given the moves it has seen.

## Results at a glance

| | |
|---|---|
| Next-move loss vs Bayes-optimal | **0.9929 vs 0.9888 nats**. KL to optimal: 0.0001 (start given), 0.006 (start unknown) |
| Linear readout of the current cell (once localised) | **98.7%** (random-init network: 6%) |
| Linear readout of the full 64-way belief | KL **0.10** nats to the exact posterior, vs 1.64 from the model's own output distribution |
| Steering the decoded belief to another cell | model's predicted legal moves become the new cell's **90%** of the time (random edit of same size: 3.5%) |
| Maze recovered from the model by steering alone | **112 / 112 interior walls correct** |
| Writing one remembered passage into the local map | belief shifts toward whichever candidate cell has that passage (+0.22 / −0.18 vs control) |
| Does a steered belief persist to the next step? | **No** (14% vs 14% unsteered): location is recomputed from the history at every token |

## 1. The model is (nearly) Bayes-optimal, and the belief is in the residual stream

Linear probes fitted at every point of the residual stream ([`experiments/probes.py`](experiments/probes.py)) show the computation unfolding with depth:

![Probes across depth](figures/probes.png)

* **Dead reckoning comes first.** Displacement since the start is linearly decodable after block 1 (R² 0.97), then fades as later layers overwrite it.
* **The belief sharpens block by block.** The KL between the decoded belief and the exact posterior falls from 3.4 nats (embedding) to **0.10** (block 4 attention). It stays at 0.31 even on steps where the true posterior is still spread over several cells, so the probe recovers multi-modal beliefs, not just the most likely cell.
* **The activations hold more than the output does.** A two-layer MLP decoding the belief from the model's own next-move distribution (plus position) only reaches KL 1.64. The residual stream carries the full 64-way belief, far more than the 4 numbers it needs to emit. This matches the belief-state geometry result of [Shai et al. 2024](https://arxiv.org/abs/2405.15943), here in a spatial setting where the belief can be drawn.
* When the probe is wrong (4% of localised steps), 64% of its mistakes land **within one cell** of the truth (diagonals included). The representation is spatially smooth.

## 2. The map is causal, and you can read the whole maze out of it

Decodability is not use. To test use, [`experiments/intervene.py`](experiments/intervene.py) takes walks where the model is certain it is in cell *c*, edits the newest token's residual vector so it should mean "you are in *c′*", and checks whether the model's predicted legal moves (moves given >10%) become exactly the open sides of *c′*.

![Steering](figures/steering.png)

* **Probe-direction steering works.** `x + α·‖μ_c′ − μ_c‖·unit(W_c′ − W_c)`, after block 3 with α=2: **90%**. Random directions of the same norm: 3.5%. The dose-response rises and then falls as the edit overshoots.
* **The location only settles at the newest token late in the network.** Pasting in the *entire* residual vector from another walk works 10% of the time at the embedding, 34% after block 2, 74% after block 3, and 100% after block 4. Before block 3, later layers simply re-read the location from the history and overwrite the edit.
* **A negative result worth keeping.** The minimum-norm edit `W⁺(z′ − z)` that exactly swaps two of the probe's outputs *fails* (≤6%). It also pins the other 62 outputs, which drags the edit off the directions the network uses. "The probe reads X" and "this edit makes the probe read X" are different claims.

Steering to each of the 64 cells in turn and recording which moves the model then allows recovers the maze **perfectly**: every one of the 112 interior walls. The environment is never consulted.

<p align="center"><img src="figures/map.png" width="640" alt="Maze recovered from the model"></p>

**No persistence.** If the belief at step *t* is steered to *c′* and the walk then continues with a move that is legal from *c′*, the prediction at *t+1* ignores the edit completely (14% vs 14%). The network does not carry its location forward like an RNN's hidden state; it **recomputes it from the raw move history at every token**. This parallel, stateless style of computation is what one expects from a fixed-depth transformer (cf. Merrill et al., [*The Illusion of State*](https://arxiv.org/abs/2404.08819)).

## 3. How it localises: an egocentric map, matched against an allocentric one

The parallel algorithm I first expected was:
(1) each token computes its displacement;
(2) each token marks which *start* cells its own move rules out;
(3) later layers average those exclusion masks.

The data reject steps (2)–(3) ([`experiments/mechanism.py`](experiments/mechanism.py)). The single-step exclusion mask is barely decodable (balanced acc. 0.58 vs 0.51 at random init). Even an MLP probe recovers the surviving-start set only weakly (0.87 vs 0.77). The model does not reason in terms of where it began.

What it does represent is a **local map of the history, in its own coordinate frame** ([`experiments/egocentric.py`](experiments/egocentric.py)). For every offset in a 7×7 window around the current position: have I been there, and which passages next to it have I walked through?

![Two reference frames](figures/frames.png)

The egocentric map peaks in **block 3** (visited cells 0.89, walked passages 0.84, vs ≈0.6 at random init) and then declines. At the same time the position in the maze's own frame climbs to 98.7% in block 4. The attention pattern agrees: block 1's heads spread attention almost evenly over the whole history (normalised entropy 0.93–1.0, the averaging needed for dead reckoning), while later heads are sharp and content-dependent.

**The local map is used, not just present** ([`experiments/egocentric_causal.py`](experiments/egocentric_causal.py)). Take moments where the exact posterior has exactly two candidate cells, A (true) and B. Both fit everything seen so far; they differ only in unexplored parts of the maze. Pick a passage, relative to the walker, that has *not* been walked, and write "I walked this passage" into the residual stream after block 3 along the local-map probe's direction:

![Egocentric map, causal test](figures/egocausal.png)

A passage that exists only around B pushes the belief toward B (+0.29 at α=8). One that exists only around A pushes it toward A (−0.11). A passage that exists around both does little (+0.07). Relative to that control the effects are +0.22 and −0.18, in opposite directions, as the matching story predicts. (The control drifts slightly toward B because the model, like the exact posterior, starts out favouring the true cell A on average, and any disruption pulls it back toward 50/50.)

**Heads divide the work** ([`experiments/head_ablation.py`](experiments/head_ablation.py)). Mean-ablating one head at a time and measuring the damage separately for known- and unknown-start walks:

<p align="center"><img src="figures/heads.png" width="560" alt="Head ablations"></p>

* **B1H1 is the integrator.** Removing it raises known-start KL from 0.0001 to 0.92. It is also the head whose attention is almost perfectly uniform over the history (normalised entropy 0.999), exactly the averaging dead reckoning needs.
* **B1H3 reads the start token.** It gives position 0 the most attention of any head, and removing it hurts known-start walks most.
* **Eight heads are localisation specialists.** All of block 3, plus B2H1 and B2H3, cost 5–75× more when the start is unknown. This matches where the egocentric sketch peaks.

The emerging picture is a tiny, learned form of the SLAM that robots use: integrate your moves → build a local map of what you have walked, *relative to yourself* → match it against a map stored *in the weights* → read off where you are.

## 4. Depth and training dynamics

![Depth sweep](figures/depth.png)

With width and training budget fixed, one block already does dead reckoning from a known start well (KL 0.003), but fails badly at localising from an unknown start (0.21, about 80× worse). Each extra block cuts localisation error by roughly 2–5× (0.21 → 0.044 → 0.015 → 0.006 → 0.0026 for 1/2/3/4/6 blocks), which fits a computation with several sequential stages: integrate, build the local map, match. (Deeper models also have more parameters, 0.2M → 1.2M, so this is depth *and* size.)

![Training dynamics](figures/dynamics.png)

Across training, displacement becomes decodable almost at once. The maze-frame position emerges over thousands of steps afterwards, and the loss on unknown-start walks keeps improving long after known-start walks are nearly solved.

## Reproduce

The quickest route is the [walkthrough notebook](notebooks/walkthrough.ipynb): about 15 s on a CPU, it reproduces the Bayes match, the belief probe, steering and map extraction.


```bash
pip install -r requirements.txt           # torch, numpy, matplotlib, pytest
python -m cartographer.train --out runs/main        # ~8 min on a laptop GPU (runs/main/model.pt is included)
python experiments/probes.py               # Fig. probes      -> results/probes.json
python experiments/intervene.py            # steering + map   -> results/intervene.json
python experiments/mechanism.py            # exclusion masks, attention stats
python experiments/egocentric.py           # egocentric map probes
python experiments/dynamics.py             # probes on training snapshots
bash   experiments/sweep_depth.sh          # depth sweep
python experiments/figures.py              # all figures
python experiments/export_web.py           # weights + probes for the demo
pytest -q tests && node tests/js_parity.mjs  # filter == brute force; JS model == PyTorch (max |Δlogit| ≈ 1e-5)
```

```
cartographer/   env.py (maze, walks, exact Bayes filter)  model.py (GPT with cache + residual hooks)
                train.py  analysis.py (ground-truth latents, probes)
experiments/    one script per experiment, each writes results/*.json
docs/           the browser demo (GitHub Pages): model.js is a line-for-line JS port of model.py
```

## Limitations and next steps

* **One maze.** All results are for a single 8×8 maze (seed 0). Map extraction should be repeated across maze seeds and sizes, and the depth sweep across seeds.
* **The circuit is not fully reverse-engineered.** The stages (dead reckoning → egocentric map → matching) are established with probes and edits on the residual stream, not traced to individual heads and neurons. The matching step, where a local map meets the stored maze, is the obvious next target.
* **Uniform random walks are friendly data.** [Vafa et al. 2024](https://arxiv.org/abs/2406.03689) show that transformers trained on real taxi routes predict well *without* forming a coherent map. The interesting question is when the clean structure seen here breaks down: biased policies, partial observability, larger worlds.
* **Steering uses class means computed with ground-truth labels** to set the edit size. The direction comes from the probe; the scale does not.

## Related work

Li et al., [*Emergent World Representations* (Othello-GPT)](https://arxiv.org/abs/2210.13382) and Nanda et al., [*Emergent Linear Representations in World Models*](https://arxiv.org/abs/2309.00941) (board-state probing and intervention) · Shai et al., [*Transformers represent belief state geometry in their residual stream*](https://arxiv.org/abs/2405.15943) · Banino et al. 2018 and Cueva & Wei 2018 (grid-like codes from path integration in RNNs) · Whittington et al., *Relating transformers to models and neural representations of the hippocampal formation* (2022) · Vafa et al., [*Evaluating the World Model Implicit in a Generative Model*](https://arxiv.org/abs/2406.03689).

---
MIT licensed.
