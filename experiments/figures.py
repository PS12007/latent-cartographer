"""Render every figure in the README from results/*.json (+ the trained model for Fig. 1).

    python experiments/figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cartographer.analysis import collect, fit_belief_probe, make_dataset  # noqa: E402
from cartographer.env import UNK, BayesFilter, Maze, sample_walks  # noqa: E402
from cartographer.model import load_model  # noqa: E402

# palette validated with the dataviz validator (light surface #f7f2e6)
MODEL, REF, STEER = "#c2381f", "#2f62a8", "#b8860b"
CTRL, INK, SOFT, SURF = "#9a9184", "#1f1b16", "#5d5446", "#f7f2e6"
FIG = "figures"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "font.size": 10.5,
    "axes.edgecolor": SOFT, "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e3dacb", "grid.linewidth": 0.8, "axes.titleweight": "normal",
    "axes.titlesize": 12, "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})


def load(p):
    with open(p) as f:
        return json.load(f)


def draw_maze(ax, maze, values=None, cmap_color=MODEL, open_=None, lw=1.6):
    H, W = maze.H, maze.W
    op = maze.open if open_ is None else open_
    if values is not None:
        v = np.asarray(values, dtype=float).reshape(H, W)
        peak = v.max()
        v = v / max(peak, 1e-9) * min(1.0, peak * 4)  # a diffuse belief stays faint
        rgba = np.zeros((H, W, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(cmap_color)
        rgba[..., 3] = np.where(v > 1e-4, 0.08 + 0.9 * v ** 0.55, 0)
        ax.imshow(rgba, extent=(0, W, H, 0), interpolation="nearest")
    segs = []
    for c in range(maze.N):
        r, q = divmod(c, W)
        if not op[c][0]: segs.append(((q, q + 1), (r, r)))
        if not op[c][3]: segs.append(((q, q), (r, r + 1)))
        if r == H - 1: segs.append(((q, q + 1), (r + 1, r + 1)))
        if q == W - 1: segs.append(((q + 1, q + 1), (r, r + 1)))
    for xs, ys in segs:
        ax.plot(xs, ys, color=INK, lw=lw, solid_capstyle="round")
    ax.set_xlim(-0.1, W + 0.1); ax.set_ylim(H + 0.1, -0.1)
    ax.set_aspect("equal"); ax.axis("off")


# ------------------------------------------------------------------ Fig 1: beliefs over a walk
def fig_beliefs():
    model, ck = load_model("runs/main/model.pt")
    maze = Maze.from_dict(ck["maze"])
    train = make_dataset(maze, 3000, 64, 0.35, seed=101)
    A = collect(model, train["tokens"], ["resid_3"])["resid_3"]
    probe = fit_belief_probe(A.reshape(-1, A.shape[-1]), torch.tensor(train["belief"]).reshape(-1, maze.N))
    rng = np.random.default_rng(11)
    # pick a walk whose posterior stays multi-modal for a while (more interesting)
    best = None
    for _ in range(200):
        tok, cells, _ = sample_walks(maze, 1, 40, 0.0, rng)
        b, _ = BayesFilter(maze).run(torch.tensor(tok))
        n_modes = (b[0, 8] > 1e-6).sum().item()
        if best is None or abs(n_modes - 4) < abs(best[2] - 4):
            best = (tok, cells, n_modes, b)
    tok, cells, _, b = best
    cache = {}
    with torch.no_grad():
        model(torch.tensor(tok), cache=cache)
        q = F.softmax(probe(cache["resid_3"][0]), -1).numpy()
    ts = [0, 2, 4, 8, 14, 24]
    fig, axes = plt.subplots(2, len(ts), figsize=(12.6, 4.9))
    for j, t in enumerate(ts):
        draw_maze(axes[0, j], maze, b[0, t].numpy(), REF, lw=1.2)
        draw_maze(axes[1, j], maze, q[t], MODEL, lw=1.2)
        r, c = divmod(int(cells[0, t]), maze.W)
        for ax in axes[:, j]:
            ax.plot(c + 0.5, r + 0.5, marker="o", ms=4.5, mfc="none", mec=INK, mew=1.2)
        axes[0, j].set_title(f"after {t} moves", fontsize=10.5, loc="center", color=SOFT)
    axes[0, 0].text(-0.6, 4, "exact Bayes", rotation=90, va="center", ha="center", color=REF, fontsize=11.5)
    axes[1, 0].text(-0.6, 4, "decoded from model", rotation=90, va="center", ha="center", color=MODEL, fontsize=11.5)
    fig.suptitle("Start unknown. The model’s internal belief (bottom) tracks the exact posterior (top), multiple modes included.",
                 x=0.02, ha="left", fontsize=12.5, color=INK)
    fig.text(0.02, 0.01, "Linear probe on the residual stream after block 3. Ring = true position (never shown to the model). Walk chosen only for being "
             "multi-modal at move 8. Here the probe slips one cell south at the end; it is exact on 96% of localised steps, and 64% of misses are adjacent cells.",
             color=SOFT, fontsize=9.5)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.87, bottom=0.09, wspace=0.06, hspace=0.06)
    fig.savefig(f"{FIG}/beliefs.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 2: probes across depth
def fig_probes():
    R = load("results/probes.json")
    tr, rd, base = R["trained"], R["random"], R["baselines"]
    keys = list(tr.keys())
    labels = [tr[k]["label"].replace(" attn", "\nattn").replace(" mlp", "\nmlp") for k in keys]
    x = np.arange(len(keys))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.1))
    specs = [
        ("disp_r2", "Displacement since start (R²)", (0, 1.02), False),
        ("belief_kl", "Belief over cells: KL to exact posterior (nats)", None, True),
        ("cell_acc_localized", "Current cell, once localised (accuracy)", (0, 1.02), False),
    ]
    for ax, (k, title, ylim, logy) in zip(axs, specs):
        ax.plot(x, [tr[p][k] for p in keys], color=MODEL, marker="o", ms=6, label="trained")
        ax.plot(x, [rd[p][k] for p in keys], color=CTRL, ls="--", marker="o", ms=5, label="random init")
        ax.set_title(title)
        ax.set_xticks(x, labels, fontsize=8.5)
        if ylim: ax.set_ylim(*ylim)
        if logy: ax.set_yscale("log")
    axs[1].axhline(base["time_only_kl"], color=CTRL, lw=1, ls=":")
    axs[1].text(len(keys) - 1, base["time_only_kl"] * 0.86, "best time-only guess", ha="right", color=SOFT, fontsize=9)
    axs[1].axhline(base["output_mlp_kl"], color=REF, lw=1.2, ls=":")
    axs[1].text(len(keys) - 1, base["output_mlp_kl"] * 1.1, "decoded from model’s own output", ha="right", color=REF, fontsize=9)
    i = int(np.argmin([tr[p]["belief_kl"] for p in keys]))
    axs[1].annotate(f"{tr[keys[i]]['belief_kl']:.2f}", (i, tr[keys[i]]["belief_kl"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=MODEL, fontsize=9.5)
    axs[0].legend(loc="lower right")
    fig.suptitle("Displacement appears in block 1; the belief over cells sharpens with every block after that.",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/probes.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 3: steering
def fig_steering():
    R = load("results/intervene.json")
    Ls = sorted(R["layers"], key=int)
    x = np.arange(len(Ls))
    lab = ["embed" if l == "0" else f"after\nblock {l}" for l in Ls]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    series = [
        ("full patch (upper bound)", [R["layers"][l]["full_patch"] for l in Ls], REF, "-"),
        ("probe direction, α=2", [R["layers"][l]["probe_dir"]["2"] for l in Ls], MODEL, "-"),
        ("class-mean difference", [R["layers"][l]["mean_diff"] for l in Ls], STEER, "-"),
        ("random direction, same norm", [R["layers"][l]["random"]["2"] for l in Ls], CTRL, "--"),
    ]
    for name, ys, col, ls in series:
        a.plot(x, ys, color=col, ls=ls, marker="o", ms=6)
        a.text(x[-1] + 0.12, ys[-1], name, color=col if col != STEER else SOFT, va="center", fontsize=9)
    a.set_xticks(x, lab, fontsize=9)
    a.set_xlim(-0.3, len(Ls) + 1.6)
    a.set_ylim(-0.02, 1.04)
    a.set_title("Steering success by depth (edit the newest token only)")
    a.set_ylabel("predicted legal moves = target cell’s")
    l3 = R["layers"]["3"]
    al = [float(k) for k in l3["probe_dir"]]
    b.plot(al, list(l3["probe_dir"].values()), color=MODEL, marker="o", ms=6, label="probe direction")
    b.plot(al, list(l3["random"].values()), color=CTRL, ls="--", marker="o", ms=5, label="random direction")
    b.set_xscale("log", base=2)
    b.set_xticks(al, [str(a_) for a_ in al])
    b.set_ylim(-0.02, 1.04)
    b.set_xlabel("edit size α (× distance between the two cells’ mean activations)")
    b.set_title("Dose-response after block 3")
    b.legend(loc="center right")
    fig.suptitle("The decoded map is causal, but only once it has been assembled at the current token (after block 3).",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/steering.png", dpi=160)
    plt.close(fig)

    # map extraction
    maze = Maze.from_dict(R["maze"])
    rec = np.array(R["map"]["recon"])
    op = np.zeros_like(maze.open)
    nxt = maze.nxt()
    for c in range(maze.N):
        r, q = divmod(c, maze.W)
        for d, (dr, dq) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            rr, qq = r + dr, q + dq
            if 0 <= rr < maze.H and 0 <= qq < maze.W:
                nb = rr * maze.W + qq
                op[c, d] = 0.5 * (rec[c, d] + rec[nb, (d + 2) % 4]) > 0.1
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.6))
    draw_maze(axs[0], maze)
    draw_maze(axs[1], maze, open_=op)
    axs[0].set_title("ground truth", loc="center", color=SOFT)
    axs[1].set_title(f"read out of the model: {R['map']['interior_correct']}/{R['map']['interior_total']} walls correct",
                     loc="center", color=MODEL)
    fig.tight_layout()
    fig.savefig(f"{FIG}/map.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 4: depth sweep
def fig_depth():
    rows = []
    for p in sorted(glob.glob("runs/depth*/train_log.json")) + ["runs/main/train_log.json"]:
        L = 4 if "main" in p else int(p.split("depth")[1].split(os.sep)[0].split("/")[0])
        last = load(p)[-1]
        rows.append((L, last["kl_known"], last["kl_unknown"]))
    rows.sort()
    Ls = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4))
    ax.plot(Ls, [r[2] for r in rows], color=MODEL, marker="o", ms=7)
    ax.plot(Ls, [r[1] for r in rows], color=REF, marker="o", ms=7)
    ax.text(Ls[-1] + 0.15, rows[-1][2], "start unknown\n(must localise)", color=MODEL, va="center", fontsize=9.5)
    ax.text(Ls[-1] + 0.15, rows[-1][1], "start given\n(path integration)", color=REF, va="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.set_xlim(0.6, Ls[-1] + 1.6)
    ax.set_xlabel("number of transformer blocks (width fixed at 128)")
    ax.set_ylabel("KL to Bayes-optimal (nats / move)")
    ax.set_title("Localisation needs depth; one block nearly solves dead reckoning")
    fig.tight_layout()
    fig.savefig(f"{FIG}/depth.png", dpi=160)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------ Fig 5: dynamics
def fig_dynamics():
    if not os.path.exists("results/dynamics.json"):
        return
    D = load("results/dynamics.json")
    steps = [d["step"] for d in D]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
    a.plot(steps, [d["kl_unknown"] for d in D], color=MODEL, marker="o", ms=5)
    a.plot(steps, [d["kl_known"] for d in D], color=REF, marker="o", ms=5)
    a.text(steps[-1] * 1.1, D[-1]["kl_unknown"], "start unknown", color=MODEL, va="center", fontsize=9.5)
    a.text(steps[-1] * 1.1, D[-1]["kl_known"], "start given", color=REF, va="center", fontsize=9.5)
    a.set_yscale("log")
    a.set_ylabel("KL to optimal (nats)")
    a.set_title("Behaviour")
    best = lambda d, k, f=max: f(v[k] for v in d["probes"].values())  # noqa: E731
    b.plot(steps, [best(d, "disp_r2") for d in D], color=REF, marker="o", ms=5)
    b.plot(steps, [best(d, "cell_acc_localized") for d in D], color=MODEL, marker="o", ms=5)
    b.text(steps[-1] * 1.1, best(D[-1], "disp_r2") + 0.03, "displacement R²", color=REF, va="center", fontsize=9.5)
    b.text(steps[-1] * 1.1, best(D[-1], "cell_acc_localized") - 0.1, "cell accuracy", color=MODEL, va="center", fontsize=9.5)
    b.set_ylim(0, 1.04)
    b.set_ylabel("best layer’s probe")
    b.set_title("What the probes find")
    b.set_xscale("log")
    b.set_xlim(steps[0] * 0.8, steps[-1] * 4)
    b.set_xlabel("training step")
    fig.suptitle("Dead reckoning is learned first; the map follows.", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{FIG}/dynamics.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 6: two reference frames
def fig_frames():
    E = load("results/egocentric.json")
    P = load("results/probes.json")
    keys = list(E["trained"].keys())
    x = np.arange(len(keys))

    def nice(lbl):
        if lbl == "embed":
            return "embed"
        l, part = lbl.split()
        return f"block {int(l[1:]) + 1}\n{part}"
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    series = [
        ("visited cells, relative to me", [E["trained"][k]["visited_bacc"] for k in keys], [E["random"][k]["visited_bacc"] for k in keys], REF),
        ("walked passages, relative to me", [E["trained"][k]["edges_bacc"] for k in keys], [E["random"][k]["edges_bacc"] for k in keys], STEER),
        ("which cell I am in (maze frame)", [P["trained"][k]["cell_acc_localized"] for k in keys], [P["random"][k]["cell_acc_localized"] for k in keys], MODEL),
    ]
    for name, ys, rs, col in series:
        ax.plot(x, ys, color=col, marker="o", ms=6)
        ax.plot(x, rs, color=col, ls=":", lw=1.2, alpha=0.7)
        i = int(np.argmax(ys))
        ax.text(x[-1] + 0.15, ys[-1], name, color=col if col != STEER else SOFT, va="center", fontsize=9.5)
        ax.plot([x[i]], [ys[i]], marker="o", ms=11, mfc="none", mec=col, mew=1.4)
    ax.set_xticks(x, [nice(E["trained"][k]["label"]) for k in keys], fontsize=8.5)
    ax.set_xlim(-0.3, len(keys) + 2.6)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("linear probe score (bal. acc. / acc.)")
    ax.set_title("Egocentric map peaks mid-network; the maze-frame position peaks at the end (dotted = random init)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/frames.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 7: egocentric map, causal
def fig_egocausal():
    R = load("results/egocentric_causal.json")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    names = {"pro_B": ("passage exists only around B", MODEL), "pro_A": ("passage exists only around A", REF),
             "neutral": ("passage exists around both", CTRL)}
    for cond, (name, col) in names.items():
        d = R["conditions"][cond]
        al = [float(a) for a in d]
        m = np.array([v["mean"] for v in d.values()])
        e = np.array([v["sem"] for v in d.values()])
        ax.plot(al, m, color=col, marker="o", ms=6, ls="--" if cond == "neutral" else "-")
        ax.fill_between(al, m - 1.96 * e, m + 1.96 * e, color=col, alpha=0.15, lw=0)
        ax.text(al[-1] + 0.4, m[-1], name, color=col if cond != "neutral" else SOFT, va="center", fontsize=9.5)
    ax.axhline(0, color=SOFT, lw=0.8)
    ax.set_xlim(-0.5, max(al) + 9)
    ax.set_xticks([a_ for a_ in al])
    ax.set_xlabel("edit size α (write “I walked this passage” into the residual after block 3)")
    ax.set_ylabel("change in P(B) / (P(A)+P(B))")
    ax.set_title(f"Two candidate cells, A and B. Editing one remembered passage moves the belief (n={R['n']})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/egocausal.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 8: head ablations
def fig_heads():
    R = load("results/head_ablation.json")
    b = R["baseline"]
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot([1e-4, 2], [1e-4, 2], color=CTRL, lw=1, ls=":")
    ax.text(1.0, 1.25, "equal harm", color=SOFT, fontsize=9, rotation=38, ha="center")
    for name, v in R["heads"].items():
        dk, du = max(v["known"] - b[0], 1e-4), max(v["unknown"] - b[1], 1e-4)
        col = REF if du > 5 * dk else MODEL if dk > 0.05 else CTRL   # specialist / integrator / minor
        ax.plot(dk, du, "o", ms=8, color=col, mec=SURF, mew=1.5)
        l, h = int(name[1]), int(name[3])
        ax.annotate(f"B{l + 1}H{h}", (dk, du), textcoords="offset points", xytext=(6, -3), fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e-4, 2); ax.set_ylim(5e-3, 2)
    ax.set_xlabel("extra KL when the start is given (dead reckoning)")
    ax.set_ylabel("extra KL when the start is unknown (localisation)")
    ax.text(6e-4, 1.2, "localisation-specific\n(>5× more harm when lost)", color=REF, fontsize=9.5, va="top")
    ax.text(0.9, 7e-3, "needed for\ndead reckoning", color=MODEL, fontsize=9.5, ha="right")
    ax.text(2.5e-3, 7e-3, "gray: minor for both", color=SOFT, fontsize=9)
    ax.set_title("Mean-ablating each of the 16 attention heads")
    fig.tight_layout()
    fig.savefig(f"{FIG}/heads.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 9: geometry of the internal map
def fig_geometry():
    G = load("results/geometry.json")
    maze = Maze.from_dict(G["maze"])
    keys = list(G["layers"].keys())
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    nxt = maze.nxt()
    deg = maze.open.sum(1)
    shades = {1: "#f0b9a8", 2: "#e07a5f", 3: "#c2381f", 4: "#6e1c0e"}   # one hue, light -> dark
    for j, (k, title) in enumerate([("resid_1", "after block 1: a flat sheet shaped like the grid"),
                                    ("resid_4", "after block 4: regrouped by which walls a cell has")]):
        ax = fig.add_subplot(gs[0, j])
        P = np.array(G["layers"][k]["pca2"])
        if j == 0:
            for c in range(maze.N):
                for m in (1, 2):
                    if maze.open[c, m]:
                        n = nxt[c, m]
                        ax.plot([P[c, 0], P[n, 0]], [P[c, 1], P[n, 1]], color=INK, lw=1.1, alpha=0.55)
            ax.scatter(P[:, 0], P[:, 1], s=30, c=MODEL, edgecolors=SURF, linewidths=1, zorder=3)
            for cc, lab in ((0, "(0,0)"), (maze.W - 1, f"(0,{maze.W - 1})"), (maze.N - maze.W, f"({maze.H - 1},0)"),
                            (maze.N - 1, f"({maze.H - 1},{maze.W - 1})")):
                ax.annotate(lab, P[cc], textcoords="offset points", xytext=(5, 4), fontsize=8.5, color=SOFT)
        else:
            for d in (1, 2, 3, 4):
                m_ = deg == d
                if m_.any():
                    ax.scatter(P[m_, 0], P[m_, 1], s=42, c=shades[d], edgecolors=SURF, linewidths=1, zorder=3,
                               label=f"{d} open side{'s' if d > 1 else ''}")
            ax.legend(loc="best", fontsize=8.5, handletextpad=0.3)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel(f"PC1–2 ({G['layers'][k]['pca_var2']:.0%} of variance)", fontsize=9, color=SOFT)
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(keys))
    for name, field, col in (("grid distance", "grid", REF), ("walking distance", "path", STEER), ("same walls", "walls", MODEL)):
        ys = [G["layers"][k][field] for k in keys]
        ax.plot(x, ys, color=col, marker="o", ms=6)
        ax.text(x[-1] + 0.12, ys[-1], name, color=col if col != STEER else SOFT, va="center", fontsize=9.5)
    ax.axhline(0, color=SOFT, lw=0.8)
    ax.set_xticks(x, ["embed"] + [f"after\nblock {i}" for i in range(1, len(keys))], fontsize=9)
    ax.set_xlim(-0.3, len(keys) + 1.2)
    ax.set_ylabel("rank corr. with distance between cell means")
    ax.set_title("What the geometry tracks", fontsize=11)
    fig.suptitle("The internal map: each dot is one cell’s average activation (left: lines are open passages).",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{FIG}/geometry.png", dpi=160)
    plt.close(fig)


def fig_map_big():
    R = load("results/intervene_big12.json")
    maze = Maze.from_dict(R["maze"])
    rec = np.array(R["map"]["recon"])
    nxt = maze.nxt()
    op = np.zeros_like(maze.open)
    wrong = []
    for c in range(maze.N):
        r, q = divmod(c, maze.W)
        for d, (dr, dq) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            rr, qq = r + dr, q + dq
            if 0 <= rr < maze.H and 0 <= qq < maze.W:
                nb = rr * maze.W + qq
                op[c, d] = 0.5 * (rec[c, d] + rec[nb, (d + 2) % 4]) > 0.1
                if d in (1, 2) and op[c, d] != maze.open[c, d]:
                    wrong.append((r, q, d))
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 5.1))
    draw_maze(axs[0], maze, lw=1.3)
    draw_maze(axs[1], maze, open_=op, lw=1.3)
    for r, q, d in wrong:  # circle each mistake
        x, y = (q + 1, r + 0.5) if d == 1 else (q + 0.5, r + 1)
        axs[1].add_patch(plt.Circle((x, y), 0.45, fill=False, ec=MODEL, lw=2))
    axs[0].set_title("ground truth (12×12)", loc="center", color=SOFT)
    axs[1].set_title(f"read out of the model: {R['map']['interior_correct']}/{R['map']['interior_total']} walls correct (misses circled)",
                     loc="center", color=MODEL, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/map_big12.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["beliefs", "probes", "steering", "depth", "dynamics", "frames", "egocausal", "heads", "geometry", "map_big"]
    for w in which:
        print(w, globals()[f"fig_{w}"]())
