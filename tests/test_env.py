import numpy as np
import torch

from cartographer.analysis import ground_truth
from cartographer.env import START0, UNK, BayesFilter, make_maze, sample_walks
from cartographer.model import GPT, GPTConfig


def test_maze_symmetric_and_connected():
    m = make_maze(8, 8, 0.35, seed=3)
    nxt = m.nxt()
    for c in range(m.N):
        for d in range(4):
            if m.open[c, d]:
                assert m.open[nxt[c, d], (d + 2) % 4], "walls must be two-sided"
    # BFS reaches every cell
    seen, todo = {0}, [0]
    while todo:
        c = todo.pop()
        for d in range(4):
            if m.open[c, d] and nxt[c, d] not in seen:
                seen.add(nxt[c, d]); todo.append(nxt[c, d])
    assert len(seen) == m.N


def test_walks_are_legal():
    m = make_maze(seed=1)
    tok, cells, known = sample_walks(m, 64, 40, 0.5, np.random.default_rng(0))
    nxt = m.nxt()
    for b in range(64):
        assert tok[b, 0] == (START0 + cells[b, 0] if known[b] else UNK)
        for t in range(40):
            mv = tok[b, t + 1]
            assert m.open[cells[b, t], mv]
            assert nxt[cells[b, t], mv] == cells[b, t + 1]


def test_filter_matches_brute_force():
    """Posterior from the filter == brute-force enumeration over start cells."""
    m = make_maze(5, 5, 0.4, seed=2)
    tok, cells, known = sample_walks(m, 20, 12, 0.0, np.random.default_rng(1))
    beliefs, pred = BayesFilter(m).run(torch.tensor(tok))
    nxt, deg = m.nxt(), m.degree()
    for b in range(20):
        post = np.zeros(m.N)
        for s in range(m.N):
            p, c = 1.0 / m.N, s
            for mv in tok[b, 1:]:
                if not m.open[c, mv]:
                    p = 0.0
                    break
                p /= deg[c]
                c = nxt[c, mv]
            post[c] += p
        np.testing.assert_allclose(beliefs[b, -1].numpy(), post / post.sum(), atol=1e-10)
    np.testing.assert_allclose(pred.sum(-1).numpy(), 1.0, atol=1e-10)


def test_true_cell_always_in_support():
    m = make_maze(seed=4)
    tok, cells, known = sample_walks(m, 128, 64, 0.5, np.random.default_rng(2))
    beliefs, _ = BayesFilter(m).run(torch.tensor(tok))
    assert (beliefs.gather(2, torch.tensor(cells)[..., None]) > 0).all()


def test_belief_support_is_shifted_consistent_starts():
    m = make_maze(seed=0)
    tok, cells, known = sample_walks(m, 64, 64, 0.3, np.random.default_rng(3))
    gt = ground_truth(m, tok, cells, known)
    for i in range(64):
        for t in (0, 3, 10, 30, 64):
            s = np.where(gt["consistent"][i, t])[0]
            r, c = np.divmod(s, m.W)
            shifted = set(((r + gt["disp"][i, t, 0]) * m.W + c + gt["disp"][i, t, 1]).tolist())
            assert shifted == set(np.where(gt["belief"][i, t] > 1e-12)[0].tolist())


def test_model_cache_and_hook_shapes():
    cfg = GPTConfig(vocab_size=69, block_size=17, n_layer=2, d_model=32, d_mlp=64)
    model = GPT(cfg).eval()
    idx = torch.randint(0, 69, (3, 17))
    cache = {}
    out = model(idx, cache=cache)
    assert out.shape == (3, 17, 4)
    assert cache["L0.attn"].shape == (3, 4, 17, 17)
    assert cache["resid_2"].shape == (3, 17, 32)
    # cached path (explicit attention) must agree with the fused SDPA path
    torch.testing.assert_close(out, model(idx), atol=1e-5, rtol=1e-5)
    # identity hook changes nothing; zeroing the final stream gives constant logits
    torch.testing.assert_close(model(idx, resid_hook=lambda l, x: x), out)
    z = model(idx, resid_hook=lambda l, x: x * 0 if l == 2 else x)
    assert torch.allclose(z, z[:1, :1].expand_as(z), atol=1e-6)


def test_slippery_filter_matches_brute_force():
    """With slips, the posterior must equal exact enumeration over all slip patterns."""
    import itertools
    m = make_maze(4, 4, 0.4, seed=5)
    slip = 0.2
    rng = np.random.default_rng(4)
    tok, cells, known = sample_walks(m, 6, 6, 1.0, rng, slip=slip)
    beliefs, _ = BayesFilter(m, slip=slip).run(torch.tensor(tok))
    nxt, deg = m.nxt(), m.degree()
    for b in range(6):
        post = np.zeros(m.N)
        for pattern in itertools.product([0, 1], repeat=6):     # 1 = slipped
            p, c = 1.0, cells[b, 0]
            for mv, sl in zip(tok[b, 1:], pattern):
                if not m.open[c, mv]:
                    p = 0.0
                    break
                p *= (1 / deg[c]) * (slip if sl else 1 - slip)
                c = c if sl else nxt[c, mv]
            post[c] += p
        np.testing.assert_allclose(beliefs[b, -1].numpy(), post / post.sum(), atol=1e-10)
