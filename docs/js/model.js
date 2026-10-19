// Line-for-line JavaScript port of cartographer/model.py (forward pass only).
// Runs the whole 4-layer transformer in the browser; a 65-token forward pass is ~50 MFLOP.

function decode(t) {
  const bin = atob(t.b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return { shape: t.shape, data: new Float32Array(buf) };
}

// erf with |error| < 1.2e-7 (Numerical Recipes erfc), so GELU matches torch's exact GELU.
function erf(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

// y[out] = W[out, in] x[in] + b[out]
function linear(x, W, b) {
  const [o, n] = W.shape, w = W.data, y = new Float32Array(o);
  for (let i = 0; i < o; i++) {
    let s = b ? b.data[i] : 0;
    const off = i * n;
    for (let j = 0; j < n; j++) s += w[off + j] * x[j];
    y[i] = s;
  }
  return y;
}

function layerNorm(x, g, b) {
  const n = x.length;
  let mu = 0; for (let i = 0; i < n; i++) mu += x[i]; mu /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (x[i] - mu) ** 2; v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5), y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - mu) * inv * g.data[i] + b.data[i];
  return y;
}

export function softmax(z) {
  let m = -Infinity; for (const v of z) m = Math.max(m, v);
  const e = Array.from(z, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export class GPT {
  constructor(json) {
    this.cfg = json.config;
    this.w = {};
    for (const [k, v] of Object.entries(json.weights)) this.w[k] = decode(v);
    this.probes = {};
    for (const [k, v] of Object.entries(json.probes)) this.probes[k] = { W: decode(v.W), b: decode(v.b), means: decode(v.means) };
    if (json.ego) this.ego = { R: json.ego.R, layer: json.ego.layer,
      visited: { W: decode(json.ego.visited.W), b: decode(json.ego.visited.b) },
      passages: { W: decode(json.ego.passages.W), b: decode(json.ego.passages.b) } };
  }

  // tokens: int[]; edit: optional {layer, delta: Float32Array} added at the last position
  // entering block `layer` (layer == n_layer edits the final stream).
  forward(tokens, edit = null) {
    const { d_model: D, n_head: H, n_layer: L } = this.cfg, w = this.w, S = tokens.length, dh = D / H;
    let x = tokens.map((t, p) => {
      const v = new Float32Array(D);
      for (let i = 0; i < D; i++) v[i] = w["tok.weight"].data[t * D + i] + w["pos.weight"].data[p * D + i];
      return v;
    });
    const resid = [], attn = [];
    const applyEdit = (l) => {
      if (edit && edit.layer === l) {
        const v = new Float32Array(x[S - 1]);
        for (let i = 0; i < D; i++) v[i] += edit.delta[i];
        x[S - 1] = v;
      }
    };
    for (let l = 0; l < L; l++) {
      applyEdit(l);
      resid.push(x);
      const P = `blocks.${l}.`;
      const qkv = x.map((xi) => linear(layerNorm(xi, w[P + "ln1.weight"], w[P + "ln1.bias"]), w[P + "attn.qkv.weight"], w[P + "attn.qkv.bias"]));
      const heads = [];
      const y = x.map(() => new Float32Array(D));
      for (let h = 0; h < H; h++) {
        const pat = [];
        for (let i = 0; i < S; i++) {
          const sc = new Float32Array(i + 1);
          for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let k = 0; k < dh; k++) s += qkv[i][h * dh + k] * qkv[j][D + h * dh + k];
            sc[j] = s / Math.sqrt(dh);
          }
          const a = softmax(sc);
          pat.push(a);
          for (let j = 0; j <= i; j++) for (let k = 0; k < dh; k++) y[i][h * dh + k] += a[j] * qkv[j][2 * D + h * dh + k];
        }
        heads.push(pat);
      }
      attn.push(heads);
      x = x.map((xi, i) => {
        const a = linear(y[i], w[P + "attn.proj.weight"], w[P + "attn.proj.bias"]);
        const r = new Float32Array(D);
        for (let k = 0; k < D; k++) r[k] = xi[k] + a[k];
        const hdn = linear(layerNorm(r, w[P + "ln2.weight"], w[P + "ln2.bias"]), w[P + "fc.weight"], w[P + "fc.bias"]).map(gelu);
        const m = linear(hdn, w[P + "out.weight"], w[P + "out.bias"]);
        for (let k = 0; k < D; k++) r[k] += m[k];
        return r;
      });
    }
    applyEdit(L);
    resid.push(x);
    const logits = x.map((xi) => linear(layerNorm(xi, w["ln_f.weight"], w["ln_f.bias"]), w["head.weight"], w["head.bias"]));
    return { logits, resid, attn };
  }

  // Egocentric map decoded from the residual stream: P(visited) for each offset in a
  // (2R+1)^2 window around the walker, and P(walked) for the passage east/south of each.
  egoMap(vec) {
    const sig = (z) => Array.from(z, (v) => 1 / (1 + Math.exp(-v)));
    return { visited: sig(linear(vec, this.ego.visited.W, this.ego.visited.b)),
             passages: sig(linear(vec, this.ego.passages.W, this.ego.passages.b)) };
  }

  probeLogits(layer, vec) {
    const p = this.probes[`resid_${layer}`];
    return linear(vec, p.W, p.b);
  }

  // Steering edit (see experiments/intervene.py, "probe_dir"):
  //   delta = alpha * |mu[target] - mu[current]| * unit(W[target] - W[current])
  // i.e. move along the probe's own direction for the target cell, with a step size set by
  // how far apart the two cells' average activations are. alpha ~ 2 works best.
  steerDelta(layer, vec, target, alpha = 2) {
    const p = this.probes[`resid_${layer}`];
    const [N, D] = p.W.shape, W = p.W.data, mu = p.means.data;
    const z = this.probeLogits(layer, vec);
    let cur = 0; for (let i = 1; i < N; i++) if (z[i] > z[cur]) cur = i;
    const delta = new Float32Array(D);
    if (cur === target) return delta;
    let dn = 0, mn = 0;
    for (let i = 0; i < D; i++) {
      const d = W[target * D + i] - W[cur * D + i]; delta[i] = d; dn += d * d;
      const m = mu[target * D + i] - mu[cur * D + i]; mn += m * m;
    }
    const s = alpha * Math.sqrt(mn) / Math.sqrt(dn);
    for (let i = 0; i < D; i++) delta[i] *= s;
    return delta;
  }
}
