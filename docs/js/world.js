// Maze world + exact Bayesian filter (JS port of cartographer/env.py).
export const MOVES = ["N", "E", "S", "W"];
const DR = [-1, 0, 1, 0], DC = [0, 1, 0, -1];
export const UNK = 4, START0 = 5;

export class World {
  constructor(m, slip = 0) {
    this.slip = slip;   // probability that a move silently fails (walker stays put)
    this.H = m.H; this.W = m.W; this.N = m.H * m.W;
    this.open = m.open.map((r) => r.map(Boolean));
    this.deg = this.open.map((r) => r.filter(Boolean).length);
    this.nxt = this.open.map((r, c) => r.map((ok, d) => {
      if (!ok) return c;
      const rr = Math.floor(c / this.W) + DR[d], cc = (c % this.W) + DC[d];
      return rr * this.W + cc;
    }));
  }

  prior(start) {
    const b = new Float64Array(this.N);
    if (start === null) b.fill(1 / this.N); else b[start] = 1;
    return b;
  }

  // posterior after observing move m
  update(b, m) {
    const out = new Float64Array(this.N);
    for (let c = 0; c < this.N; c++) if (b[c] && this.open[c][m]) {
      out[this.nxt[c][m]] += (1 - this.slip) * b[c] / this.deg[c];
      out[c] += this.slip * b[c] / this.deg[c];
    }
    let s = 0; for (const v of out) s += v;
    for (let c = 0; c < this.N; c++) out[c] /= s;
    return out;
  }

  predictive(b) {
    const p = [0, 0, 0, 0];
    for (let c = 0; c < this.N; c++) if (b[c]) for (let d = 0; d < 4; d++) if (this.open[c][d]) p[d] += b[c] / this.deg[c];
    return p;
  }
}

export function entropy(p) {
  let h = 0; for (const v of p) if (v > 1e-12) h -= v * Math.log(v);
  return h;
}
