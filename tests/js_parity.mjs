// node tests/js_parity.mjs — checks the browser model against the PyTorch reference.
import { readFileSync } from "node:fs";
import { GPT, softmax } from "../docs/js/model.js";
import { World, START0, UNK } from "../docs/js/world.js";

const J = JSON.parse(readFileSync(new URL("../docs/assets/model.json", import.meta.url)));
const model = new GPT(J), world = new World(J.maze);
const out = model.forward(J.reference.tokens);
let err = 0;
out.logits.forEach((r, i) => r.forEach((v, j) => (err = Math.max(err, Math.abs(v - J.reference.logits[i][j])))));
let perr = 0;
for (const [k, ref] of Object.entries(J.reference.probe_last)) {
  const l = +k.split("_")[1];
  softmax(model.probeLogits(l, out.resid[l].at(-1))).forEach((v, i) => (perr = Math.max(perr, Math.abs(v - ref[i]))));
}
console.log(`max |logit diff| ${err.toExponential(2)}, max |probe prob diff| ${perr.toExponential(2)}`);
let kerr = 0;
if (J.reference.ablate_0_1_logits) {
  const ko = model.forward(J.reference.tokens, null, new Set(["0,1"]));
  ko.logits.forEach((r, i) => r.forEach((v, j) => (kerr = Math.max(kerr, Math.abs(v - J.reference.ablate_0_1_logits[i][j])))));
  console.log(`knock-out B1H1: max |logit diff| ${kerr.toExponential(2)}`);
}
if (err > 1e-3 || perr > 1e-3 || kerr > 1e-3) { console.error("PARITY FAIL"); process.exit(1); }

// steering sanity: 40-move walks, steer at resid_3 with alpha 2 to a random cell with different walls
let ok = 0, n = 0, seed = 1;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
for (let trial = 0; trial < 60; trial++) {
  let c = Math.floor(rnd() * world.N); const toks = [UNK]; let b = world.prior(null);
  for (let t = 0; t < 40; t++) {
    const legal = [0, 1, 2, 3].filter((d) => world.open[c][d]);
    const m = legal[Math.floor(rnd() * legal.length)];
    toks.push(m); b = world.update(b, m); c = world.nxt[c][m];
  }
  if (Math.max(...b) < 0.99) continue;
  let tgt; do { tgt = Math.floor(rnd() * world.N); } while (world.open[tgt].every((v, d) => v === world.open[c][d]));
  const o = model.forward(toks);
  const delta = model.steerDelta(3, o.resid[3].at(-1), tgt, 2);
  const p = softmax(model.forward(toks, { layer: 3, delta }).logits.at(-1));
  n++; if (p.every((v, d) => (v > 0.1) === world.open[tgt][d])) ok++;
}
console.log(`JS steering success ${ok}/${n}`);
