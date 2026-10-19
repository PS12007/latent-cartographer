import { GPT, softmax } from "./model.js";
import { World, MOVES, UNK, START0, entropy } from "./world.js";

const $ = (s) => document.querySelector(s);
const C = {
  paper: "#f2ead8", ink: "#1f1b16", soft: "#8b7f6a", rule: "#cbbd9c",
  verm: [194, 56, 31], indigo: [36, 64, 107], ochre: [168, 116, 26],
};

let J, model, world, L, MAXT;
const WORLDS = {
  model: "A fixed 8×8 maze. One time in three the walk starts at a known cell; otherwise the model must work out where it is.",
  model_big12: "A 12×12 maze (144 cells) with the same 0.8M-parameter model. Localising takes longer; offline extraction recovers 261 of 264 edges between cells.",
  model_slip15: "Slippery: 15% of moves silently fail (the red dot sometimes stays put). Even from a known start the belief spreads out, and the model has to keep it graded.",
};
const cache = {};

const S = { start: null, cells: [], tokens: [], beliefs: [], layer: 3, alpha: 2, head: 0, steer: null, out: null, timer: null, ablated: new Set() };

function rand(n) { return Math.floor(Math.random() * n); }

function newWalk(start) {
  S.steer = null;
  const known = start !== null && start !== undefined;
  S.trueStart = known ? start : rand(world.N);
  S.start = known ? start : null;
  S.cells = [S.trueStart];
  S.tokens = [known ? START0 + start : UNK];
  S.beliefs = [world.prior(S.start)];
  update();
}

function step(m) {
  const c = S.cells.at(-1);
  if (!world.open[c][m] || S.tokens.length > MAXT) return false;
  S.steer = null;
  S.cells.push(Math.random() < world.slip ? c : world.nxt[c][m]);   // in the slippery world a move can silently fail
  S.tokens.push(m);
  S.beliefs.push(world.update(S.beliefs.at(-1), m));
  return true;
}

function randomStep() {
  const c = S.cells.at(-1);
  const legal = [0, 1, 2, 3].filter((d) => world.open[c][d]);
  return step(legal[rand(legal.length)]);
}

// ---- drawing
function drawMaze(cv, values, color, opts = {}) {
  const ctx = cv.getContext("2d");
  const W = cv.width, pad = opts.pad ?? 10, cs = (W - 2 * pad) / world.W;
  ctx.clearRect(0, 0, W, cv.height);
  ctx.fillStyle = "rgba(255,252,243,.35)";
  ctx.fillRect(pad, pad, cs * world.W, cs * world.H);
  if (values) {
    let mx = 0; for (const v of values) mx = Math.max(mx, v);
    for (let c = 0; c < world.N; c++) {
      const v = values[c];
      if (v <= 1e-4) continue;
      const a = Math.min(1, Math.pow(v / Math.max(mx, 1e-9), 0.55)) * (0.25 + 0.75 * Math.min(1, mx * 1.5));
      const r = Math.floor(c / world.W), q = c % world.W;
      ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${(0.08 + 0.85 * a).toFixed(3)})`;
      ctx.fillRect(pad + q * cs + 1.5, pad + r * cs + 1.5, cs - 3, cs - 3);
    }
  }
  if (opts.grid !== false) { // faint survey grid
    ctx.strokeStyle = C.rule; ctx.lineWidth = 0.5; ctx.setLineDash([2, 3]);
    for (let i = 1; i < world.W; i++) { ctx.beginPath(); ctx.moveTo(pad + i * cs, pad); ctx.lineTo(pad + i * cs, pad + world.H * cs); ctx.stroke(); }
    for (let i = 1; i < world.H; i++) { ctx.beginPath(); ctx.moveTo(pad, pad + i * cs); ctx.lineTo(pad + world.W * cs, pad + i * cs); ctx.stroke(); }
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = opts.wallColor ?? C.ink; ctx.lineCap = "round"; ctx.lineWidth = opts.wall ?? 2.4;
  const op = opts.open ?? world.open;
  ctx.beginPath();
  for (let c = 0; c < world.N; c++) {
    const r = Math.floor(c / world.W), q = c % world.W, x = pad + q * cs, y = pad + r * cs;
    if (!op[c][0]) { ctx.moveTo(x, y); ctx.lineTo(x + cs, y); }
    if (!op[c][3]) { ctx.moveTo(x, y); ctx.lineTo(x, y + cs); }
    if (r === world.H - 1) { ctx.moveTo(x, y + cs); ctx.lineTo(x + cs, y + cs); }
    if (q === world.W - 1) { ctx.moveTo(x + cs, y); ctx.lineTo(x + cs, y + cs); }
  }
  ctx.stroke();
  return { pad, cs };
}

function center(c, g) {
  return [g.pad + (c % world.W + 0.5) * g.cs, g.pad + (Math.floor(c / world.W) + 0.5) * g.cs];
}

function drawTruth() {
  const cv = $("#truth");
  const g = drawMaze(cv, null, null);
  const ctx = cv.getContext("2d");
  if (!$("#reveal").checked) return;
  // trail
  ctx.strokeStyle = `rgba(${C.verm.join(",")},.45)`; ctx.lineWidth = 2; ctx.lineJoin = "round";
  ctx.beginPath();
  S.cells.forEach((c, i) => { const [x, y] = center(c, g); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  const [sx, sy] = center(S.cells[0], g);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.strokeRect(sx - 6, sy - 6, 12, 12);
  const [x, y] = center(S.cells.at(-1), g);
  ctx.fillStyle = `rgb(${C.verm.join(",")})`; ctx.beginPath(); ctx.arc(x, y, 8, 0, 7); ctx.fill();
  ctx.strokeStyle = C.paper; ctx.lineWidth = 2; ctx.stroke();
}

function probeDist(layer, vec) { return softmax(model.probeLogits(layer, vec)); }

function update() {
  const out = model.forward(S.tokens, null, S.ablated);
  S.out = out;
  const last = S.tokens.length - 1;
  const bayes = S.beliefs.at(-1);
  const mind = probeDist(S.layer, out.resid[S.layer][last]);

  drawTruth();
  drawMaze($("#bayes"), bayes, C.indigo);
  const g = drawMaze($("#mind"), mind, C.verm);
  if (S.steer !== null) {
    const ctx = $("#mind").getContext("2d");
    const [x, y] = center(S.steer, g);
    ctx.strokeStyle = `rgb(${C.ochre.join(",")})`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, g.cs * 0.42, 0, 7); ctx.stroke();
  }
  $("#layerName").textContent = S.layer === 0 ? "the embedding" : `block ${S.layer}`;

  // depth strip
  const strip = $("#strip");
  if (!strip.children.length) {
    for (let l = 0; l <= L; l++) {
      const d = document.createElement("div");
      d.className = "cell"; d.dataset.layer = l;
      d.innerHTML = `<canvas width="140" height="140"></canvas><div>${l === 0 ? "embed" : "block " + l}</div>`;
      d.onclick = () => { S.layer = l; S.steer = null; update(); };
      strip.appendChild(d);
    }
  }
  [...strip.children].forEach((d, l) => {
    d.classList.toggle("sel", l === S.layer);
    drawMaze(d.querySelector("canvas"), probeDist(l, out.resid[l][last]), C.verm, { pad: 4, wall: 1.4, grid: false });
  });

  // metrics
  const cand = bayes.filter((v) => v > 1e-9).length;
  let ov = 0; for (let c = 0; c < world.N; c++) ov += Math.min(bayes[c], mind[c]);
  $("#mSteps").textContent = last;
  $("#mH").textContent = entropy(bayes).toFixed(2) + " nats";
  $("#mCand").textContent = cand;
  $("#mOv").textContent = (100 * ov).toFixed(0) + "%";

  // next-move bars
  const pm = softmax(out.logits[last]);
  const po = world.predictive(bayes);
  let ps = null;
  if (S.steer !== null) {
    const delta = model.steerDelta(S.layer, out.resid[S.layer][last], S.steer, S.alpha);
    ps = softmax(model.forward(S.tokens, { layer: S.layer, delta }, S.ablated).logits[last]);
  }
  const here = S.cells.at(-1);
  $("#bars").innerHTML = MOVES.map((m, d) => `
    <div class="bar-row ${world.open[here][d] || !$("#reveal").checked ? "" : "wall"}">
      <span class="lbl">${m}</span>
      <div class="track">
        <div class="b model" style="width:${(pm[d] * 100).toFixed(1)}%"></div>
        <div class="b opt" style="width:${(po[d] * 100).toFixed(1)}%"></div>
        ${ps ? `<div class="b steer" style="width:${(ps[d] * 100).toFixed(1)}%"></div>` : ""}
      </div>
      <span>${((ps ?? pm)[d] * 100).toFixed(0)}%</span>
    </div>`).join("");

  const v = $("#verdict");
  if (ps) {
    const said = MOVES.filter((_, d) => ps[d] > 0.1).join(" ") || "—";
    const truth = MOVES.filter((_, d) => world.open[S.steer][d]).join(" ");
    const ok = MOVES.every((_, d) => (ps[d] > 0.1) === world.open[S.steer][d]);
    v.className = "verdict on";
    v.innerHTML = `Steered to cell <b>(${Math.floor(S.steer / world.W)},${S.steer % world.W})</b> at ${S.layer === 0 ? "the embedding" : "block " + S.layer}. ` +
      `The model now expects <span class="mono">${said}</span>; that cell’s open sides are <span class="mono">${truth}</span>. ` +
      (ok ? "<b>It navigates by the map it was steered to.</b>" : "<i>Partial: the edit didn’t fully take at this depth.</i>");
  } else v.className = "verdict";

  // compass enable
  document.querySelectorAll(".compass button").forEach((b) => {
    b.disabled = !world.open[here][+b.dataset.move] || last >= MAXT;
  });

  // tape
  $("#tape").innerHTML = S.tokens.map((t, i) => {
    const s = i === 0 ? (t === UNK ? "?" : `@${Math.floor((t - START0) / world.W)},${(t - START0) % world.W}`) : MOVES[t];
    return `<span class="${i === 0 ? "start" : ""} ${i === last ? "last" : ""}">${s}</span>`;
  }).join("") + (last >= MAXT ? ` <em class="note">context full</em>` : "");

  drawAttention();
  if (model.ego) drawEgo(out.resid[model.ego.layer][last]);
}

// ---- Fig. e: egocentric sketch
const DRS = [-1, 0, 1, 0], DCS = [0, 1, 0, -1];
function trueEgo() {
  const R = model.ego.R, K = 2 * R + 1;
  let r = 0, c = 0;
  const cells = new Set(["0,0"]), edges = new Set();
  for (let i = 1; i < S.tokens.length; i++) {
    const m = S.tokens[i], r2 = r + DRS[m], c2 = c + DCS[m];
    if (m === 1) edges.add(`${r},${c},0`); else if (m === 3) edges.add(`${r2},${c2},0`);
    else if (m === 2) edges.add(`${r},${c},1`); else edges.add(`${r2},${c2},1`);
    r = r2; c = c2; cells.add(`${r},${c}`);
  }
  const visited = new Array(K * K).fill(0), passages = new Array(K * K * 2).fill(0);
  for (const k of cells) { const [a, b] = k.split(",").map(Number); const i = a - r + R, j = b - c + R; if (i >= 0 && j >= 0 && i < K && j < K) visited[i * K + j] = 1; }
  for (const k of edges) { const [a, b, d] = k.split(",").map(Number); const i = a - r + R, j = b - c + R; if (i >= 0 && j >= 0 && i < K && j < K) passages[(i * K + j) * 2 + d] = 1; }
  return { visited, passages };
}

function drawEgoCanvas(cv, ego, color) {
  const R = model.ego.R, K = 2 * R + 1, ctx = cv.getContext("2d"), W = cv.width, pad = 8, cs = (W - 2 * pad) / K;
  ctx.clearRect(0, 0, W, cv.height);
  ctx.fillStyle = "rgba(255,252,243,.35)"; ctx.fillRect(pad, pad, cs * K, cs * K);
  ctx.strokeStyle = C.rule; ctx.lineWidth = 0.5; ctx.setLineDash([2, 3]);
  for (let i = 0; i <= K; i++) {
    ctx.beginPath(); ctx.moveTo(pad + i * cs, pad); ctx.lineTo(pad + i * cs, pad + K * cs); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, pad + i * cs); ctx.lineTo(pad + K * cs, pad + i * cs); ctx.stroke();
  }
  ctx.setLineDash([]);
  const rgba = (a) => `rgba(${color.join(",")},${a.toFixed(3)})`;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    const v = ego.visited[i * K + j];
    if (v > 0.05) { ctx.fillStyle = rgba(0.12 + 0.3 * v); ctx.fillRect(pad + j * cs + 2, pad + i * cs + 2, cs - 4, cs - 4); }
  }
  ctx.lineCap = "round"; ctx.lineWidth = cs * 0.22;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) for (let d = 0; d < 2; d++) {
    const v = ego.passages[(i * K + j) * 2 + d];
    if (v < 0.08) continue;
    const x = pad + (j + 0.5) * cs, y = pad + (i + 0.5) * cs;
    ctx.strokeStyle = rgba(Math.min(1, 0.15 + 0.85 * v));
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (d === 0 ? cs : 0), y + (d === 1 ? cs : 0)); ctx.stroke();
  }
  const x = pad + (R + 0.5) * cs, y = pad + (R + 0.5) * cs;
  ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, cs * 0.3, 0, 7); ctx.stroke();
}

function drawEgo(vec) {
  drawEgoCanvas($("#egoTrue"), trueEgo(), C.indigo);
  drawEgoCanvas($("#egoModel"), model.egoMap(vec), C.verm);
}

function drawAttention() {
  const hp = $("#headpick");
  if (!hp.children.length) {
    for (let l = 0; l < L; l++) for (let h = 0; h < model.cfg.n_head; h++) {
      const b = document.createElement("button");
      b.textContent = `B${l + 1}H${h}`; b.dataset.k = l * 100 + h;
      b.onclick = () => { S.attnLayer = l; S.head = h; drawAttention(); };
      hp.appendChild(b);
    }
    S.attnLayer = 0;
  }
  [...hp.children].forEach((b) => {
    b.classList.toggle("sel", +b.dataset.k === S.attnLayer * 100 + S.head);
    b.classList.toggle("off", S.ablated.has(`${Math.floor(+b.dataset.k / 100)},${+b.dataset.k % 100}`));
  });
  const key = `${S.attnLayer},${S.head}`;
  $("#koBtn").textContent = S.ablated.has(key) ? `Restore B${S.attnLayer + 1}H${S.head}` : `Knock out B${S.attnLayer + 1}H${S.head}`;
  $("#koReset").disabled = S.ablated.size === 0;
  const row = S.out.attn[S.attnLayer][S.head].at(-1);
  $("#attn").innerHTML = S.tokens.map((t, i) => {
    const a = row[i], s = i === 0 ? (t === UNK ? "?" : "@") : MOVES[t];
    return `<span title="${(a * 100).toFixed(1)}%" style="background:rgba(${C.verm.join(",")},${Math.min(1, a * 3).toFixed(3)});color:${a > 0.2 ? C.paper : C.ink}">${s}</span>`;
  }).join("");
}

// ---- interaction
function cellAt(cv, ev, pad) {
  const r = cv.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * cv.width, y = ((ev.clientY - r.top) / r.height) * cv.height;
  const cs = (cv.width - 2 * pad) / world.W;
  const q = Math.floor((x - pad) / cs), rr = Math.floor((y - pad) / cs);
  if (q < 0 || rr < 0 || q >= world.W || rr >= world.H) return null;
  return rr * world.W + q;
}

$("#truth").onclick = (e) => { const c = cellAt($("#truth"), e, 10); if (c !== null) { stopPlay(); newWalk(c); } };
$("#mind").onclick = (e) => { const c = cellAt($("#mind"), e, 10); if (c !== null) { stopPlay(); S.steer = c; update(); } };
$("#reveal").onchange = update;
$("#alpha").oninput = (e) => { S.alpha = +e.target.value; $("#alphaV").textContent = S.alpha.toFixed(1); update(); };
$("#lost").onclick = () => { stopPlay(); newWalk(null); };
$("#wander").onclick = () => { for (let i = 0; i < 8; i++) randomStep(); update(); };
document.querySelectorAll(".compass button").forEach((b) => (b.onclick = () => { if (step(+b.dataset.move)) update(); }));

function stopPlay() { clearInterval(S.timer); S.timer = null; $("#play").textContent = "Auto-walk ▸"; }
$("#play").onclick = () => {
  if (S.timer) return stopPlay();
  $("#play").textContent = "Pause ❚❚";
  S.timer = setInterval(() => { if (!randomStep()) stopPlay(); update(); }, 380);
};

const KEYS = { ArrowUp: 0, w: 0, ArrowRight: 1, d: 1, ArrowDown: 2, s: 2, ArrowLeft: 3, a: 3 };
addEventListener("keydown", (e) => {
  if (e.key in KEYS) { e.preventDefault(); if (step(KEYS[e.key])) update(); }
});

// ---- Fig. e: extract the whole maze by steering
function localisedWalk() {
  // a walk after which the walker's cell is (nearly) certain; in the slippery world, a short known-start walk
  const slippy = world.slip > 0;
  let best = null;
  for (let tries = 0; tries < 400; tries++) {
    let c = rand(world.N), b = world.prior(slippy ? c : null);
    const toks = [slippy ? START0 + c : UNK];
    for (let t = 0; t < (slippy ? 3 : 40); t++) {
      const legal = [0, 1, 2, 3].filter((d) => world.open[c][d]);
      const m = legal[rand(legal.length)];
      toks.push(m); b = world.update(b, m);
      if (!(Math.random() < world.slip)) c = world.nxt[c][m];
    }
    const conf = Math.max(...b);
    if (conf > (slippy ? 0.8 : 0.99)) return toks;
    if (!best || conf > best[0]) best = [conf, toks];
  }
  return best[1];
}

let extracting = false;
async function extract() {
  if (extracting) return;
  extracting = true;
  $("#extractBtn").disabled = true;
  const layer = 3, alpha = 2;
  const bases = [localisedWalk(), localisedWalk(), localisedWalk()].map((t) => ({ t, o: model.forward(t) }));
  const P = Array.from({ length: world.N }, () => null);
  const recon = () => world.open.map((row, c) => row.map((trueOpen, d) => {
    const nb = world.nxt[c][d];
    if (!trueOpen && nb === c) return false;                       // outer boundary (no neighbour)
    const r = Math.floor(c / world.W) + [-1, 0, 1, 0][d], q = (c % world.W) + [0, 1, 0, -1][d];
    if (r < 0 || q < 0 || r >= world.H || q >= world.W) return false;
    const nbc = r * world.W + q;
    const votes = [];
    if (P[c]) votes.push(P[c][d]);
    if (P[nbc]) votes.push(P[nbc][(d + 2) % 4]);
    return votes.length ? votes.reduce((a, b) => a + b) / votes.length > 0.1 : true;
  }));
  for (let c = 0; c < world.N; c++) {
    const acc = [0, 0, 0, 0];
    for (const { t, o } of bases) {
      const delta = model.steerDelta(layer, o.resid[layer].at(-1), c, alpha);
      softmax(model.forward(t, { layer, delta }).logits.at(-1)).forEach((v, d) => (acc[d] += v / bases.length));
    }
    P[c] = acc;
    const hl = new Float64Array(world.N); hl[c] = 1;
    drawMaze($("#extModel"), hl, C.ochre, { open: recon() });
    await new Promise((r) => setTimeout(r, 30));
  }
  const rec = recon();
  let ok = 0, tot = 0;
  for (let c = 0; c < world.N; c++) for (const d of [1, 2]) {
    const r = Math.floor(c / world.W), q = c % world.W;
    if ((d === 1 && q + 1 < world.W) || (d === 2 && r + 1 < world.H)) { tot++; ok += rec[c][d] === world.open[c][d]; }
  }
  drawMaze($("#extModel"), null, null, { open: rec });
  $("#extractScore").textContent = `${ok} / ${tot} interior edges (passages and walls) recovered correctly`;
  $("#extractBtn").disabled = false;
  extracting = false;
}
$("#extractBtn").onclick = extract;
$("#koBtn").onclick = () => {
  const key = `${S.attnLayer},${S.head}`;
  if (S.ablated.has(key)) S.ablated.delete(key); else S.ablated.add(key);
  update();
};
$("#koReset").onclick = () => { S.ablated.clear(); update(); };
async function loadWorld(name) {
  $("#app").classList.add("loading");
  if (!cache[name]) cache[name] = await (await fetch(`assets/${name}.json`)).json();
  J = cache[name];
  model = new GPT(J);
  world = new World(J.maze, J.slip || 0);
  L = model.cfg.n_layer; MAXT = model.cfg.block_size - 1;
  // self-test: JS forward pass vs the PyTorch reference stored with the weights
  const ref = model.forward(J.reference.tokens);
  let err = 0;
  ref.logits.forEach((row, i) => row.forEach((v, j) => (err = Math.max(err, Math.abs(v - J.reference.logits[i][j])))));
  console.log(`${name}: JS vs PyTorch max |Δlogit| = ${err.toExponential(2)}`);
  $("#parity").textContent = `JS ⇄ PyTorch parity: max |Δlogit| = ${err.toExponential(1)}`;
  // reset everything that depends on the model
  $("#strip").innerHTML = ""; $("#headpick").innerHTML = "";
  S.ablated = new Set(); S.layer = Math.min(3, L); S.head = 0; S.attnLayer = 0; S.steer = null;
  document.querySelectorAll(".worlds button").forEach((b) => b.classList.toggle("sel", b.dataset.world === name));
  $("#worldNote").textContent = WORLDS[name];
  $(".ego").classList.toggle("na", !model.ego);
  $("#extractScore").textContent = "";
  drawMaze($("#extTruth"), null, null);
  drawMaze($("#extModel"), null, null, { open: world.open.map((row, c) => row.map((_, d) => {
    const r = Math.floor(c / world.W) + DRS[d], q = (c % world.W) + DCS[d];
    return r >= 0 && q >= 0 && r < world.H && q < world.W;           // only the outer boundary until extracted
  })) });
  newWalk(world.slip > 0 ? rand(world.N) : null);   // slippery world: show a known start spreading out
  for (let i = 0; i < 6; i++) randomStep();
  update();
  $("#app").classList.remove("loading");
}

document.querySelectorAll(".worlds button").forEach((b) => (b.onclick = () => { if (!extracting) { stopPlay(); loadWorld(b.dataset.world); } }));
const hashWorld = location.hash.startsWith("#world=") ? location.hash.slice(7).split("&")[0] : "model";
await loadWorld(WORLDS[hashWorld] ? hashWorld : "model");
if (location.hash.includes("extract")) setTimeout(extract, 300);
