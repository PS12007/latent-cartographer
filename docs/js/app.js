import { GPT, softmax } from "./model.js";
import { World, MOVES, UNK, START0, entropy } from "./world.js";

const $ = (s) => document.querySelector(s);
const C = {
  paper: "#f2ead8", ink: "#1f1b16", soft: "#8b7f6a", rule: "#cbbd9c",
  verm: [194, 56, 31], indigo: [36, 64, 107], ochre: [168, 116, 26],
};

const res = await fetch("assets/model.json");
const J = await res.json();
const model = new GPT(J);
const world = new World(J.maze);
const L = model.cfg.n_layer, MAXT = model.cfg.block_size - 1;

// ---- self-test: JS forward pass vs PyTorch reference
{
  const out = model.forward(J.reference.tokens);
  let err = 0;
  out.logits.forEach((row, i) => row.forEach((v, j) => (err = Math.max(err, Math.abs(v - J.reference.logits[i][j])))));
  console.log(`JS vs PyTorch max |Δlogit| = ${err.toExponential(2)}`);
  const f = document.querySelector("footer");
  f.insertAdjacentHTML("beforeend", ` · JS ⇄ PyTorch parity: max |Δlogit| = ${err.toExponential(1)}`);
}

const S = { start: null, cells: [], tokens: [], beliefs: [], layer: 3, alpha: 2, head: 0, steer: null, out: null, timer: null };

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
  S.cells.push(world.nxt[c][m]);
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
  const out = model.forward(S.tokens);
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
    ps = softmax(model.forward(S.tokens, { layer: S.layer, delta }).logits[last]);
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
}

function drawAttention() {
  const hp = $("#headpick");
  if (!hp.children.length) {
    for (let l = 0; l < L; l++) for (let h = 0; h < model.cfg.n_head; h++) {
      const b = document.createElement("button");
      b.textContent = `L${l + 1}H${h}`; b.dataset.k = l * 100 + h;
      b.onclick = () => { S.attnLayer = l; S.head = h; drawAttention(); };
      hp.appendChild(b);
    }
    S.attnLayer = 0;
  }
  [...hp.children].forEach((b) => b.classList.toggle("sel", +b.dataset.k === S.attnLayer * 100 + S.head));
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
  for (;;) {
    let c = rand(world.N), b = world.prior(null);
    const toks = [UNK];
    for (let t = 0; t < 40; t++) {
      const legal = [0, 1, 2, 3].filter((d) => world.open[c][d]);
      const m = legal[rand(legal.length)];
      toks.push(m); b = world.update(b, m); c = world.nxt[c][m];
    }
    if (Math.max(...b) > 0.99) return toks;
  }
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
  $("#extractScore").textContent = `${ok} / ${tot} interior walls recovered correctly`;
  $("#extractBtn").disabled = false;
  extracting = false;
}
drawMaze($("#extTruth"), null, null);
drawMaze($("#extModel"), null, null, { open: world.open.map((r) => r.map(() => true)) });
$("#extractBtn").onclick = extract;
if (location.hash === "#extract") setTimeout(extract, 300);

newWalk(null);
for (let i = 0; i < 6; i++) randomStep();
update();
$("#app").classList.remove("loading");
