/* ═══════════════════════════════════════════════════════════════════════════
   FleetMind Mission Control — client engine.

   • Polls /state at 8 Hz for world data, renders the map on <canvas> at 60 fps.
   • Forklift motion is exponentially smoothed toward the latest snapshot so trucks
     GLIDE (with fading motion trails) instead of teleporting between polls.
   • Roster cards, KPI counters, activity feed and chat are driven from the same state.
   • Chat posts to /chat -> GenAI agent (NIM when up, offline parser otherwise).
   ═══════════════════════════════════════════════════════════════════════════ */
"use strict";

const POLL_MS = 125;                 // 8 Hz data
const SMOOTH_TAU = 0.10;             // motion smoothing time-constant (s)
const TRAIL_MAX = 46;                // trail points per forklift

const PHASE = {
  idle:       { c: "#8b97a8", label: "Idle" },
  navigating: { c: "#3b82f6", label: "Navigating" },
  picking:    { c: "#f5b301", label: "Picking" },
  lifting:    { c: "#f5b301", label: "Lifting" },
  carrying:   { c: "#76b900", label: "Carrying" },
  dropping:   { c: "#e0721c", label: "Dropping" },
  returning:  { c: "#a371f7", label: "Returning" },
};
const NV = "#76b900", AMBER = "#f5b301", RED = "#ff4d5e";

const EXAMPLES = [
  "What's the status of the fleet?",
  "Move pallet 1 to stage 1",
  "Clear all the pallets off the racks into staging",
  "Send all forklifts home",
  "Spill in stage 2 — block it",
];

// ── state ────────────────────────────────────────────────────────────────
let snap = null;              // latest /state snapshot
let plan = null;              // latest /plan (cuOpt assignment + CBS paths)
let bounds = null;            // cached world bounds {minX,maxX,minY,maxY}
let selected = null;         // selected forklift name (roster click)
let online = false;
const view = {};             // per-forklift smoothed render state {x,y,yaw,trail[]}
const kpiShown = { delivered: 0, transit: 0, waiting: 0, active: 0, throughput: 0, sim: 0 };
const kpiTarget = { ...kpiShown };
const spark = { delivered: [], throughput: [] };
let lastSparkT = -1;

// activity diffing
const prevPhase = {};
let prevDelivered = new Set();
let seededEvents = false;

// ── DOM refs ──────────────────────────────────────────────────────────────
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const rosterEl = document.getElementById("roster");
const chatLog = document.getElementById("chatLog");
const activityLog = document.getElementById("activityLog");
const rcards = {};           // name -> {el, refs...}

// ═══════════════════════════════════════════════════════════════════════════
// Networking
// ═══════════════════════════════════════════════════════════════════════════
async function poll() {
  try {
    const r = await fetch("/state", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    snap = await r.json();
    setOnline(true);
    ingest(snap);
  } catch (e) {
    setOnline(false);
  }
}

async function health() {
  try {
    const r = await fetch("/health", { cache: "no-store" });
    const j = await r.json();
    document.getElementById("agentMode").textContent = (j.backend || "mock");
  } catch {}
}

// cuOpt + CBS plan (the last dispatched optimisation). Polled slowly; it only
// changes when the operator asks the agent to optimise/clear the fleet.
async function pollPlan() {
  try {
    const r = await fetch("/plan", { cache: "no-store" });
    if (!r.ok) return;
    const j = await r.json();
    plan = (j && j.solver) ? j : null;
    renderPlanPanel();
  } catch {}
}

function setOnline(v) {
  if (v === online) return;
  online = v;
  const pill = document.getElementById("statusPill");
  const txt = document.getElementById("statusText");
  pill.classList.toggle("online", v);
  pill.classList.toggle("offline", !v);
  txt.textContent = v ? "BRIDGE ONLINE" : "BRIDGE OFFLINE";
}

// ═══════════════════════════════════════════════════════════════════════════
// Ingest snapshot -> init view states, compute bounds, diff events, update DOM
// ═══════════════════════════════════════════════════════════════════════════
function ingest(s) {
  const fks = s.forklifts || {};

  computeBounds(s);   // expand-only; keeps wandering forklifts on-map

  // init/refresh smoothed view states
  for (const [name, fk] of Object.entries(fks)) {
    if (!view[name]) {
      view[name] = { x: fk.x, y: fk.y, yaw: fk.yaw, trail: [] };
      rebuildRoster();
    }
  }

  detectEvents(s);
  updateRosterData(s);
  updateKpiTargets(s);
}

function computeBounds(s) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const eat = (x, y) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  };
  const nodes = (s.graph && s.graph.nodes) || {};
  for (const [, xy] of Object.entries(nodes)) eat(xy[0], xy[1]);
  for (const z of Object.values(s.zones || {})) eat(z.x, z.y);
  for (const p of Object.values(s.pallets || {})) eat(p.x, p.y);
  for (const fk of Object.values(s.forklifts || {})) eat(fk.x, fk.y);
  if (minX === Infinity) return;
  const padX = (maxX - minX) * 0.10 + 2.5, padY = (maxY - minY) * 0.10 + 2.5;
  const b = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  // Expand-only: keep everything ever seen in view so wandering trucks never
  // clip to the map edge, but the frame stays stable (no zoom jitter).
  if (!bounds) bounds = b;
  else bounds = {
    minX: Math.min(bounds.minX, b.minX), maxX: Math.max(bounds.maxX, b.maxX),
    minY: Math.min(bounds.minY, b.minY), maxY: Math.max(bounds.maxY, b.maxY),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// World -> screen transform (fit + flip Y)
// ═══════════════════════════════════════════════════════════════════════════
let TF = { s: 1, ox: 0, oy: 0 };
function computeTransform(w, h) {
  if (!bounds) return;
  const wx = bounds.maxX - bounds.minX, wy = bounds.maxY - bounds.minY;
  const s = Math.min(w / wx, h / wy);
  TF.s = s;
  TF.ox = (w - wx * s) / 2 - bounds.minX * s;
  TF.oy = (h - wy * s) / 2 + bounds.maxY * s;  // flipped
}
const sx = (x) => TF.ox + x * TF.s;
const sy = (y) => TF.oy - y * TF.s;

// ═══════════════════════════════════════════════════════════════════════════
// Render loop (60 fps)
// ═══════════════════════════════════════════════════════════════════════════
let lastT = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;

  // smooth view states toward latest snapshot
  const k = 1 - Math.exp(-dt / SMOOTH_TAU);
  if (snap) {
    for (const [name, fk] of Object.entries(snap.forklifts || {})) {
      const v = view[name]; if (!v) continue;
      v.x += (fk.x - v.x) * k;
      v.y += (fk.y - v.y) * k;
      v.yaw += wrap(fk.yaw - v.yaw) * k;
      const moving = (fk.speed || 0) > 0.05;
      const tail = v.trail[v.trail.length - 1];
      if (moving && (!tail || dist(tail, v) > 0.18)) {
        v.trail.push({ x: v.x, y: v.y });
        if (v.trail.length > TRAIL_MAX) v.trail.shift();
      } else if (!moving && v.trail.length) {
        if (Math.random() < 0.25) v.trail.shift();   // fade out when parked
      }
    }
  }

  draw(now);
  animateKpis(dt);
  updateClock();
  requestAnimationFrame(frame);
}

function draw(now) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!snap || !bounds) return;
  computeTransform(w, h);

  drawLanes(snap);
  drawRacks(snap);
  drawZones(snap);
  drawPallets(snap);
  drawPlanPaths(snap, now);
  drawRoutes(snap, now);
  drawTrails(snap);
  drawForklifts(snap, now);
  drawConflicts(snap, now);
}

function drawLanes(s) {
  const nodes = s.graph.nodes;
  ctx.lineCap = "round";
  for (const [a, b] of s.graph.edges) {
    if (!nodes[a] || !nodes[b]) continue;
    const x1 = sx(nodes[a][0]), y1 = sy(nodes[a][1]);
    const x2 = sx(nodes[b][0]), y2 = sy(nodes[b][1]);
    ctx.strokeStyle = "rgba(40,54,76,.55)"; ctx.lineWidth = 7;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.strokeStyle = "rgba(60,80,112,.6)"; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (const [id, xy] of Object.entries(nodes)) {
    if (id.startsWith("rack")) continue;
    ctx.fillStyle = "rgba(70,92,128,.55)";
    ctx.beginPath(); ctx.arc(sx(xy[0]), sy(xy[1]), 1.6, 0, 7); ctx.fill();
  }
}

function drawRacks(s) {
  const nodes = s.graph.nodes;
  const u = TF.s;
  for (const [id, xy] of Object.entries(nodes)) {
    if (!id.startsWith("rack")) continue;
    const x = sx(xy[0]), y = sy(xy[1]), r = 1.5 * u;
    roundRect(x - r, y - r, r * 2, r * 2, 6);
    const g = ctx.createLinearGradient(x, y - r, x, y + r);
    g.addColorStop(0, "#1b2534"); g.addColorStop(1, "#141c28");
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = "#2c3a51"; ctx.lineWidth = 1.4; ctx.stroke();
    // shelf lines
    ctx.strokeStyle = "rgba(70,92,128,.4)"; ctx.lineWidth = 1;
    for (const f of [-0.45, 0.1]) {
      ctx.beginPath(); ctx.moveTo(x - r + 3, y + f * r); ctx.lineTo(x + r - 3, y + f * r); ctx.stroke();
    }
    ctx.fillStyle = "rgba(90,112,148,.55)";
    ctx.font = "600 9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("RACK", x, y + r * 0.55);
  }
}

function drawZones(s) {
  const u = TF.s;
  for (const [id, z] of Object.entries(s.zones || {})) {
    const x = sx(z.x), y = sy(z.y), r = 1.25 * u;
    const col = z.blocked ? RED : NV;
    roundRect(x - r, y - r, r * 2, r * 2, 7);
    ctx.fillStyle = hexA(col, .09); ctx.fill();
    ctx.save();
    ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.setLineDash([7, 5]);
    ctx.stroke(); ctx.restore();
    ctx.fillStyle = col;
    ctx.font = "700 9.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(id.toUpperCase() + (z.blocked ? "  ⛔" : ""), x, y + r + 5);
  }
}

function drawPallets(s) {
  const u = TF.s;
  for (const [id, p] of Object.entries(s.pallets || {})) {
    if (p.carried_by) continue;
    const x = sx(p.x), y = sy(p.y), r = 0.55 * u;
    const col = p.delivered ? NV : AMBER;
    // shadow
    roundRect(x - r + 2, y - r + 3, r * 2, r * 2, 4); ctx.fillStyle = "rgba(0,0,0,.35)"; ctx.fill();
    // body
    roundRect(x - r, y - r, r * 2, r * 2, 4);
    const g = ctx.createLinearGradient(x, y - r, x, y + r);
    g.addColorStop(0, hexA(col, 1)); g.addColorStop(1, hexA(col, .78));
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.lineWidth = 1; ctx.stroke();
    // slats
    ctx.strokeStyle = "rgba(0,0,0,.22)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y - r + 2); ctx.lineTo(x, y + r - 2); ctx.stroke();
    ctx.fillStyle = col;
    ctx.font = "700 9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(id.replace("WH_Palette_", "P"), x, y - r - 4);
  }
}

function drawRoutes(s, now) {
  const nodes = s.graph.nodes;
  const dash = -(now / 42) % 1000;
  for (const [name, fk] of Object.entries(s.forklifts || {})) {
    const route = fk.route || []; if (!route.length) continue;
    const v = view[name] || fk;
    const col = (PHASE[fk.phase] || PHASE.idle).c;
    const pts = [[sx(v.x), sy(v.y)]];
    for (const n of route) if (nodes[n]) pts.push([sx(nodes[n][0]), sy(nodes[n][1])]);
    const dim = selected && selected !== name;
    // glow
    ctx.strokeStyle = hexA(col, dim ? .05 : .16); ctx.lineWidth = 6; ctx.setLineDash([]);
    stroke(pts);
    // marching ants
    ctx.save();
    ctx.strokeStyle = hexA(col, dim ? .35 : .95); ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 6]); ctx.lineDashOffset = dash;
    stroke(pts); ctx.restore();
    // destination marker
    const d = pts[pts.length - 1];
    ctx.strokeStyle = hexA(col, dim ? .4 : 1); ctx.lineWidth = 1.6; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(d[0] - 4, d[1] - 4); ctx.lineTo(d[0] + 4, d[1] + 4);
    ctx.moveTo(d[0] + 4, d[1] - 4); ctx.lineTo(d[0] - 4, d[1] + 4); ctx.stroke();
  }
  ctx.setLineDash([]);
}

// cuOpt/CBS plan — the deconflicted node paths CBS produced, drawn faintly over the
// roadmap so the operator can see the optimiser's intent behind the live routes.
function drawPlanPaths(s, now) {
  if (!plan || !plan.cbs || !plan.cbs.paths) return;
  const nodes = (s.graph && s.graph.nodes) || {};
  const dash = (now / 60) % 1000;
  ctx.save();
  ctx.lineCap = "round";
  for (const [name, path] of Object.entries(plan.cbs.paths)) {
    if (!path || path.length < 2) continue;
    const dim = selected && selected !== name;
    const pts = [];
    for (const n of path) if (nodes[n]) pts.push([sx(nodes[n][0]), sy(nodes[n][1])]);
    if (pts.length < 2) continue;
    ctx.strokeStyle = hexA(NV, dim ? .10 : .32);
    ctx.lineWidth = 2.4; ctx.setLineDash([2, 7]); ctx.lineDashOffset = dash;
    stroke(pts);
    // waypoint pips
    ctx.setLineDash([]);
    for (const p of pts) {
      ctx.fillStyle = hexA(NV, dim ? .15 : .5);
      ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, 7); ctx.fill();
    }
  }
  ctx.restore();
  ctx.setLineDash([]);
}

// Live CBS conflict indicator — pulsing link between forklifts at collision risk.
function drawConflicts(s, now) {
  const conflicts = s.conflicts || [];
  if (!conflicts.length) return;
  const pulse = 0.5 + 0.5 * Math.sin(now / 160);
  ctx.save();
  for (const c of conflicts) {
    const va = view[c.a], vb = view[c.b];
    if (!va || !vb) continue;
    const ax = sx(va.x), ay = sy(va.y), bx = sx(vb.x), by = sy(vb.y);
    const hi = c.severity === "high";
    // risk link
    ctx.strokeStyle = hexA(RED, (hi ? .55 : .32) + .35 * pulse);
    ctx.lineWidth = hi ? 2.4 : 1.6; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    // risk rings around each truck
    ctx.setLineDash([]);
    for (const [px, py] of [[ax, ay], [bx, by]]) {
      const r = (0.9 + 0.5 * pulse) * TF.s;
      ctx.strokeStyle = hexA(RED, .4 + .4 * pulse); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.stroke();
    }
    // caution glyph at midpoint
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    ctx.fillStyle = hexA(RED, .9 + .1 * pulse);
    ctx.font = "700 13px 'JetBrains Mono', monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("\u26A0", mx, my - 10 * TF.s * 0 - 12);
  }
  ctx.restore();
  ctx.setLineDash([]);
}

// Optional plan summary panel (renders only if #planPanel exists in the DOM).
function renderPlanPanel() {
  const el = document.getElementById("planBody");
  if (!el) return;
  if (!plan || !plan.assignments) {
    el.innerHTML = '<div class="plan-empty">No active plan. Ask the agent to optimise the fleet.</div>';
    return;
  }
  const cbs = plan.cbs || {};
  const rows = Object.entries(plan.assignments).map(([fk, tasks]) => {
    const t = (tasks || []).map(s => s.replace("WH_Palette_", "P").replace("\u2192", " \u2192 ")).join(", ");
    return `<div class="plan-row"><span class="plan-fk">${fk}</span><span class="plan-task">${t || "—"}</span></div>`;
  }).join("");
  el.innerHTML =
    `<div class="plan-head">solver <b>${plan.solver}</b> · cost <b>${Math.round(plan.total_cost)}</b>` +
    ` · CBS <b>${cbs.conflicts_found || 0}</b> conflict(s) ${cbs.resolved ? "resolved" : "pending"}</div>` +
    rows;
}

function drawTrails(s) {
  for (const [name, v] of Object.entries(view)) {
    const fk = (s.forklifts || {})[name]; if (!fk) continue;
    const col = (PHASE[fk.phase] || PHASE.idle).c;
    const dim = selected && selected !== name;
    for (let i = 1; i < v.trail.length; i++) {
      const a = v.trail[i - 1], b = v.trail[i];
      const t = i / v.trail.length;
      ctx.strokeStyle = hexA(col, (dim ? .18 : .5) * t);
      ctx.lineWidth = 1 + 4 * t; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(sx(a.x), sy(a.y)); ctx.lineTo(sx(b.x), sy(b.y)); ctx.stroke();
    }
  }
}

function drawForklifts(s, now) {
  const u = TF.s;
  for (const [name, fk] of Object.entries(s.forklifts || {})) {
    const v = view[name] || { x: fk.x, y: fk.y, yaw: fk.yaw };
    const x = sx(v.x), y = sy(v.y), yaw = -v.yaw;    // screen yaw (y flipped)
    const col = (PHASE[fk.phase] || PHASE.idle).c;
    const sel = selected === name;
    const dim = selected && !sel;
    ctx.globalAlpha = dim ? 0.5 : 1;

    // glow halo
    const halo = ctx.createRadialGradient(x, y, 0, x, y, 1.5 * u);
    halo.addColorStop(0, hexA(col, .38)); halo.addColorStop(1, hexA(col, 0));
    ctx.fillStyle = halo;
    ctx.beginPath(); ctx.arc(x, y, 1.5 * u, 0, 7); ctx.fill();

    if (sel) {
      ctx.strokeStyle = hexA(col, .9); ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
      ctx.lineDashOffset = now / 60;
      ctx.beginPath(); ctx.arc(x, y, 1.15 * u, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    }

    // directional wedge body
    const L = 0.95 * u, Wd = 0.72 * u;
    const c = Math.cos(yaw), si = Math.sin(yaw);
    const tip = [x + c * L, y + si * L];
    const bl = [x - c * L * 0.65 - si * Wd, y - si * L * 0.65 + c * Wd];
    const br = [x - c * L * 0.65 + si * Wd, y - si * L * 0.65 - c * Wd];
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]); ctx.lineTo(bl[0], bl[1]);
    ctx.lineTo(x - c * L * 0.25, y - si * L * 0.25); ctx.lineTo(br[0], br[1]);
    ctx.closePath();
    const bg = ctx.createLinearGradient(bl[0], bl[1], tip[0], tip[1]);
    bg.addColorStop(0, hexA(col, .8)); bg.addColorStop(1, hexA(col, 1));
    ctx.fillStyle = bg; ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 1.4; ctx.stroke();

    // carried pallet on the forks
    if (fk.carrying) {
      const px = x + c * L * 1.15, py = y + si * L * 1.15, pr = 0.42 * u;
      roundRect(px - pr, py - pr, pr * 2, pr * 2, 3);
      ctx.fillStyle = NV; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 1; ctx.stroke();
    }

    // name badge
    ctx.globalAlpha = 1;
    const label = name.replace("forklift", "F");
    ctx.font = "700 11px 'Inter', sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const bw = ctx.measureText(label).width + 12;
    roundRect(x - bw / 2, y + 1.15 * u, bw, 16, 5);
    ctx.fillStyle = hexA(col, dim ? .6 : .95); ctx.fill();
    ctx.fillStyle = "#06090c"; ctx.fillText(label, x, y + 1.15 * u + 8.5);
  }
  ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// Roster
// ═══════════════════════════════════════════════════════════════════════════
function rebuildRoster() {
  if (!snap) return;
  const names = Object.keys(snap.forklifts || {});
  rosterEl.innerHTML = "";
  document.getElementById("rosterCount").textContent = names.length;
  for (const n of names) rosterEl.appendChild(makeRosterCard(n));
}

function makeRosterCard(name) {
  const el = document.createElement("div");
  el.className = "rcard";
  el.innerHTML = `
    <div class="rc-top">
      <div class="rc-avatar">${name.replace("forklift", "F")}</div>
      <div class="rc-name">${name.replace("forklift", "Forklift ")}</div>
      <div class="rc-pill">Idle</div>
    </div>
    <div class="rc-task">Standing by</div>
    <div class="rc-prog"><span></span></div>
    <div class="rc-metrics">
      <div class="rc-metric"><div class="m-val v-spd">0.0</div><div class="m-lab">m/s</div></div>
      <div class="rc-metric"><div class="m-val v-lift">0.0</div><div class="m-lab">lift</div></div>
      <div class="rc-metric"><div class="m-val v-yaw">0°</div><div class="m-lab">yaw</div></div>
      <div class="rc-metric"><div class="m-val v-prox">—</div><div class="m-lab">prox</div></div>
    </div>`;
  el.addEventListener("click", () => {
    selected = (selected === name) ? null : name;
    for (const [nm, r] of Object.entries(rcards)) r.el.classList.toggle("selected", nm === selected);
  });
  rcards[name] = {
    el,
    pill: el.querySelector(".rc-pill"),
    task: el.querySelector(".rc-task"),
    prog: el.querySelector(".rc-prog > span"),
    spd: el.querySelector(".v-spd"), lift: el.querySelector(".v-lift"),
    yaw: el.querySelector(".v-yaw"), prox: el.querySelector(".v-prox"),
    avatar: el.querySelector(".rc-avatar"),
  };
  return el;
}

function updateRosterData(s) {
  for (const [name, fk] of Object.entries(s.forklifts || {})) {
    const r = rcards[name]; if (!r) continue;
    const ph = PHASE[fk.phase] || PHASE.idle;
    r.el.style.setProperty("--phase", ph.c);
    r.pill.textContent = ph.label;
    r.task.innerHTML = taskText(fk);
    r.prog.style.width = routeProgress(name, fk) + "%";
    r.spd.textContent = (fk.speed || 0).toFixed(1);
    r.lift.textContent = (fk.lift_height || 0).toFixed(2);
    r.yaw.textContent = Math.round(deg(fk.yaw)) + "°";
    const det = fk.object_detected;
    r.prox.textContent = det ? (fk.object_distance || 0).toFixed(1) + "m" : "clear";
    r.prox.style.color = det ? AMBER : "";
  }
}

function taskText(fk) {
  const tgt = (fk.target || "").replace("WH_Palette_", "P").replace("stage_", "Stage ");
  if (fk.phase === "carrying" || fk.phase === "dropping")
    return `Delivering <b>${(fk.carrying || "").replace("WH_Palette_", "P")}</b> → <b>${tgt || "?"}</b>`;
  if (fk.phase === "navigating") return `En route to <b>${tgt || "?"}</b>`;
  if (fk.phase === "picking" || fk.phase === "lifting") return `Picking <b>${tgt || "?"}</b>`;
  if (fk.phase === "returning") return `Returning to home`;
  return "Standing by";
}

const routeSeen = {};
function routeProgress(name, fk) {
  const len = (fk.route || []).length;
  if (fk.phase === "idle") { routeSeen[name] = 0; return 0; }
  routeSeen[name] = Math.max(routeSeen[name] || 0, len);
  const max = routeSeen[name] || 1;
  return Math.round(100 * (1 - len / (max + 1)));
}

// ═══════════════════════════════════════════════════════════════════════════
// KPIs (count-up) + sparklines
// ═══════════════════════════════════════════════════════════════════════════
function updateKpiTargets(s) {
  const pals = Object.values(s.pallets || {});
  const delivered = pals.filter(p => p.delivered).length;
  const transit = pals.filter(p => p.carried_by).length;
  const waiting = pals.length - delivered - transit;
  const fks = Object.values(s.forklifts || {});
  const active = fks.filter(f => f.phase !== "idle").length;
  const throughput = pals.length ? Math.round(100 * delivered / pals.length) : 0;
  Object.assign(kpiTarget, { delivered, transit, waiting, active, throughput, sim: s.t || 0 });

  if (Math.floor(s.t) !== lastSparkT) {
    lastSparkT = Math.floor(s.t);
    spark.delivered.push(delivered); spark.throughput.push(throughput);
    if (spark.delivered.length > 60) spark.delivered.shift();
    if (spark.throughput.length > 60) spark.throughput.shift();
    drawSpark("sparkDelivered", spark.delivered, NV);
    drawSpark("sparkThroughput", spark.throughput, NV);
  }
}

function animateKpis(dt) {
  const k = 1 - Math.exp(-dt / 0.25);
  const set = (id, key, fmt) => {
    kpiShown[key] += (kpiTarget[key] - kpiShown[key]) * k;
    document.getElementById(id).childNodes[0].nodeValue = fmt(kpiShown[key]);
  };
  set("kpiDelivered", "delivered", v => Math.round(v));
  set("kpiTransit", "transit", v => Math.round(v));
  set("kpiWaiting", "waiting", v => Math.round(v));
  set("kpiActive", "active", v => Math.round(v));
  set("kpiThroughput", "throughput", v => Math.round(v));
  set("kpiSim", "sim", v => Math.round(v));
}

function drawSpark(id, data, col) {
  const cv = document.getElementById(id); if (!cv || data.length < 2) return;
  const c = cv.getContext("2d"), w = cv.width, h = cv.height;
  c.clearRect(0, 0, w, h);
  const max = Math.max(1, ...data), min = Math.min(...data);
  const rng = Math.max(1, max - min);
  const X = i => (i / (data.length - 1)) * w;
  const Y = v => h - 3 - ((v - min) / rng) * (h - 6);
  c.beginPath();
  data.forEach((v, i) => i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)));
  c.lineTo(w, h); c.lineTo(0, h); c.closePath();
  const g = c.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, hexA(col, .35)); g.addColorStop(1, hexA(col, 0));
  c.fillStyle = g; c.fill();
  c.beginPath();
  data.forEach((v, i) => i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)));
  c.strokeStyle = col; c.lineWidth = 1.5; c.stroke();
}

// ═══════════════════════════════════════════════════════════════════════════
// Activity feed (client-side event diffing)
// ═══════════════════════════════════════════════════════════════════════════
function detectEvents(s) {
  const delivered = new Set(
    Object.entries(s.pallets || {}).filter(([, p]) => p.delivered).map(([id]) => id));
  if (!seededEvents) {
    prevDelivered = delivered;
    for (const [n, fk] of Object.entries(s.forklifts || {})) prevPhase[n] = fk.phase;
    seededEvents = true;
    logEvent(NV, `<b>Fleet online</b> · ${Object.keys(s.forklifts || {}).length} forklifts ready`);
    return;
  }
  for (const [name, fk] of Object.entries(s.forklifts || {})) {
    const was = prevPhase[name], is = fk.phase;
    if (was !== is) {
      const F = name.replace("forklift", "F");
      const col = (PHASE[is] || PHASE.idle).c;
      if (is === "navigating") logEvent(col, `<b>${F}</b> dispatched → ${(fk.target || "").replace("WH_Palette_", "P").replace("stage_", "Stage ")}`);
      else if (is === "picking") logEvent(col, `<b>${F}</b> picking ${(fk.target || "").replace("WH_Palette_", "P")}`);
      else if (is === "carrying") logEvent(col, `<b>${F}</b> lifted load, carrying`);
      else if (is === "returning") logEvent(col, `<b>${F}</b> returning home`);
      else if (is === "idle" && was === "returning") logEvent(PHASE.idle.c, `<b>${F}</b> parked at home`);
      prevPhase[name] = is;
    }
  }
  for (const id of delivered) {
    if (!prevDelivered.has(id))
      logEvent(NV, `<b>${id.replace("WH_Palette_", "Pallet ")}</b> delivered to staging ✓`);
  }
  // zone blocks
  for (const [zid, z] of Object.entries(s.zones || {})) {
    const key = "zone_" + zid;
    if (z.blocked && !detectEvents[key]) { logEvent(RED, `<b>${zid.toUpperCase()}</b> blocked — incident flagged`); detectEvents[key] = true; }
    if (!z.blocked) detectEvents[key] = false;
  }
  prevDelivered = delivered;
}

function logEvent(col, html) {
  const ev = document.createElement("div");
  ev.className = "ev";
  ev.innerHTML = `<span class="ev-time">${clockStr()}</span>` +
                 `<span class="ev-dot" style="background:${col}"></span>` +
                 `<span class="ev-txt">${html}</span>`;
  activityLog.prepend(ev);
  while (activityLog.children.length > 40) activityLog.lastChild.remove();
}

// ═══════════════════════════════════════════════════════════════════════════
// Chat
// ═══════════════════════════════════════════════════════════════════════════
function initChat() {
  const chips = document.getElementById("chips");
  EXAMPLES.forEach(x => {
    const c = document.createElement("div");
    c.className = "chip"; c.textContent = x;
    c.onclick = () => { document.getElementById("chatBox").value = x; sendChat(); };
    chips.appendChild(c);
  });
  addMsg("agent", "FleetMind online. Give me an order — e.g. “clear all the pallets off the racks into staging.”");
  document.getElementById("chatForm").addEventListener("submit", e => { e.preventDefault(); sendChat(); });
}

function addMsg(who, text, cls = "") {
  const m = document.createElement("div");
  m.className = `msg ${who} ${cls}`.trim();
  m.textContent = text;
  chatLog.appendChild(m);
  chatLog.scrollTop = chatLog.scrollHeight;
  return m;
}

async function sendChat() {
  const box = document.getElementById("chatBox");
  const msg = box.value.trim(); if (!msg) return;
  box.value = "";
  const btn = document.getElementById("chatSend"); btn.disabled = true;
  addMsg("user", msg);
  const thinking = addMsg("agent", "Coordinating fleet…", "think");
  try {
    const r = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    const j = await r.json();
    thinking.remove();
    addMsg("agent", j.reply || "(no response)");
  } catch (e) {
    thinking.remove();
    addMsg("agent", "(connection error — is the bridge running?)");
  } finally { btn.disabled = false; box.focus(); }
}

// ═══════════════════════════════════════════════════════════════════════════
// Legend + clock + helpers
// ═══════════════════════════════════════════════════════════════════════════
function buildLegend() {
  const el = document.getElementById("legend");
  const items = [
    ["Navigating", PHASE.navigating.c], ["Carrying", PHASE.carrying.c],
    ["Returning", PHASE.returning.c], ["Pallet", AMBER],
    ["Staging", NV], ["Blocked", RED],
  ];
  el.innerHTML = items.map(([l, c]) => `<span class="lg"><i style="background:${c}"></i>${l}</span>`).join("");
}

function updateClock() {
  document.getElementById("clock").textContent = clockStr();
}
function clockStr() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

// geometry / color helpers
function wrap(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }
function deg(r) { return r * 180 / Math.PI; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function stroke(pts) {
  ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// ═══════════════════════════════════════════════════════════════════════════
// Live 3D stream (Isaac Sim WebRTC) — vendored @nvidia/omniverse-webrtc lib.
// Isaac Sim allows only ONE WebRTC client, so we connect lazily on "Live 3D"
// and disconnect when the operator switches back to the 2D map.
// ═══════════════════════════════════════════════════════════════════════════
const STREAM = {
  // Signaling server = whichever host serves this console (the DGX). When opened
  // directly from a file/localhost, fall back to the DGX Tailscale address.
  host: (location.hostname && location.hostname !== "localhost" && location.hostname !== "127.0.0.1")
        ? location.hostname : "100.104.13.18",
  signalingPort: 49100,
  streamer: null,      // AppStreamer module (dynamic import)
  connecting: false,
  connected: false,
  requested: false,    // operator wants the stream visible
};

function setStreamMsg(msg, showRetry) {
  const m = document.getElementById("streamMsg");
  const r = document.getElementById("streamRetry");
  if (m) m.textContent = msg;
  if (r) r.hidden = !showRetry;
  const spin = document.querySelector(".stream-spinner");
  if (spin) spin.style.display = showRetry ? "none" : "";
}
function showStreamOverlay(show) {
  const ov = document.getElementById("streamOverlay");
  if (ov) ov.classList.toggle("hidden", !show);
}

async function startStream() {
  if (STREAM.connected || STREAM.connecting) return;
  STREAM.connecting = true;
  showStreamOverlay(true);
  setStreamMsg("Connecting to Isaac Sim stream…", false);
  try {
    if (!STREAM.streamer) {
      const mod = await import("./vendor/omniverse-webrtc-streaming-library.js");
      STREAM.streamer = mod.AppStreamer;
      STREAM.StreamType = mod.StreamType;
    }
    const streamConfig = {
      videoElementId: "remote-video",
      audioElementId: "remote-audio",
      authenticate: false,
      maxReconnects: 20,
      server: STREAM.host,
      signalingServer: STREAM.host,
      signalingPort: STREAM.signalingPort,
      mediaServer: STREAM.host,
      nativeTouchEvents: true,
      width: 1920, height: 1080, fps: 60,
      onUpdate: (m) => { /* status stream */ },
      onStart: (m) => {
        if (m && m.action === "start" && m.status === "success") {
          STREAM.connected = true; STREAM.connecting = false;
          setStreamMsg("Stream live", false);
          showStreamOverlay(false);
          const v = document.getElementById("remote-video");
          if (v) { v.muted = true; v.playsInline = true; v.play().catch(() => {}); }
        } else if (m && m.status === "error") {
          streamFailed(m.info || "stream error");
        }
      },
      onStop: () => { STREAM.connected = false; },
      onTerminate: () => { STREAM.connected = false; },
    };
    await STREAM.streamer.connect({
      streamSource: STREAM.StreamType.DIRECT,
      streamConfig,
    });
  } catch (e) {
    streamFailed(String(e && e.message ? e.message : e));
  }
}

function streamFailed(info) {
  STREAM.connecting = false; STREAM.connected = false;
  showStreamOverlay(true);
  setStreamMsg("Stream unavailable: " + info, true);
}

function stopStream() {
  try {
    if (STREAM.streamer && (STREAM.connected || STREAM.connecting)) {
      STREAM.streamer.stop();
      // The library's stop() leaves its internal stream handle set; without
      // clearing it a later connect() re-enters a half-torn-down session and the
      // <video> shows a frozen frame (needs a page refresh). Mirror NVIDIA's
      // web-viewer-sample, which nulls the private _stream after stop().
      try { STREAM.streamer._stream = null; } catch {}
    }
  } catch {}
  const v = document.getElementById("remote-video");
  if (v) { try { v.srcObject = null; } catch {} }
  STREAM.connected = false; STREAM.connecting = false;
}

function setStageView(v) {
  const stage = document.getElementById("stage");
  if (!stage) return;
  stage.dataset.view = v;
  document.getElementById("stageTitle").textContent =
    v === "stream" ? "Live 3D · Isaac Sim" : "Live warehouse map";
  document.querySelectorAll("#viewToggle .vt-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === v));
  if (v === "stream") {
    STREAM.requested = true;
    if (STREAM.connected) {
      // Already live — the element may have been auto-paused while hidden; resume.
      const vid = document.getElementById("remote-video");
      if (vid) vid.play().catch(() => {});
    } else {
      startStream();
    }
  } else {
    STREAM.requested = false;
    // Keep the WebRTC session ALIVE in the background. Isaac allows only ONE
    // client, and tearing the peer connection down then reconnecting on the next
    // toggle left the <video> frozen (required a refresh). Hiding via CSS keeps
    // the live MediaStream current, so switching back to Live 3D is instant and
    // always shows live forklift motion. The session is only released on unload.
  }
}

function initStageToggle() {
  document.querySelectorAll("#viewToggle .vt-btn").forEach((b) =>
    b.addEventListener("click", () => setStageView(b.dataset.view)));
  const retry = document.getElementById("streamRetry");
  if (retry) retry.addEventListener("click", () => { STREAM.connecting = false; startStream(); });
  // Release Isaac's single WebRTC client slot cleanly when the console closes.
  window.addEventListener("pagehide", stopStream);
}

// ═══════════════════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════════════════
buildLegend();
initChat();
initStageToggle();
health();
poll();
pollPlan();
setInterval(poll, POLL_MS);
setInterval(pollPlan, 1500);
setInterval(health, 4000);
requestAnimationFrame(frame);
