# FleetMind — Hackathon Q&A

Anticipated judge questions with concise, defensible answers. Grouped by theme so you can
jump to whatever a judge probes.

---

## 1. The pitch & problem

**Q: What does FleetMind do in one sentence?**
It lets a warehouse operator command a fleet of autonomous forklifts in plain English, and
turns that intent into an optimised, collision-free plan that runs live in an NVIDIA Isaac
Sim digital twin.

**Q: What problem are you solving?**
Warehouse fleet coordination today is either manual (radios, spreadsheets, human dispatch)
or locked inside expensive, rigid WMS software. We make it conversational and autonomous:
the operator states a goal, and the system handles assignment, routing, deconfliction, and
battery management automatically.

**Q: Who is the user?**
A warehouse floor supervisor or operations manager — someone who knows *what* they want
("clear the receiving dock", "move pallet 3 to staging") but shouldn't have to micro-manage
*which* truck goes *where* in *what order*.

**Q: Why is this hard / why hasn't it been done?**
Three hard problems have to work together in real time: (1) multi-vehicle task assignment
(an NP-hard VRP), (2) guaranteed collision-free multi-agent paths, and (3) natural-language
understanding — all closing the loop against live physics at interactive latency.

---

## 2. Architecture & how it works

**Q: Walk me through what happens when someone types a command.**
Operator text → the LLM agent picks a tool → the tool takes a live snapshot → **Roadmap**
builds the nav graph → **cuOpt** assigns pallets to trucks and orders the tasks → **CBS**
deconflicts the paths → the mission is dispatched over the Fleet Bus → the scene controller
drives the forklifts at physics rate → telemetry flows back out to `/state` → the UI renders
it at 8 Hz. It's a clean one-directional pipeline with a telemetry return path.

**Q: How do the three planning pieces fit together?**
They run in sequence over a shared map. The **Roadmap** is the waypoint graph. **cuOpt**
decides *who does what, in what order* (using the roadmap for distances). We translate that
assignment into start/goal graph nodes, then **CBS** computes *collision-free timed paths*
on the same roadmap. Roadmap → cuOpt → CBS → dispatch.

**Q: What's the "Fleet Bus"?**
An in-process, thread-safe singleton that decouples the planning/HTTP world from the
physics/scene world — commands flow one way, telemetry the other. It's the seam that lets us
swap simulators without touching anything above it.

**Q: Is the planning in the simulator or separate?**
Separate. All planning lives in the agent's tools layer. The simulator backends only
*execute* pre-planned missions. That separation is why the same brain runs against both the
mock and Isaac Sim.

---

## 3. NVIDIA tech / the "why NVIDIA" question

**Q: Which NVIDIA technologies are you using and why?**
- **Isaac Sim** — the photorealistic physics digital twin the fleet actually runs in.
- **cuOpt** — GPU-accelerated route/assignment optimization (the VRP solver).
- **NIM** — local LLM inference (Llama 3.1) for the natural-language agent.
- **GB10 / DGX-class hardware** — runs the whole stack (sim + solver + LLM) on one machine.

**Q: Why cuOpt instead of a plain solver?**
Vehicle routing is NP-hard; cuOpt solves large instances on the GPU in the ~2 s budget we
need for interactive use. It also scales far past our 2-truck demo to real fleet sizes
without a re-architecture. We keep a local greedy+2-opt fallback so the demo never dies if
the cuOpt server is unavailable.

**Q: Are you really running Isaac Sim or is it a mock?**
Both are real and interchangeable. The live demo runs the actual Isaac Sim scene with PhysX;
the mock is a kinematic twin with an identical interface we use off-GPU for development and
as a safety net.

---

## 4. The algorithms (deep-dive judges)

**Q: How does the roadmap decide where the nodes go?**
It infers the floor bounding box from the positions of all forklifts, pallets, and zones
(plus a margin), lays a uniform `nx × ny` grid over it, and punches out cells near pallets
and loading zones so routes arc around obstacles. It's a 4-connected grid, so paths are
clean Manhattan-style runs. Nothing about the floor size is hardcoded.

**Q: What optimization does cuOpt actually solve?**
A capacitated pickup-and-delivery VRP: assign each pallet-to-zone job to a forklift and
order each truck's tasks to minimise total travel cost. It's battery-aware — trucks below the
low-battery threshold are excluded when charged trucks can cover the work.

**Q: How does CBS guarantee no collisions?**
Conflict-Based Search: a high-level best-first search over a constraint tree, with a
space-time A\* at the low level. When two trucks would occupy the same node at the same time
tick (or swap across an edge), it adds a constraint forbidding it for one agent and replans.
Idle trucks are pinned as stationary obstacles so moving trucks route around them.

**Q: What if CBS can't resolve everything in time?**
We stagger releases — the motion stays proactively safe — and there's a second, independent
*reactive* avoidance layer inside the scene controller that brakes or side-steps every
physics step as conflicts develop mid-route (because both trucks keep moving after the static
plan was made).

**Q: Two layers of collision avoidance — why?**
CBS is planning-time and global but based on a snapshot; the reactive layer is step-time and
local. Together they cover both "plan a safe route" and "react if reality drifts from the
plan."

---

## 5. The LLM agent

**Q: How does natural language become an action?**
The agent exposes a set of tools (move_pallet, optimize_and_dispatch, block_zone, etc.) as
function schemas. NIM does tool-calling: it reads the command and returns which tool to call
with what arguments. If NIM is unreachable, a regex-based offline intent parser maps common
phrases to the same tools — so the demo works even with no LLM.

**Q: What if the LLM hallucinates or picks a bad action?**
Tools validate their inputs against the live world state, and every fleet-moving command
still goes through cuOpt + CBS — the LLM can't bypass the optimizer or the safety layer. The
worst case is a no-op with an error message, not an unsafe move.

**Q: Why a local LLM (NIM) instead of a cloud API?**
Latency, privacy, and offline operation — a warehouse shouldn't depend on an internet round
trip to dispatch a forklift. It all runs on the local GB10 box.

---

## 6. Battery, safety, incidents

**Q: How does the battery model work?**
Each truck drains charge proportional to distance driven and recharges while parked near a
charger. The scene is the source of truth; the value is published in `/state` and read by
cuOpt to bound range and prefer fuller trucks. Low trucks are routed home to charge while a
fuller truck takes the job.

**Q: What happens on an incident (spill / blocked aisle)?**
The operator can block a zone ("there's a spill in bay 2"); the zone is flagged, the fleet
re-routes around it, and the UI shows a hazard overlay. The planner treats it as an obstacle
on the next plan.

---

## 7. Scale & generalisation

**Q: You showed 2 forklifts — does this scale?**
Nothing in the architecture is hardcoded to two. cuOpt is built for large fleets, CBS handles
many agents, and the roadmap/UI derive their size from the entities present. The demo is
small for clarity, not capability.

**Q: How hard is it to move to a different warehouse?**
For a same-size facility it's a constants edit (forklift/pallet/zone/charger positions);
bounds, the nav-grid, and the UI scale all re-derive automatically. A physically different
building additionally needs a new warehouse USD model and a camera reframe. No algorithm
changes.

**Q: Real robots instead of simulation?**
The backend is swappable behind a fixed interface. Replacing the sim backend with a ROS 2
bridge to real AMRs is the intended path — the planning, agent, and UI stay identical. We
already have a `ros2` module scaffold for exactly this.

---

## 8. Business / ROI

**Q: What's the ROI story?**
Warehouse automation typically targets labour reduction, higher throughput (picks/hour), and
fewer forklift incidents (a major cost centre). Optimised routing cuts travel distance and
fleet idle time; conversational control cuts operator training and dispatch overhead. (We
have a separate ROI research brief with cited figures.)

**Q: Who would buy this / what's the market?**
3PLs, distribution centres, and manufacturers running AMR or mixed fleets — anyone with
enough vehicles that manual coordination becomes the bottleneck.

**Q: What's your differentiation vs an existing WMS?**
Conversational control, GPU-optimised planning that re-solves in real time, and a physics
digital twin for safe "what-if" testing before touching real hardware — in one integrated,
locally-hosted stack.

---

## 9. Process / what you built

**Q: What was the hardest part?**
Making three real-time systems (assignment, deconfliction, physics) cooperate at interactive
latency, and keeping the pallet physics stable (we carry pallets kinematically on the forks
and flip them to dynamic on drop to avoid PhysX contact storms freezing the fleet).

**Q: What would you do with more time?**
Wire battery all the way into cuOpt range constraints, connect the ROS 2 bridge to real
hardware, add charging-station scheduling, and refactor the duplicated layout constants into
one shared config module.

**Q: What are the current limitations?**
The layout constants are duplicated in two files and must be kept in sync; the demo uses a
small fixed fleet; and battery-to-cuOpt wiring is partial. None are architectural — they're
finish-work.

---

## 10. Quick rapid-fire facts

| Question | Answer |
|---|---|
| Language / stack | Python, FastAPI, standard-library planning, HTML/JS canvas UI |
| Optimizer | NVIDIA cuOpt (local greedy + 2-opt fallback) |
| Deconfliction | Conflict-Based Search (space-time A\*) |
| LLM | NVIDIA NIM, Llama 3.1, with an offline regex fallback |
| Simulator | NVIDIA Isaac Sim (PhysX) + a kinematic mock twin |
| State API | FastAPI on port 8080; UI polls `/state` at 8 Hz |
| Runs on | A single NVIDIA GB10 / DGX-class machine |
| External dependencies at demo time | None required — offline fallbacks for both LLM and optimizer |
