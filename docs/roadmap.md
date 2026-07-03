# Roadmap — waypoint graph construction

`src/agent/planning/roadmap.py`

The `Roadmap` class builds the waypoint graph that the CBS deconfliction solver and the A\*
router reason over. It is **backend-agnostic**: it builds itself from a plain `/state`
snapshot, so it works identically against the offline mock and the real Isaac Sim scene.

---

## Decision tree — which path is taken

`Roadmap.from_snapshot()` inspects the snapshot and picks one of three construction modes:

```
snapshot already has "graph.nodes"?  ──yes──▶  adopt it verbatim
                │ no
    grid=(nx, ny) passed as argument?  ──yes──▶  _uniform_grid()   ← live Isaac scene
                │ no
                └────────────────────────────▶  _synthesise_grid() ← cell-sized fallback
```

---

## Mode 1 — adopt an existing graph

If the `/state` snapshot already contains `graph.nodes`, the roadmap uses those coordinates
directly, dropping any node whose id starts with `rack` (those are obstacle slabs rendered
in the UI; planned paths must never route through them).

This path is taken by the **offline mock** (`WarehouseSim`), which ships its own
hand-crafted 20-node graph in every snapshot.

---

## Mode 2 — uniform grid (the live Isaac Sim scene)

When called with `grid=(20, 20)` (the value used by `IsaacNavBackend`), the roadmap
synthesises a fixed $n_x \times n_y$ mesh over the warehouse floor. Construction has two
steps.

### Step A — compute the floor bounding box

`_floor_bounds()` collects the $(x, y)$ position of **every forklift, pallet, and zone**
from the snapshot and derives the floor extent from that data:

$$
\bigl(\min x - m,\; \max x + m,\; \min y - m,\; \max y + m\bigr)
\quad m = \text{margin (default 2.0 m)}
$$

Nothing about the floor size is hardcoded — the bounds are inferred from wherever the
entities actually are, plus a safety border. If the snapshot is empty the origin is used as
a fallback.

### Step B — lay down the $n_x \times n_y$ mesh

The bounding box is divided into evenly spaced columns and rows:

$$
\Delta x = \frac{\max x - \min x}{n_x - 1} \qquad \Delta y = \frac{\max y - \min y}{n_y - 1}
$$

Each candidate cell $(c, r)$ is placed at:

$$
x = \min x + c \cdot \Delta x \qquad y = \min y + r \cdot \Delta y
$$

A cell is **skipped** (punched out of the graph) if it falls within:

| Clearance | Default | Reason |
|---|---|---|
| `pallet_clear` of a resting pallet | 1.4 m | Routes arc around loaded rack faces instead of grazing them |
| `zone_clear` of a staging zone | 1.3 m | Through-traffic never cuts across a loading/receiving area |

Only pallets that are **not delivered and not currently carried** count as obstacles.

### Edges — 4-connected grid

Each surviving node is linked to its right `(+1, 0)` and up `(0, +1)` neighbours
(undirected). Movement is therefore **Manhattan-style** — no diagonals. Combined with
`astar_straight()` (which adds a turn-penalty), trucks produce clean L-shaped trajectories
that are natural for a forklift.

```
┌───┬───┬───┬───┐
│   │   │   │   │  each surviving node connects
├───┼───┼───┼───┤  to its right and upper neighbour
│   │   │ × │   │  × = punched out (near pallet)
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘
```

---

## Mode 3 — cell-sized fallback grid

`_synthesise_grid()` follows the same logic but uses a fixed **cell size** (default 2.0 m)
instead of a fixed node count. It blocks whole grid cells that contain a resting pallet.
This mode is the default when the snapshot has no graph and no `grid=(nx, ny)` argument is
supplied.

---

## Query methods

Once built, the roadmap exposes:

| Method | Purpose |
|---|---|
| `nearest(x, y)` | Snap a world coordinate to the closest node |
| `astar(start, goal, avoid)` | Shortest path; `avoid` blocks cells (e.g. a parked forklift) |
| `astar_straight(start, goal, turn_penalty, avoid)` | Shortest path with a turn-penalty so trucks prefer long straight runs over zig-zags |
| `nodes_within(x, y, radius)` | All nodes inside a circle — used to mark a forklift's footprint as blocked |
| `collapse_collinear(pts)` | Collapse redundant interior points on a straight run into a single segment |
| `path_length(path)` | Sum of Euclidean edge lengths along a node sequence |

---

## Key constants (set by the caller, not the class)

| Constant | Where set | Value | Meaning |
|---|---|---|---|
| `grid` | `isaac_nav_bridge.py` | `(20, 20)` | Node count for the Isaac scene |
| `cell` | fallback default | `2.0 m` | Cell spacing for the synthesised grid |
| `margin` | both modes | `2.0 m` | Border added around the floor bounding box |
| `pallet_clear` | both modes | `1.4 m` | Clearance radius around resting pallets |
| `zone_clear` | both modes | `1.3 m` | Clearance radius around staging zones |
