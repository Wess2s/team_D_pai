"""
FleetMind — warehouse scene + BUS-DRIVEN autonomy, loaded INTO the running Isaac Sim
streaming app via Kit's `--exec` hook (runheadless.sh owns the render loop).

This is the production source of truth for the deployed scene. It:

  build()          -> new stage, warehouse, physics + ground, 2 rigged forklifts,
                      6 pallets + payloads, and 3 staging-zone floor markers.
  start_autonomy() -> play the timeline + register a physics-step callback that drives
                      each forklift along the waypoint route the fleet bus hands it,
                      performs pick / drop (raise/lower fork + carry the pallet prim),
                      and streams pose / phase / carried-pallet telemetry back onto the bus.
  start_bridge()   -> run the FastAPI control bridge (SIM_BACKEND=isaac) in a daemon thread
                      IN THIS SAME PROCESS, so its IsaacNavBackend shares the bus singleton
                      with the controller. The agent + UI talk to :8080 exactly as they do
                      against the mock.

The forklift is a REAR-wheel drive + REAR-wheel steer vehicle. The controller empirically
calibrates the drive sign and steer sign at runtime (once per forklift), then closes a
heading loop toward each waypoint. Joint conventions (verified): back_wheel_drive negative
= forward, back_wheel_swivel positive = left, lift_joint higher = up.
"""

import math
import os
import sys
import threading

import numpy as np
import omni.usd
import omni.timeline
import omni.physx
from pxr import (
    Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, PhysicsSchemaTools, PhysxSchema,
)
from isaacsim.core.utils.stage import add_reference_to_stage

try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:  # pragma: no cover
    from isaacsim.core.utils.nucleus import get_assets_root_path

try:
    from isaacsim.core.prims import SingleArticulation
except Exception:  # pragma: no cover
    from isaacsim.core.prims.single_articulation import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

try:
    import carb
    _log = carb.log_warn
except Exception:  # pragma: no cover
    _log = print

# Make the repo's `src/` importable inside Kit's Python (repo root = parent of scenes/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.sim.fleet_bus import bus  # noqa: E402

# ===========================================================================
# CONFIG  — must stay in sync with src/sim/isaac_nav_bridge.py
# ===========================================================================
FORKLIFTS = {                       # name -> (x, y, yaw_degrees)
    "AMR_1": (-6.0, -3.0,   0.0),
    "AMR_2": ( 6.0,  3.0, 180.0),
}
PALLETS = [                         # (x, y) pallet-grid positions -> Pallet_00..02
    (-3.0, -3.0), (-3.0, 3.0), (1.0, 5.0),
]
ZONES = {                           # staging bays (match isaac_nav_bridge.ZONES)
    "stage_1": (-6.0, 7.0),
    "stage_2": ( 0.0, 7.0),
    "stage_3": ( 6.0, 7.0),
}
CHARGERS = {                        # charging docks (each forklift's home = its charger)
    "charge_1": (-6.0, -3.0),
    "charge_2": ( 6.0,  3.0),
}
# Battery model: drains with travelled distance, recharges when a truck is parked
# on a charger dock. Values are advisory (the real telemetry has no battery).
BATTERY_DRAIN_PER_M = 0.6         # % consumed per metre driven
BATTERY_CHARGE_PER_STEP = 0.25    # % gained per physics step while docked
CHARGER_RADIUS = 1.6              # m — how close a truck must park to charge
PALLET_TOP_Z = 0.14
PAYLOAD_DROP = 0.06

# Cinematic overview camera — frames the whole floor (forklifts ±6, pallets ±3,
# staging at y=7). Sits INSIDE the warehouse footprint (eye within the ±6 x-range so
# it never clips the outer walls) at ~7 m — above the 3 m forklifts, below the roof —
# looking down-and-forward across the operating area.
CAMERA_PATH   = "/World/FleetCam"
CAMERA_EYE    = (4.0, -8.5, 7.0)
CAMERA_TARGET = (0.0, 2.0, 0.4)

FORKLIFT_SUBPATH = "/Isaac/Samples/Rigging/Forklift/forklift_b_rigged_cm.usd"
PALLET_SUBPATH = "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd"
SPAWN_Z = 0.03
GROUND_STATIC_FRICTION  = 0.9
GROUND_DYNAMIC_FRICTION = 0.8

# ---- Kinematic path-following controller ----------------------------------
# The forklift base is driven KINEMATICALLY: each physics step we set its world pose
# directly along the A* waypoint route (constant speed, yaw slewed toward the travel
# direction) and zero its velocity. This is deterministic and demo-safe — it removes
# the rear-wheel drive/steer physics, the runtime drive/steer sign calibration, the
# stall watchdog and the reverse-recovery that made the truck drive the wrong way,
# orbit the pallet, or ram it. The fork (lift_joint) is still actuated by its physics
# drive, and a carried pallet is glued to the forks (see _carry_follow).
KIN_SPEED     = 1.6                 # m/s base travel speed along the route
TURN_RATE     = math.radians(140.0) # rad/s max yaw slew toward travel heading
# The rigged model's visual forward (fork direction) is offset from the articulation
# root's reported yaw. We orient the truck by setting root_yaw = travel - MODEL_YAW_OFFSET
# so the forks point along travel (~ -90°: the asset points toward -Y at reported yaw 0).
# Kinematic control means this is fixed once and never drifts.
MODEL_YAW_OFFSET = math.radians(-90.0)

WAYPOINT_DIST = 0.6                 # m, advance to next waypoint within this
ARRIVE_DIST   = 0.30                # m, intermediate leg goal reached
PICK_ARRIVE   = 0.20                # m, tight final arrival for pick/drop
PICK_SNAP     = 1.60                # m, max dist to leg end for the fork to (dis)engage
# Forks reach this far ahead of the truck centre (travel/forward direction). It MUST
# match the bridge's APPROACH_LEN standoff so that, at pick arrival, the fork tip sits
# exactly over the pallet cell -> the load is engaged in place (no teleport onto the
# truck) and lifts smoothly. A carried pallet is glued to this fork tip.
FORK_REACH    = 1.4
LIFT_RAISE    = 0.35                # m visible fork travel (pallet clears the floor)
LIFT_RATE     = 0.01                # m/step fork travel
WARMUP_STEPS  = 5
SETTLE_STEPS  = 60                  # let the base settle on the ground, capture rest Z
ACT_STEPS     = 30                  # steps to hold during pick/drop lift
LOG_EVERY     = 120
# --- live inter-forklift collision avoidance (reactive, runs every step) ----------
# The bridge plans a static route around wherever the OTHER truck is at dispatch time,
# but both trucks move, so a crossing/head-on can still develop mid-route. These trucks
# therefore watch each other every physics step and yield reactively. Two behaviours:
#   * vs a MOVING higher-priority truck -> BRAKE in place and wait for it to pass, then
#     resume (the moving truck keeps its right of way and clears the area).
#   * vs a STATIONARY/parked/stopped truck -> DRIVE AROUND it (side-step perpendicular to
#     our travel) since it will never move for us, then rejoin the route once clear.
# Priority (right-of-way) goes to a loaded truck first, then to AMR_1 — a strict total
# order, so exactly one of any moving pair yields and they never deadlock.
YIELD_DIST    = 3.5                 # m, react when the other truck is this close
EVADE_SPEED   = 0.8                 # fraction of KIN_SPEED used for the side-step
YIELD_BEHIND  = -0.3                # cos threshold: ignore a truck clearly behind us
# ===========================================================================



_RT = {}   # runtime state (keeps physics-step subscription & controllers alive)
_BUS = bus()

# WH_Palette_0N (bus id) <-> /World/Pallets/Pallet_0(N-1) (USD prim)
_PALLET_PATH = {f"WH_Palette_{i + 1:02d}": f"/World/Pallets/Pallet_{i:02d}"
                for i in range(len(PALLETS))}


def _set_pose(stage, prim_path, xyz, yaw_deg=0.0, scale=1.0):
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xform.AddRotateZOp().Set(float(yaw_deg))
    if scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))


def _move_prim_xy(stage, prim_path, x, y, z):
    """Cheap translate-only pose update for a carried pallet (kinematic follow)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(x), float(y), float(z)))
            return
    xform.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))


def _disable_physics(stage, prim_path):
    """Strip collision + rigid-body physics from a prop subtree so it becomes a pure
    visual we place kinematically. The warehouse SM_PaletteA pallet ships heavy
    collision geometry; once a forklift's forks overlapped it, PhysX contact
    generation exploded (aggregate-pair overflow) and the sim slowed to a crawl —
    freezing the whole fleet. Our pick/carry/drop is entirely kinematic (we teleport
    the pallet prim), so the pallets need no physics at all."""
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            a = prim.GetAttribute("physics:collisionEnabled")
            if a:
                a.Set(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            a = prim.GetAttribute("physics:rigidBodyEnabled")
            if a:
                a.Set(False)


def _make_pallet_physics(stage, prim_path, forklift_paths):
    """Give a pallet REAL rigid-body physics + collision, but filter its collision
    against every forklift so the forks never generate contacts with it.

    That fork/pallet contact explosion (PhysX aggregate-pair overflow) is exactly what
    previously froze the whole fleet, so filtering the pair is what makes real pallet
    physics safe here. The pallet stays KINEMATIC while it rests on the rack and while it
    rides the forks (so it tracks the fork tip exactly, with no jitter or solver fighting
    against the kinematically-driven base); on drop it is switched to a DYNAMIC body
    (`_set_pallet_kinematic(..., False)`) so it settles onto the floor under gravity with
    no clipping. Collision vs the ground plane and other pallets stays live throughout.

    CRITICAL: the SM_PaletteA collision ships as a *triangle mesh*, and PhysX cannot make
    a triangle-mesh body dynamic ("dynamic meshes (without SDF) are not supported"). We
    therefore force every collider to a CONVEX HULL approximation, which is a valid
    dynamic collider — otherwise the kinematic→dynamic flip on drop is rejected and the
    pallet clips straight through the floor."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)          # placed/carried kinematically until drop
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(25.0)
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            a = p.GetAttribute("physics:collisionEnabled")
            if a:
                a.Set(True)
            # Convex hull so the body is a legal DYNAMIC collider (triangle meshes aren't).
            mc = UsdPhysics.MeshCollisionAPI.Apply(p)
            mc.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    # Filter pallet<->forklift contacts so the forks slide under the load without
    # generating the contact storm that overflowed PhysX and stalled the sim.
    fp = UsdPhysics.FilteredPairsAPI.Apply(prim)
    rel = fp.CreateFilteredPairsRel()
    for fpath in forklift_paths.values():
        rel.AddTarget(Sdf.Path(fpath))


def _set_pallet_kinematic(stage, prim_path, kinematic):
    """Flip a pallet between kinematic (placed/carried) and dynamic (dropped -> settles)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    a = prim.GetAttribute("physics:kinematicEnabled")
    if a:
        a.Set(bool(kinematic))
    else:
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(bool(kinematic))


def _bbox_size_x(stage, prim_path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        return 0.0
    return float(rng.GetMax()[0] - rng.GetMin()[0])


def _yaw_from_quat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _configure_drives(stage, base_path):
    """Override the rigged asset's shipped drive params (it ships a -200 rad/s wheel
    target from the rigging tutorial, which flings the forklift at spawn)."""
    drive_j = stage.GetPrimAtPath(f"{base_path}/back_wheel_joints/back_wheel_drive")
    steer_j = stage.GetPrimAtPath(f"{base_path}/back_wheel_joints/back_wheel_swivel")
    lift_j  = stage.GetPrimAtPath(f"{base_path}/lift_joint/lift_joint")

    if drive_j and drive_j.IsValid():
        d = UsdPhysics.DriveAPI.Get(drive_j, "angular") or \
            UsdPhysics.DriveAPI.Apply(drive_j, "angular")
        d.CreateStiffnessAttr(0.0)
        d.CreateDampingAttr(1.0e5)
        d.CreateTargetVelocityAttr(0.0)
        d.CreateMaxForceAttr(1.0e7)

    if steer_j and steer_j.IsValid():
        s = UsdPhysics.DriveAPI.Get(steer_j, "angular") or \
            UsdPhysics.DriveAPI.Apply(steer_j, "angular")
        s.CreateStiffnessAttr(1.0e6)
        s.CreateDampingAttr(1.0e5)
        s.CreateTargetPositionAttr(0.0)
        s.CreateMaxForceAttr(1.0e7)

    if lift_j and lift_j.IsValid():
        l = UsdPhysics.DriveAPI.Get(lift_j, "linear") or \
            UsdPhysics.DriveAPI.Apply(lift_j, "linear")
        l.CreateStiffnessAttr(1.0e6)
        l.CreateDampingAttr(1.0e5)
        l.CreateTargetPositionAttr(0.0)
        l.CreateMaxForceAttr(1.0e7)


def _zone_marker(stage, zid, x, y):
    """A thin coloured floor disc so staging bays are visible in the stream."""
    path = f"/World/Zones/{zid}"
    disc = UsdGeom.Cylinder.Define(stage, path)
    disc.CreateRadiusAttr(0.9)
    disc.CreateHeightAttr(0.02)
    disc.CreateAxisAttr("Z")
    disc.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.6, 0.2)])
    _set_pose(stage, path, (x, y, 0.01))


def _charger_marker(stage, cid, x, y):
    """A blue floor pad marking a charging dock (a forklift's home/charge node)."""
    path = f"/World/Chargers/{cid}"
    disc = UsdGeom.Cylinder.Define(stage, path)
    disc.CreateRadiusAttr(0.8)
    disc.CreateHeightAttr(0.02)
    disc.CreateAxisAttr("Z")
    disc.CreateDisplayColorAttr([Gf.Vec3f(0.10, 0.45, 1.0)])
    _set_pose(stage, path, (x, y, 0.008))


def _node_marker(stage, idx, x, y):
    """A small grey dot marking a waypoint graph node on the floor (route network)."""
    path = f"/World/Waypoints/Node_{idx:03d}"
    disc = UsdGeom.Cylinder.Define(stage, path)
    disc.CreateRadiusAttr(0.05)
    disc.CreateHeightAttr(0.012)
    disc.CreateAxisAttr("Z")
    disc.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.52, 0.65)])
    _set_pose(stage, path, (x, y, 0.005))


def _build_waypoint_markers(stage):
    """Draw the drivable routing network on the floor: a thin grey line for every graph
    edge (the "mesh") plus a small dot at each node. The bus graph is the 20×20 mesh the
    bridge publishes, so this shows exactly the network the planner routes over."""
    UsdGeom.Xform.Define(stage, "/World/Waypoints")
    graph = _BUS.graph or {}
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])

    # Edges as one BasisCurves prim: each edge is a 2-point linear segment.
    if edges:
        pts, counts = [], []
        for a, b in edges:
            if a in nodes and b in nodes:
                ax, ay = nodes[a]
                bx, by = nodes[b]
                pts.append(Gf.Vec3f(float(ax), float(ay), 0.006))
                pts.append(Gf.Vec3f(float(bx), float(by), 0.006))
                counts.append(2)
        net = UsdGeom.BasisCurves.Define(stage, "/World/Waypoints/_Network")
        net.CreateTypeAttr("linear")
        net.CreatePointsAttr(pts)
        net.CreateCurveVertexCountsAttr(counts)
        net.CreateWidthsAttr([0.02] * len(pts))
        net.SetWidthsInterpolation("vertex")
        net.CreateDisplayColorAttr([Gf.Vec3f(0.28, 0.34, 0.46)])

    for i, (nid, xy) in enumerate(sorted(nodes.items())):
        _node_marker(stage, i, float(xy[0]), float(xy[1]))
    _log(f"[FleetMind] Waypoint network: {len(nodes)} nodes, {len(edges)} edges drawn.")


def _make_route_overlay(stage, name):
    """A thin red poly-line on the floor showing a forklift's currently-planned route,
    updated every physics step from the truck's active leg waypoints.

    CRITICAL: the width is authored ONCE here with `constant` interpolation (a single
    value for the whole curve). The RTX real-time renderer tessellates curve width at
    creation and ignores later width edits, so a curve created empty/at default renders as
    a 1 m tube forever. Authoring a constant 0.04 m width up front — and only updating the
    POINTS at runtime — keeps it a thin ribbon."""
    UsdGeom.Xform.Define(stage, "/World/Routes")
    path = f"/World/Routes/{name}_route"
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    # Seed with a degenerate 2-point curve so width tessellation is established now.
    curve.CreatePointsAttr([Gf.Vec3f(0, 0, -5), Gf.Vec3f(0, 0.001, -5)])
    curve.CreateCurveVertexCountsAttr([2])
    w = curve.CreateWidthsAttr([0.04])
    w.SetMetadata("interpolation", "constant")
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.05, 0.05)])
    return path


def _update_route_overlay(stage, name, pts):
    """Set the red route line to `pts` (world XY list). Only POINTS change — the constant
    width authored in `_make_route_overlay` is never touched (see note there). An empty
    list hides the curve by collapsing it to a sub-floor degenerate segment."""
    prim = stage.GetPrimAtPath(f"/World/Routes/{name}_route")
    if not prim or not prim.IsValid():
        return
    curve = UsdGeom.BasisCurves(prim)
    if len(pts) < 2:
        curve.GetPointsAttr().Set([Gf.Vec3f(0, 0, -5), Gf.Vec3f(0, 0.001, -5)])
        curve.GetCurveVertexCountsAttr().Set([2])
        return
    vpts = [Gf.Vec3f(float(x), float(y), 0.04) for x, y in pts]
    curve.GetPointsAttr().Set(vpts)
    curve.GetCurveVertexCountsAttr().Set([len(vpts)])


def _look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera local-to-world transform for a camera at `eye` looking at `target`.
    USD cameras look down their local -Z with +Y up."""
    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    fwd = (target - eye).GetNormalized()
    z_axis = -fwd
    x_axis = Gf.Cross(Gf.Vec3d(*up), z_axis).GetNormalized()
    y_axis = Gf.Cross(z_axis, x_axis)
    return Gf.Matrix4d(
        x_axis[0], x_axis[1], x_axis[2], 0.0,
        y_axis[0], y_axis[1], y_axis[2], 0.0,
        z_axis[0], z_axis[1], z_axis[2], 0.0,
        eye[0],    eye[1],    eye[2],    1.0,
    )


def _setup_camera(stage):
    """A cinematic 3/4 overview camera framing the whole warehouse floor."""
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    cam.CreateFocalLengthAttr(18.0)                     # wide-ish so the floor fits
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    cam.CreateFocusDistanceAttr(22.0)
    xform = UsdGeom.Xformable(cam)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(_look_at_matrix(CAMERA_EYE, CAMERA_TARGET))
    _log(f"[FleetMind] FleetCam created at {CAMERA_EYE} -> {CAMERA_TARGET}.")


def _pov_cam_path(name):
    return f"/World/Cameras/{name}_POV"


def _make_follow_cam(stage, name):
    """A driver's-eye chase camera per forklift, updated every physics step to ride
    behind/above the truck looking forward over the forks. Kept as a top-level prim (not
    a child of the scaled forklift) so the forklift's cm→m scale never distorts it; the
    operator selects it from the viewport camera dropdown to watch that truck's POV."""
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    cam = UsdGeom.Camera.Define(stage, _pov_cam_path(name))
    cam.CreateFocalLengthAttr(20.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    cam.CreateFocusDistanceAttr(6.0)
    UsdGeom.Xformable(cam).AddTransformOp()
    _log(f"[FleetMind] Follow camera created for {name}.")


def _update_follow_cam(stage, name, x, y, yaw):
    """Place the forklift's chase camera: up and just behind the mast, looking forward
    and slightly down over the forks (fork direction = yaw + MODEL_YAW_OFFSET)."""
    prim = stage.GetPrimAtPath(_pov_cam_path(name))
    if not prim or not prim.IsValid():
        return
    fwd = yaw + MODEL_YAW_OFFSET
    fdx, fdy = math.cos(fwd), math.sin(fwd)
    eye = (x - fdx * 0.4, y - fdy * 0.4, 2.8)      # driver head height, behind the mast
    target = (x + fdx * 4.5, y + fdy * 4.5, 0.7)   # forward over the forks, toward the load
    mtx = _look_at_matrix(eye, target)
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(mtx)
            return
    xform.AddTransformOp().Set(mtx)


def _declutter_lights(stage):
    """Deactivate the ceiling structure hanging in the FleetCam's immediate foreground.

    The Simple_Warehouse roof carries beams, brackets and lamp fixtures; a ceiling beam
    (e.g. SM_BeamA_9M29 at ~(3,-9,8.8)) sits barely a metre in front of the overview
    camera eye (~(4,-8.5,7)) and renders as a big dark bar across the stream. We walk only
    the TOP-LEVEL warehouse parts (never their materials) and deactivate the ceiling-level
    ones sitting right in front of the lens, so the obstruction clears while the floor,
    walls and the rest of the lighting are untouched."""
    root = stage.GetPrimAtPath("/World/Warehouse")
    if not root or not root.IsValid():
        return
    cache = UsdGeom.XformCache()
    ex, ey = CAMERA_EYE[0], CAMERA_EYE[1]
    targets = ("SM_BeamA", "SM_BracketBeam", "SM_LampCeilingA", "SM_CeilingA")
    removed = []
    for prim in root.GetChildren():          # top-level parts only — never touch /Looks/
        name = prim.GetName()
        if not any(name.startswith(t) for t in targets):
            continue
        t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        wx, wy, wz = float(t[0]), float(t[1]), float(t[2])
        # Ceiling-level (z > 4) and within ~5 m of the camera eye in XY == in the lens's
        # near foreground. This clears the beam/lamp blob without gutting the ceiling.
        if wz > 4.0 and math.hypot(wx - ex, wy - ey) < 5.0:
            prim.SetActive(False)
            removed.append(f"{prim.GetName()}@({wx:+.1f},{wy:+.1f},{wz:+.1f})")
    _log(f"[FleetMind] Decluttered {len(removed)} foreground ceiling prim(s): {removed}")


def _apply_view():
    """Point the streamed viewport at FleetCam and strip the editor chrome so the
    stream shows ONLY the 3D simulation. All best-effort (wrapped per step)."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
        if vp is not None:
            vp.camera_path = CAMERA_PATH
            _log("[FleetMind] Viewport camera set to FleetCam.")
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] set active camera skipped: {e}")

    # Hide editor panels (Stage, Property, Console, Content, toolbars, layers…).
    try:
        import omni.ui as ui
        keep = {"Viewport", "Viewport 1", "Viewport 2", "DockSpace"}
        for w in ui.Workspace.get_windows():
            title = getattr(w, "title", "") or ""
            if title not in keep:
                try:
                    w.visible = False
                except Exception:
                    pass
        _log("[FleetMind] Editor panels hidden (viewport-only).")
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] hide panels skipped: {e}")

    # Hide the top menu bar for a fully clean frame.
    try:
        from omni.kit.mainwindow import get_main_window
        mw = get_main_window()
        for getter in ("get_main_menu_bar_widget", "get_menu_bar_widget"):
            fn = getattr(mw, getter, None)
            if fn:
                try:
                    fn().visible = False
                except Exception:
                    pass
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] hide menubar skipped: {e}")


def _schedule_apply_view(delay_frames=90):
    """Run _apply_view after the streaming app's UI is fully up, then unsubscribe."""
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        st = {"n": 0, "sub": None}

        def _cb(_e):
            st["n"] += 1
            if st["n"] >= delay_frames:
                _apply_view()
                if st["sub"] is not None:
                    st["sub"].unsubscribe()
                    st["sub"] = None

        st["sub"] = app.get_update_event_stream().create_subscription_to_pop(
            _cb, name="fleetmind-apply-view")
        _log("[FleetMind] View setup scheduled.")
    except Exception as e:  # pragma: no cover
        _log(f"[FleetMind] schedule view skipped ({e}); applying immediately.")
        _apply_view()


def build():
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac assets root path")

    warehouse = assets_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    forklift  = assets_root + FORKLIFT_SUBPATH
    # SM_PaletteA_01 — the flat wooden warehouse pallet with open fork slots (lower,
    # cleaner to fork under than the tall plastic Props/Pallet). "a09" isn't a real
    # asset; PaletteA_01 is the closest warehouse pallet.
    pallet    = assets_root + PALLET_SUBPATH
    payloads  = [
        assets_root + "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
        assets_root + "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
        assets_root + "/Isaac/Props/KLT_Bin/small_KLT.usd",
    ]

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.Xform.Define(stage, "/World")

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    # Enlarge the GPU broadphase / contact buffers. The warehouse mesh + detailed
    # pallet collision geometry overflowed the default aggregate-pair capacity
    # (PhysX logged "increase totalAggregatePairsCapacity to 1026" and stalled the
    # sim the moment a forklift engaged a pallet). Roomy fixed caps keep it stable.
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx_scene.CreateGpuTotalAggregatePairsCapacityAttr(64 * 1024)
    physx_scene.CreateGpuFoundLostAggregatePairsCapacityAttr(64 * 1024)
    physx_scene.CreateGpuFoundLostPairsCapacityAttr(64 * 1024)
    physx_scene.CreateGpuMaxRigidContactCountAttr(1024 * 1024)
    physx_scene.CreateGpuMaxRigidPatchCountAttr(160 * 1024)
    PhysicsSchemaTools.addGroundPlane(
        stage, "/World/GroundPlane", "Z", 200.0,
        Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.35),
    )
    mat_path = "/World/PhysicsMaterials/GroundMaterial"
    UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    pmat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pmat.CreateStaticFrictionAttr(GROUND_STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr(GROUND_DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr(0.0)
    ground_prim = stage.GetPrimAtPath("/World/GroundPlane")
    binding = UsdShade.MaterialBindingAPI.Apply(ground_prim)
    binding.Bind(
        UsdShade.Material(mat_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    add_reference_to_stage(usd_path=warehouse, prim_path="/World/Warehouse")
    _declutter_lights(stage)

    add_reference_to_stage(usd_path=forklift, prim_path="/World/_probe")
    size_x = _bbox_size_x(stage, "/World/_probe")
    fork_scale = 0.01 if size_x > 50.0 else 1.0
    stage.RemovePrim("/World/_probe")
    _log(f"[FleetMind] Forklift probe size_x={size_x:.2f} -> scale={fork_scale}")

    # Probe the pallet too — warehouse SM_ assets ship in cm, so a raw reference would
    # appear ~100x too big. Auto-scale to metres (a EUR pallet is ~1.2 m across).
    add_reference_to_stage(usd_path=pallet, prim_path="/World/_ppb")
    pal_x = _bbox_size_x(stage, "/World/_ppb")
    pallet_scale = 0.01 if pal_x > 50.0 else 1.0
    stage.RemovePrim("/World/_ppb")
    _log(f"[FleetMind] Pallet probe size_x={pal_x:.2f} -> scale={pallet_scale}")

    UsdGeom.Xform.Define(stage, "/World/AMRs")
    forklift_paths = {}
    for name, (x, y, yaw) in FORKLIFTS.items():
        p = f"/World/AMRs/{name}"
        add_reference_to_stage(usd_path=forklift, prim_path=p)
        _set_pose(stage, p, (x, y, SPAWN_Z), yaw, scale=fork_scale)
        _configure_drives(stage, p)
        forklift_paths[name] = p
        _BUS.register_forklift(name, x, y, math.radians(yaw))
        _log(f"[FleetMind] Placed {name} at ({x}, {y}, yaw={yaw})")
        _make_follow_cam(stage, name)

    UsdGeom.Xform.Define(stage, "/World/Pallets")
    UsdGeom.Xform.Define(stage, "/World/Payloads")
    for i, (x, y) in enumerate(PALLETS):
        pp = f"/World/Pallets/Pallet_{i:02d}"
        add_reference_to_stage(usd_path=pallet, prim_path=pp)
        _set_pose(stage, pp, (x, y, 0.0), 0.0, scale=pallet_scale)
        _make_pallet_physics(stage, pp, forklift_paths)
        yp = f"/World/Payloads/Payload_{i:02d}"
        add_reference_to_stage(usd_path=payloads[i % len(payloads)], prim_path=yp)
        _set_pose(stage, yp, (x, y, PALLET_TOP_Z + PAYLOAD_DROP), 0.0)
        _disable_physics(stage, yp)
        _BUS.set_pallet(f"WH_Palette_{i + 1:02d}", x=x, y=y, carried_by=None, delivered=False)

    UsdGeom.Xform.Define(stage, "/World/Zones")
    for zid, (x, y) in ZONES.items():
        _zone_marker(stage, zid, x, y)
        _BUS.set_zone(zid, x=x, y=y, blocked=False)

    UsdGeom.Xform.Define(stage, "/World/Chargers")
    for cid, (x, y) in CHARGERS.items():
        _charger_marker(stage, cid, x, y)

    # Draw a red route line per forklift; the waypoint-node dots are drawn lazily on the
    # first physics step (the bridge publishes the graph AFTER build()).
    UsdGeom.Xform.Define(stage, "/World/Waypoints")
    for name in FORKLIFTS:
        _make_route_overlay(stage, name)

    _setup_camera(stage)

    _log(f"[FleetMind] Scene built: {len(FORKLIFTS)} forklifts, {len(PALLETS)} pallets, "
         f"{len(ZONES)} staging zones, {len(CHARGERS)} chargers.")
    return forklift_paths


def _apply_cmd(art, idx, wheel_vel, steer_ang, lift_pos):
    art.apply_action(ArticulationAction(
        joint_velocities=np.array([wheel_vel]),
        joint_indices=np.array([idx["drive"]]),
    ))
    art.apply_action(ArticulationAction(
        joint_positions=np.array([steer_ang, lift_pos]),
        joint_indices=np.array([idx["steer"], idx["lift"]]),
    ))


def _quat_z(yaw):
    """World orientation quaternion [w, x, y, z] for a pure yaw about +Z."""
    return np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)])


def _slew(cur, target, max_step):
    """Rotate `cur` toward `target` by at most `max_step` (shortest way)."""
    d = _wrap(target - cur)
    if abs(d) <= max_step:
        return _wrap(target)
    return _wrap(cur + math.copysign(max_step, d))


def _set_base_pose(c, x, y, yaw):
    """Kinematically place the forklift base and zero its velocity so the physics
    solver never fights the commanded path. The lift joint is still actuated so the
    fork can raise/lower and carry the pallet."""
    art = c["art"]
    art.set_world_pose(
        position=np.array([x, y, c.get("ground_z", SPAWN_Z)]),
        orientation=_quat_z(yaw),
    )
    try:
        art.set_velocities(np.zeros(6))
    except Exception:
        try:
            art.set_linear_velocity(np.zeros(3))
            art.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
    _apply_cmd(art, c["idx"], 0.0, 0.0, c["lift"])



def _init_controllers():
    ctrls = {}
    for name, path in _RT["paths"].items():
        try:
            art = SingleArticulation(prim_path=path, name=name)
            art.initialize()
            names = list(art.dof_names)
            idx = {
                "drive": names.index("back_wheel_drive"),
                "steer": names.index("back_wheel_swivel"),
                "lift":  names.index("lift_joint"),
            }
            _apply_cmd(art, idx, 0.0, 0.0, 0.0)
            pos0, quat0 = art.get_world_pose()
            hx, hy, hdeg = FORKLIFTS[name]
            ctrls[name] = {
                "art": art, "idx": idx,
                "phase": "settle", "k": 0,
                # Kinematic base: pose is commanded directly along the route (no drive/
                # steer physics, no sign calibration). We pin the truck to its intended
                # spawn from step 1 so a bad spawn contact can't fling/tilt it; ground_z
                # is the fixed glide height.
                "home": (float(hx), float(hy)), "home_yaw": math.radians(hdeg),
                "ground_z": SPAWN_Z,
                # Commanded pose is AUTHORITATIVE: we integrate it ourselves and drive
                # the sim toward it, and NEVER read the physics pose back for control.
                # Reading back a floating-base articulation that we teleport each step
                # let PhysX penetration shove the read-back position and it ran away to
                # (-800,-1300) at 4.5 m/s. Commanded authority keeps logic/telemetry
                # exact; set_world_pose + zero-velocity keeps the visual glued to it.
                "cx": float(hx), "cy": float(hy), "cyaw": math.radians(hdeg),
                "last_xy": None,
                "lift_down": 0.0, "lift_up": LIFT_RAISE, "lift": 0.0,
                # mission execution
                "seq": -1, "legs": [], "leg_i": 0, "wp_i": 0,
                "carrying": None, "carry_path": None,
            }
            _log(f"[FleetMind] {name} articulation ready | dofs={len(names)} "
                 f"| start=({float(pos0[0]):.2f},{float(pos0[1]):.2f})")
        except Exception as e:
            _log(f"[FleetMind] ERROR init {name}: {e}")
    _RT["ctrls"] = ctrls


def _pull_command(name, c):
    """Adopt a newer mission from the bus (resets leg/waypoint cursors)."""
    cmd = _BUS.get_command(name)
    if cmd.seq != c["seq"] and cmd.legs:
        c["seq"] = cmd.seq
        c["legs"] = cmd.legs
        c["leg_i"] = 0
        c["wp_i"] = 0
        c["phase"] = "navigate"
        _log(f"[FleetMind] {name} mission seq={cmd.seq} legs="
             f"{[(l.action, l.target) for l in cmd.legs]}")


def _phase_label(c):
    if c["phase"] in ("settle", "calib_drive", "calib_steer"):
        return "idle"
    if c.get("yielding"):
        return "yielding"
    if c["phase"] == "navigate":
        return "carrying" if c["carrying"] else "navigating"
    if c["phase"] == "act":
        leg = c["legs"][c["leg_i"]]
        return "lifting" if leg.action in ("pick", "drop") else "navigating"
    if c["phase"] == "return":
        return "returning"
    return "idle"


def _publish(name, c, x, y, yaw, speed, target, route):
    _BUS.update_telemetry(
        name, x=x, y=y, yaw=yaw, speed=speed,
        phase=_phase_label(c), carrying=c["carrying"], lift_height=c["lift"],
        target=target, route=route, goal_kind=None,
        object_detected="None", object_distance=0.0,
        path_blocked=False,
        battery=c.get("battery", 100.0),
    )


def _tick_battery(c, x, y):
    """Drain the battery with distance driven; recharge while parked on a charger."""
    b = c.get("battery", 100.0)
    last = c.get("_bat_xy")
    c["_bat_xy"] = (x, y)
    moved = 0.0 if last is None else math.hypot(x - last[0], y - last[1])
    if moved > 1e-4:
        b -= moved * BATTERY_DRAIN_PER_M
    elif c.get("phase") in ("idle", "settle"):
        near = min((math.hypot(x - cx, y - cy) for (cx, cy) in CHARGERS.values()), default=1e9)
        if near < CHARGER_RADIUS:
            b += BATTERY_CHARGE_PER_STEP
    c["battery"] = max(0.0, min(100.0, b))


def _has_right_of_way(name, c, other_name, other_c):
    """Strict right-of-way order so exactly one truck of any pair yields (no deadlock):
    a LOADED truck outranks an empty one; if both are the same, AMR_1 outranks AMR_2."""
    if bool(c["carrying"]) != bool(other_c["carrying"]):
        return bool(c["carrying"])
    return name < other_name


def _nearest_other(name, cx, cy):
    """The closest OTHER forklift as (other_name, ox, oy, other_c, dist), or None."""
    best = None
    for on, oc in _RT.get("ctrls", {}).items():
        if on == name:
            continue
        ox, oy = oc["cx"], oc["cy"]
        d = math.hypot(ox - cx, oy - cy)
        if best is None or d < best[4]:
            best = (on, ox, oy, oc, d)
    return best


def _yield_check(name, c, cx, cy, cyaw, travel, dt):
    """Reactive collision avoidance vs the other forklift, evaluated every step.

    Returns (handled, cx, cy, cyaw, speed):
      * handled=False -> no conflict, caller runs its normal route-following.
      * handled=True  -> this truck is yielding; the returned pose/speed have already
                         been applied (brake for a moving truck, or side-step around a
                         stationary/stopped one).
    `travel` is the truck's desired travel heading (world radians), or None if it has no
    active waypoint.
    """
    near = _nearest_other(name, cx, cy)
    c["yielding"] = False
    if not near:
        return (False, cx, cy, cyaw, 0.0)
    on, ox, oy, oc, dist = near
    if dist >= YIELD_DIST:
        return (False, cx, cy, cyaw, 0.0)

    # A truck actively driving its route is "moving"; one that is idle/parked/picking OR
    # currently braked-and-yielding is treated as a STATIONARY obstacle (it won't clear
    # for us, so we must go around it rather than wait forever).
    other_moving = (oc.get("phase") == "navigate") and not oc.get("yielding")
    # Give way to a MOVING truck only when we are the lower-priority one; ALWAYS avoid a
    # stationary blocker regardless of priority (no deadlock: it has no goal to reach).
    if other_moving and _has_right_of_way(name, c, on, oc):
        return (False, cx, cy, cyaw, 0.0)

    to_ox, to_oy = (ox - cx), (oy - cy)
    tod = math.hypot(to_ox, to_oy) or 1.0
    # Ignore a truck clearly behind our travel direction (it isn't in our way).
    if travel is not None:
        cosang = math.cos(travel) * (to_ox / tod) + math.sin(travel) * (to_oy / tod)
        if cosang < YIELD_BEHIND:
            return (False, cx, cy, cyaw, 0.0)

    c["yielding"] = True
    if not other_moving and travel is not None:
        # STATIONARY blocker ahead: steer AROUND it. Side-step perpendicular to travel,
        # toward the side the blocker isn't on, with a little forward creep so we make
        # progress past it; the route is rejoined automatically once it's no longer ahead.
        px, py = -math.sin(travel), math.cos(travel)
        lat = px * to_ox + py * to_oy          # >0: blocker sits to our left
        s = -1.0 if lat > 0 else 1.0
        step = KIN_SPEED * EVADE_SPEED * dt
        cx += (s * px * 0.9 + math.cos(travel) * 0.3) * step
        cy += (s * py * 0.9 + math.sin(travel) * 0.3) * step
        c["cx"], c["cy"] = cx, cy
        _set_base_pose(c, cx, cy, cyaw)
        return (True, cx, cy, cyaw, step / max(dt, 1e-3))
    # MOVING higher-priority truck: brake in place and wait; resume next step once clear.
    _set_base_pose(c, cx, cy, cyaw)
    return (True, cx, cy, cyaw, 0.0)


def _step_one(name, c, dt):
    art, idx = c["art"], c["idx"]

    # ---- settle: pin XY+yaw at home, let Z settle, capture resting height --- #
    if c["phase"] == "settle":
        c["k"] += 1
        hx, hy = c["home"]
        pos, _ = art.get_world_pose()
        mz = float(pos[2])
        # Pin XY+yaw to the intended spawn (no fling) but leave Z free so the body
        # settles onto the floor; we don't zero velocity here so it CAN drop.
        art.set_world_pose(position=np.array([hx, hy, mz]),
                           orientation=_quat_z(c["home_yaw"]))
        _apply_cmd(art, idx, 0.0, 0.0, c["lift_down"])
        if c["k"] >= SETTLE_STEPS:
            c["ground_z"] = min(1.0, max(-0.2, mz))   # clamp against a fling artifact
            c["phase"], c["k"] = "idle", 0
            _log(f"[FleetMind] {name} ready (kinematic) home=({hx:+.2f},{hy:+.2f}) "
                 f"yaw={math.degrees(c['home_yaw']):+.0f} z={c['ground_z']:.3f}")
        _publish(name, c, hx, hy, c["home_yaw"], 0.0, None, [])
        c["last_xy"] = (hx, hy)
        return hx, hy, c["home_yaw"]

    # From here on the COMMANDED pose is the source of truth.
    cx, cy, cyaw = c["cx"], c["cy"], c["cyaw"]
    c["k"] += 1

    # ---- idle: hold commanded pose, wait for a mission ------------------ #
    if c["phase"] == "idle":
        _set_base_pose(c, cx, cy, cyaw)
        _pull_command(name, c)
        _publish(name, c, cx, cy, cyaw, 0.0, None, [])
        c["last_xy"] = (cx, cy)
        return cx, cy, cyaw

    # A newer mission always preempts.
    _pull_command(name, c)
    leg = c["legs"][c["leg_i"]]

    # ---- navigate: integrate the commanded pose along the route --------- #
    if c["phase"] == "navigate":
        wps = leg.waypoints
        speed = 0.0
        if c["wp_i"] >= len(wps):
            _set_base_pose(c, cx, cy, cyaw)
            c["phase"], c["k"] = "act", 0
            _log(f"[FleetMind] {name} arrived leg {c['leg_i']} '{leg.action}' "
                 f"target={leg.target} at ({cx:+.2f},{cy:+.2f}) -> act")
        else:
            tx, ty = wps[c["wp_i"]]
            dx, dy = tx - cx, ty - cy
            dist = math.hypot(dx, dy)
            travel = math.atan2(dy, dx)
            # Reactive yield to the other forklift BEFORE moving: brake or side-step if a
            # crossing/head-on is developing (route was planned static; trucks move live).
            handled, cx, cy, cyaw, yspeed = _yield_check(name, c, cx, cy, cyaw, travel, dt)
            if handled:
                route = [leg.target] if leg.target else []
                _publish(name, c, cx, cy, cyaw, yspeed, leg.target, route)
                _carry_follow(c, cx, cy, cyaw)
                c["last_xy"] = (cx, cy)
                return cx, cy, cyaw
            is_last = (c["wp_i"] == len(wps) - 1)
            if is_last:
                reach = PICK_ARRIVE if leg.action in ("pick", "drop") else ARRIVE_DIST
            else:
                reach = WAYPOINT_DIST
            if dist < reach:
                c["wp_i"] += 1
            else:
                # Step the commanded pose toward the waypoint at constant speed (clamped
                # so we never overshoot) and slew heading toward travel so the forks
                # lead. Deterministic: arrival is exact, no orbit/stall/ram/runaway.
                step = min(KIN_SPEED * dt, dist)
                cx += step * math.cos(travel)
                cy += step * math.sin(travel)
                cyaw = _slew(cyaw, _wrap(travel - MODEL_YAW_OFFSET), TURN_RATE * dt)
                c["cx"], c["cy"], c["cyaw"] = cx, cy, cyaw
                _set_base_pose(c, cx, cy, cyaw)
                speed = step / max(dt, 1e-3)
        route = [leg.target] if leg.target else []
        _publish(name, c, cx, cy, cyaw, speed, leg.target, route)
        _carry_follow(c, cx, cy, cyaw)
        c["last_xy"] = (cx, cy)
        return cx, cy, cyaw

    # ---- act: perform the leg's terminal action ------------------------- #
    if c["phase"] == "act":
        _set_base_pose(c, cx, cy, cyaw)
        # Only engage/disengage the fork when genuinely at the leg's end point.
        gx, gy = leg.waypoints[-1] if leg.waypoints else (cx, cy)
        at_target = math.hypot(gx - cx, gy - cy) <= PICK_SNAP
        if leg.action == "pick":
            # The forks are already low and, at arrival, reaching UNDER the pallet
            # (fork tip == pallet cell). Engage the load in place first, then raise
            # the fork so _carry_follow lifts the pallet smoothly off the floor —
            # no teleport onto the truck, no pre-raised "spawn on" pop.
            if at_target and c["carrying"] is None:
                c["carrying"] = leg.target
                c["carry_path"] = leg.pallet_path or _PALLET_PATH.get(leg.target)
                if c["carry_path"]:
                    # Make the load kinematic so it rides the fork tip exactly (also
                    # re-arms a pallet that was previously dropped as a dynamic body).
                    _set_pallet_kinematic(_RT["stage"], c["carry_path"], True)
                _BUS.set_pallet(leg.target, carried_by=name)
                _log(f"[FleetMind] {name} ENGAGED {leg.target} (forks under load)")
            if c["carrying"] is not None:
                c["lift"] = min(c["lift_up"], c["lift"] + LIFT_RATE)
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            _carry_follow(c, cx, cy, cyaw)
            if c["k"] >= ACT_STEPS and c["lift"] >= c["lift_up"] - 1e-3:
                _advance_leg(c)
        elif leg.action == "drop":
            # Lower the fork first so the carried pallet descends with it, then release
            # it onto the staging cell (delivered) — a smooth set-down, not a snap.
            if at_target and c["carrying"] is not None:
                c["lift"] = max(c["lift_down"], c["lift"] - LIFT_RATE)
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            _carry_follow(c, cx, cy, cyaw)
            if at_target and c["lift"] <= c["lift_down"] + 1e-3 and c["carrying"] is not None:
                dx, dy = leg.drop_xy if leg.drop_xy else _fork_xy(cx, cy, cyaw)
                if c["carry_path"]:
                    # Position over the drop cell, then hand the pallet to gravity: it
                    # switches from kinematic to a dynamic rigid body and settles onto the
                    # floor under real physics (collision vs the ground stops the clip).
                    _move_prim_xy(_RT["stage"], c["carry_path"], dx, dy, c["lift"])
                    _set_pallet_kinematic(_RT["stage"], c["carry_path"], False)
                _BUS.set_pallet(c["carrying"], x=dx, y=dy, carried_by=None, delivered=True)
                _log(f"[FleetMind] {name} DROPPED {c['carrying']} at ({dx:+.2f},{dy:+.2f})")
                c["carrying"], c["carry_path"] = None, None
            if c["k"] >= ACT_STEPS and c["lift"] <= c["lift_down"] + 1e-3:
                _advance_leg(c)
        else:  # goto / home — nothing to actuate
            _advance_leg(c)
        _publish(name, c, cx, cy, cyaw, 0.0, leg.target, [leg.target] if leg.target else [])
        c["last_xy"] = (cx, cy)
        return cx, cy, cyaw

    _set_base_pose(c, cx, cy, cyaw)
    _publish(name, c, cx, cy, cyaw, 0.0, None, [])
    c["last_xy"] = (cx, cy)
    return cx, cy, cyaw


def _advance_leg(c):
    c["leg_i"] += 1
    c["wp_i"] = 0
    c["k"] = 0
    if c["leg_i"] >= len(c["legs"]):
        c["phase"] = "idle"
        c["legs"] = []
        c["leg_i"] = 0
    else:
        c["phase"] = "navigate"


def _fork_xy(x, y, yaw):
    """World XY of the fork tip: FORK_REACH ahead of the truck centre along its
    travel/forward heading (root_yaw + MODEL_YAW_OFFSET == travel direction)."""
    fwd = yaw + MODEL_YAW_OFFSET
    return x + math.cos(fwd) * FORK_REACH, y + math.sin(fwd) * FORK_REACH


def _carry_follow(c, x, y, yaw):
    if c["carrying"] and c["carry_path"]:
        fx, fy = _fork_xy(x, y, yaw)
        _move_prim_xy(_RT["stage"], c["carry_path"], fx, fy, c["lift"])


def _active_route_pts(c, x, y):
    """Remaining route to draw as the red floor line: the truck's current position
    followed by the not-yet-reached waypoints of every remaining leg. Empty when idle."""
    if c["phase"] in ("settle", "idle") or c["leg_i"] >= len(c["legs"]):
        return []
    pts = [(x, y)]
    leg = c["legs"][c["leg_i"]]
    pts.extend(leg.waypoints[c["wp_i"]:])
    for nleg in c["legs"][c["leg_i"] + 1:]:
        pts.extend(nleg.waypoints)
    return pts


def _on_step(dt):
    _RT["warm"] += 1
    if _RT["warm"] < WARMUP_STEPS:
        return
    if "ctrls" not in _RT:
        _init_controllers()
        return
    if not _RT.get("wp_drawn"):
        # Bridge publishes the graph after build(); draw the node dots once it's up.
        if (_BUS.graph or {}).get("nodes"):
            _build_waypoint_markers(_RT["stage"])
            _RT["wp_drawn"] = True
    _RT["n"] += 1
    do_log = (_RT["n"] % LOG_EVERY == 0)
    step_dt = dt if dt and dt > 1e-4 else (1.0 / 60.0)
    for name, c in _RT["ctrls"].items():
        try:
            x, y, yaw = _step_one(name, c, step_dt)
            _tick_battery(c, x, y)
            _update_follow_cam(_RT["stage"], name, x, y, yaw)
            _update_route_overlay(_RT["stage"], name, _active_route_pts(c, x, y))
            if do_log:
                _log(f"[FleetMind] {name} [{_phase_label(c)}] "
                     f"pos=({x:+.2f},{y:+.2f}) yaw={math.degrees(yaw):+.0f} "
                     f"carry={c['carrying']} leg={c['leg_i']}/{len(c['legs'])} "
                     f"wp={c['wp_i']}")
        except Exception as e:
            if do_log:
                _log(f"[FleetMind] {name} step error: {e}")


def start_autonomy(forklift_paths):
    _RT["paths"] = forklift_paths
    _RT["stage"] = omni.usd.get_context().get_stage()
    _RT["warm"] = 0
    _RT["n"] = 0
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    timeline.play()
    physx = omni.physx.get_physx_interface()
    _RT["sub"] = physx.subscribe_physics_step_events(_on_step)
    _log("[FleetMind] Autonomy started: timeline playing, controller subscribed.")


def start_bridge():
    """Run the control bridge (SIM_BACKEND=isaac) in a daemon thread, in-process, so its
    IsaacNavBackend shares the fleet-bus singleton with this controller."""
    os.environ["SIM_BACKEND"] = "isaac"

    def _serve():
        try:
            import uvicorn
            from src.ros2.bridge_server import app
            _log("[FleetMind] Bridge starting on :8080 (SIM_BACKEND=isaac)")
            uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
        except Exception as e:
            _log(f"[FleetMind] Bridge NOT started ({e}). "
                 f"pip install fastapi uvicorn into Isaac's python to enable it, "
                 f"or run the bridge as a separate SIM_BACKEND=isaac process.")

    threading.Thread(target=_serve, name="fleetmind-bridge", daemon=True).start()


paths = build()
start_autonomy(paths)
start_bridge()
_schedule_apply_view()
