"""
FleetMind — warehouse scene + BUS-DRIVEN autonomy, loaded INTO the running Isaac Sim
streaming app via Kit's `--exec` hook (runheadless.sh owns the render loop).

This is the production source of truth for the deployed scene. It:

    build()          -> new stage, warehouse, physics + ground, 2 rigged forklifts,
                                            3 pallets and 3 staging-zone floor markers.
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
    (-3.0, -3.0), (-3.0, 3.0), (1.0, 3.0),
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
PALLET_FLOOR_Z = 0.0                 # pallet origin rests here on the floor / a bay pad

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
# TODO: Find a better way to make the pallet physics feel realistic. The current values are a hack to make the pallet not slide around too much when being carried.
REALISTIC_PALLET_PHYSICS = True
PALLET_STATIC_FRICTION  = 0.5
PALLET_DYNAMIC_FRICTION = 0.4

# ---- Kinematic path-following controller ----------------------------------
# The forklift base is driven KINEMATICALLY: each physics step we set its world pose
# directly along the A* waypoint route (constant speed, yaw slewed toward the travel
# direction) and zero its velocity. This is deterministic and demo-safe — it removes
# the rear-wheel drive/steer physics, the runtime drive/steer sign calibration, the
# stall watchdog and the reverse-recovery that made the truck drive the wrong way,
# orbit the pallet, or ram it. The fork (lift_joint) is still actuated by its physics
# drive, and a carried pallet is tracked from its live physics pose (see _carry_follow).
KIN_SPEED     = 1.6                 # m/s base travel speed along the route
TURN_RATE     = math.radians(140.0) # rad/s max yaw slew toward travel heading
CARRY_TURN_RATE = math.radians(25.0) # rad/s max yaw slew while carrying a live pallet
# Battery model: each forklift starts full and drains with distance travelled; when it
# sits idle at (near) its home charger it trickles back up. cuOpt reads the live % from
# the snapshot and (a) won't route a truck past its remaining range and (b) prefers the
# more-charged truck — see cuopt_planner. BATTERY_DRAIN_PER_M must match the planner's
# BATTERY_RANGE_PER_PCT (range = battery / drain, i.e. 1/0.5). Drain is steep enough that
# the busiest truck in a multi-pallet dispatch visibly crosses the low-battery threshold
# (ends ~20%) so cuOpt then holds it back to recharge on the next dispatch — yet it never
# strands mid-tour (full range 50 m > the ~40 m busiest split leg). A lone truck clearing
# EVERY pallet (~57 m) now exceeds one charge, so cuOpt splits/holds instead of solo-routing.
BATTERY_FULL         = 100.0
BATTERY_DRAIN_PER_M  = 2.0          # % of charge spent per metre driven (== 1 / 0.5 m/%)
BATTERY_CHARGE_PER_S = 10.0         # % regained per second while idle on the charger
BATTERY_CHARGE_RADIUS = 1.5         # m from home within which charging happens
# The rigged model's visual forward (fork direction) is offset from the articulation
# root's reported yaw. We orient the truck by setting root_yaw = travel - MODEL_YAW_OFFSET
# so the forks point along travel (~ -90°: the asset points toward -Y at reported yaw 0).
# Kinematic control means this is fixed once and never drifts.
MODEL_YAW_OFFSET = math.radians(-90.0)

WAYPOINT_DIST = 0.6                 # m, advance to next waypoint within this
ARRIVE_DIST   = 0.30                # m, intermediate leg goal reached
PICK_ARRIVE   = 0.20                # m, tight final arrival for pick/drop
PICK_SNAP     = 1.60                # m, max dist to leg end for the fork to (dis)engage
PICK_INSERT_SPEED = 0.18            # m/s, very slow forward creep to insert forks
PICK_INSERT_TOL   = 0.16            # m, fork-tip distance to pallet centre considered inserted
PICK_INSERT_MAX   = 0.45            # m, safety cap on final insertion distance
PICK_SLOW_DIST    = 2.0             # m, start slowing before the final pick waypoint
PICK_SLOW_MIN     = 0.50            # speed scale at the final metres of pick approach
DROP_BACKOFF_DIST = 1.2             # m, reverse after setting a pallet down to disengage forks
DROP_BACKOFF_SPEED = 0.75           # m/s, slow reverse while backing out from a drop
# Forks reach this far ahead of the truck centre (travel/forward direction). The bridge's
# pick APPROACH_LEN is intentionally a bit larger than this reach, so normal-speed
# navigation stops with the fork tip just SHORT of the pallet. Only the slow insertion
# creep enters the pallet slots; otherwise the kinematic truck pushes the pallet.
FORK_REACH    = 1.4
LIFT_RAISE    = 0.42                # m visible fork travel (pallet clears the floor)
LIFT_RATE     = 0.01                # m/step fork travel
LIFT_DOWN_OFFSET = 0.07             # m above the joint lower limit for the fork's rest pose
# Simplified-mode carried-load vertical anchoring: the lift joint command (`c["lift"]`)
# is not the same as the visual tine world Z on this rig. Without an offset the pallet can
# appear clipped below the forks while "carried". Realistic mode reads the live pallet pose.
FORK_CARRY_Z_OFFSET = 0.16          # m above lift command where the pallet sits on forks
PALLET_LIFTED_Z = 0.05              # m above floor: treat pallet as physically lifted
CARRY_SPEED_SCALE = 0.65            # slower loaded travel so dynamic pallet contacts settle
CARRY_ACCEL       = 0.55            # m/s^2 acceleration cap while carrying a live pallet
LOADED_EXIT_MIN   = 0.55            # m, after pick back straight out before following route
LOADED_EXIT_EXTRA = 0.25            # m beyond fork insertion, keeps first loaded move straight
LOADED_EXIT_MAX   = 0.90            # m, cap straight reverse so we do not over-back into aisle
LOADED_EXIT_SPEED = 0.35            # m/s, gentle reverse while pallet is newly lifted
LOADED_EXIT_ACCEL = 0.18            # m/s^2, avoid instant slip impulse when backing out
LOADED_WAYPOINT_DIST = 0.18         # m, do not skip small correction waypoints while loaded
TURN_SLOW_FULL = math.radians(70.0) # yaw error at which turn-slowing reaches full effect
TURN_SLOW_MIN = 0.45                # unloaded minimum speed scale during sharp turns
CARRY_TURN_SLOW_MIN = 0.20          # loaded minimum speed scale during sharp turns
PICK_LIFT_TIMEOUT_STEPS = 60        # don't wait forever for a noisy/missing lifted signal
WARMUP_STEPS  = 5
SETTLE_STEPS  = 60                  # let the base settle on the ground, capture rest Z
ACT_STEPS     = 30                  # steps to hold during pick/drop lift
LOG_EVERY     = 120
# --- live inter-forklift collision avoidance (reactive, runs every step) ----------
# The bridge plans a static route around wherever the OTHER truck is at dispatch time,
# but both trucks move, so a crossing/head-on can still develop mid-route. These trucks
# therefore watch each other every physics step: when a conflict develops, the give-way
# truck STOPS and holds until the other has cleared, then resumes its route. It never
# side-steps (that pushed it INTO the other's path and clipped) — it simply waits.
# Right-of-way is a strict total order so exactly one truck of any pair yields (no
# deadlock): a LOADED truck outranks an empty one; ties break by name (AMR_1 first).
YIELD_DIST    = 4.5                 # m, stop and hold when another moving truck blocks us
YIELD_BEHIND  = -0.2               # cos threshold: ignore a truck clearly behind us
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


def _configure_physics_material(mat_prim, static_friction, dynamic_friction):
    """Author USD/PhysX material attrs with best-effort combine modes."""
    pmat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pmat.CreateStaticFrictionAttr(float(static_friction))
    pmat.CreateDynamicFrictionAttr(float(dynamic_friction))
    pmat.CreateRestitutionAttr(0.0)
    try:
        physx_mat_api = getattr(PhysxSchema, "PhysxMaterialAPI", None)
        if physx_mat_api is None:
            return
        physx_mat = physx_mat_api.Apply(mat_prim)
        max_token = getattr(getattr(PhysxSchema, "Tokens", object()), "max", "max")
        min_token = getattr(getattr(PhysxSchema, "Tokens", object()), "min", "min")
        for attr_name, value in (
                ("CreateFrictionCombineModeAttr", max_token),
                ("CreateRestitutionCombineModeAttr", min_token)):
            create_attr = getattr(physx_mat, attr_name, None)
            if create_attr is not None:
                create_attr(value)
    except Exception:
        pass


def _make_pallet_physics(stage, prim_path, forklift_paths):
    """Give a pallet rigid-body physics, collision, damping, and a physics material.

    In realistic mode the pallet remains a dynamic rigid body so the fork lift/carry is
    contact-driven. In simplified mode forklift/pallet contacts are filtered and the
    carried pallet can be moved kinematically by the controller.

    CRITICAL: the SM_PaletteA collision ships as a *triangle mesh*, and PhysX cannot make
    a triangle-mesh body dynamic unless it has an SDF collision representation. We
    therefore force every realistic-mode collider to SDF (falling back to convex hull only
    in simplified mode), so the dynamic pallet keeps a detailed collision shape instead
    of a loose hull approximation."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(not REALISTIC_PALLET_PHYSICS)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(55.0 if REALISTIC_PALLET_PHYSICS else 25.0)
    try:
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        prb.CreateEnableCCDAttr(True)
        if REALISTIC_PALLET_PHYSICS:
            prb.CreateLinearDampingAttr(1.2)
            prb.CreateAngularDampingAttr(3.0)
    except Exception:
        pass
    sdf_approx = getattr(UsdPhysics.Tokens, "sdf", "sdf")
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            a = p.GetAttribute("physics:collisionEnabled")
            if a:
                a.Set(True)
            mc = UsdPhysics.MeshCollisionAPI.Apply(p)
            if REALISTIC_PALLET_PHYSICS:
                mc.CreateApproximationAttr().Set(sdf_approx)
                try:
                    sdf_api = getattr(PhysxSchema, "PhysxSDFMeshCollisionAPI", None)
                    if sdf_api is not None:
                        sdf = sdf_api.Apply(p)
                        create_resolution = getattr(sdf, "CreateSdfResolutionAttr", None)
                        if create_resolution is not None:
                            create_resolution(128)
                        create_subgrid = getattr(sdf, "CreateSdfSubgridResolutionAttr", None)
                        if create_subgrid is not None:
                            create_subgrid(6)
                except Exception:
                    pass
            else:
                mc.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    if not REALISTIC_PALLET_PHYSICS:
        # Filter pallet<->forklift contacts only in the simplified carry mode.
        fp = UsdPhysics.FilteredPairsAPI.Apply(prim)
        rel = fp.CreateFilteredPairsRel()
        for fpath in forklift_paths.values():
            rel.AddTarget(Sdf.Path(fpath))
    else:
        mat_path = f"{prim_path}/PhysicsMaterial"
        UsdShade.Material.Define(stage, mat_path)
        mat_prim = stage.GetPrimAtPath(mat_path)
        _configure_physics_material(mat_prim, PALLET_STATIC_FRICTION, PALLET_DYNAMIC_FRICTION)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(mat_prim),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )


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


def _set_rigid_body_velocities(stage, prim_path, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
    """Reset a rigid body's linear / angular velocity so PhysX does not carry stale
    motion across a teleport/reset."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    try:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateVelocityAttr().Set(Gf.Vec3f(float(linear[0]), float(linear[1]), float(linear[2])))
        rb.CreateAngularVelocityAttr().Set(
            Gf.Vec3f(float(angular[0]), float(angular[1]), float(angular[2]))
        )
    except Exception:
        for attr_name, value in (
            ("physics:velocity", linear),
            ("physics:angularVelocity", angular),
            ("state:linear:physics:velocity", linear),
            ("state:angular:physics:velocity", angular),
        ):
            try:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    attr.Set(Gf.Vec3f(float(value[0]), float(value[1]), float(value[2])))
            except Exception:
                pass


def _set_rigid_body_enabled(stage, prim_path, enabled=True):
    """Ensure a rigid body is enabled after a reset/carry/drop cycle."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    try:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(bool(enabled))
    except Exception:
        try:
            attr = prim.GetAttribute("physics:rigidBodyEnabled")
            if attr and attr.IsValid():
                attr.Set(bool(enabled))
        except Exception:
            pass


def _set_collision_enabled(stage, prim_path, enabled=True):
    """Restore collision flags on a pallet subtree after a reset."""
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        try:
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr and attr.IsValid():
                attr.Set(bool(enabled))
        except Exception:
            pass


def _log_pallet_physics_state(stage, label):
    """Log a compact pallet PhysX/collider summary for reset/collider debugging."""
    bits = []
    for i, _xy in enumerate(PALLETS):
        pp = f"/World/Pallets/Pallet_{i:02d}"
        prim = stage.GetPrimAtPath(pp)
        if not prim or not prim.IsValid():
            bits.append(f"P{i}:missing")
            continue
        rb = prim.GetAttribute("physics:rigidBodyEnabled")
        kin = prim.GetAttribute("physics:kinematicEnabled")
        enabled_colliders = 0
        total_colliders = 0
        for p in Usd.PrimRange(prim):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                continue
            total_colliders += 1
            attr = p.GetAttribute("physics:collisionEnabled")
            if not attr or attr.Get() is not False:
                enabled_colliders += 1
        bits.append(
            f"P{i}:rb={rb.Get() if rb else None},kin={kin.Get() if kin else None},"
            f"coll={enabled_colliders}/{total_colliders}"
        )
    _log(f"[FleetMind] Pallet physics {label}: {'; '.join(bits)}")


def _rearm_reset_pallet_physics(stage):
    """After a reset teleport, release pallets back to dynamic mode with zeroed state.

    Teleporting a live rigid body and immediately expecting stable carry behaviour left
    stale solver velocity/contact state behind after a reset. Re-arming one beat later
    gives PhysX a clean dynamic body again.
    """
    for i, _xy in enumerate(PALLETS):
        pp = f"/World/Pallets/Pallet_{i:02d}"
        _set_rigid_body_enabled(stage, pp, True)
        _set_collision_enabled(stage, pp, True)
        _set_rigid_body_velocities(stage, pp)
        _set_pallet_kinematic(stage, pp, False)
        _set_rigid_body_velocities(stage, pp)
    _log_pallet_physics_state(stage, "re-armed")


def _reset_forklift_articulation(c):
    """Hard-reset a forklift articulation so no stale motion survives reset."""
    art = c["art"]
    hx, hy = c["home"]
    hyaw = c["home_yaw"]
    _set_base_pose(c, hx, hy, hyaw)
    try:
        art.set_joint_velocities(np.zeros(len(art.dof_names)))
    except Exception:
        pass
    try:
        positions = art.get_joint_positions()
        if positions is not None and len(positions) >= 3:
            positions = np.array(positions, dtype=float)
            positions[c["idx"]["steer"]] = 0.0
            positions[c["idx"]["lift"]] = c["lift_down"]
            art.set_joint_positions(positions)
    except Exception:
        pass
    _apply_cmd(art, c["idx"], 0.0, 0.0, c["lift_down"])


def _bbox_size_x(stage, prim_path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    if rng.IsEmpty():
        return 0.0
    return float(rng.GetMax()[0] - rng.GetMin()[0])


def _prim_world_xyz(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


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


def _joint_limit(prim, attr_name, fallback):
    """Read a scalar joint limit attr from USD, falling back when missing/invalid."""
    if not prim or not prim.IsValid():
        return float(fallback)
    try:
        attr = prim.GetAttribute(attr_name)
        if attr:
            value = attr.Get()
            if value is not None:
                return float(value)
    except Exception:
        pass
    return float(fallback)


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
    _RT["pallet_scale"] = pallet_scale
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
    for i, (x, y) in enumerate(PALLETS):
        pp = f"/World/Pallets/Pallet_{i:02d}"
        add_reference_to_stage(usd_path=pallet, prim_path=pp)
        _set_pose(stage, pp, (x, y, 0.0), 0.0, scale=pallet_scale)
        _make_pallet_physics(stage, pp, forklift_paths)
        _BUS.set_pallet(f"WH_Palette_{i + 1:02d}", x=x, y=y, carried_by=None, delivered=False)
    _log_pallet_physics_state(stage, "built")

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
            lift_joint_prim = _RT["stage"].GetPrimAtPath(f"{path}/lift_joint/lift_joint")
            lift_lower_limit = _joint_limit(lift_joint_prim, "physics:lowerLimit", 0.0)
            lift_up_limit = _joint_limit(lift_joint_prim, "physics:upperLimit", LIFT_RAISE)
            lift_down = min(float(lift_up_limit), lift_lower_limit + LIFT_DOWN_OFFSET)
            lift_up = max(lift_down, min(float(lift_up_limit), LIFT_RAISE))
            idx = {
                "drive": names.index("back_wheel_drive"),
                "steer": names.index("back_wheel_swivel"),
                "lift":  names.index("lift_joint"),
            }
            _apply_cmd(art, idx, 0.0, 0.0, lift_down)
            pos0, quat0 = art.get_world_pose()
            hx, hy, hdeg = FORKLIFTS[name]
            ctrls[name] = {
                "art": art, "idx": idx, "name": name,
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
                # exact; set_world_pose + zero-velocity keeps the visual locked to it.
                "cx": float(hx), "cy": float(hy), "cyaw": math.radians(hdeg),
                "last_xy": None,
                "lift_down": lift_down, "lift_up": lift_up, "lift": lift_down,
                # mission execution
                "seq": -1, "legs": [], "leg_i": 0, "wp_i": 0,
                "carrying": None, "carry_path": None,
                "drop_pending": None, "drop_pending_xy": None, "drop_pending_path": None,
                "drop_backoff_m": 0.0,
                "pick_engaged": None,
                "pick_insert_m": 0.0,
                "loaded_exit_m": 0.0,
                "loaded_exit_goal": 0.0,
                "speed_cmd": 0.0,
                "battery": BATTERY_FULL,
            }
            _log(f"[FleetMind] {name} articulation ready | dofs={len(names)} "
                 f"| start=({float(pos0[0]):.2f},{float(pos0[1]):.2f})")
        except Exception as e:
            _log(f"[FleetMind] ERROR init {name}: {e}")
    _RT["ctrls"] = ctrls


def _pull_command(name, c):
    """Adopt a newer mission from the bus (resets leg/waypoint cursors).

    IMPORTANT: while the truck is still CARRYING a pallet we DEFER adopting a new mission
    until the current load has been dropped. Otherwise a command that arrives mid-delivery
    (classically "send all forklifts home") preempts the drop leg while `carrying` stays
    set, and `_carry_follow` then glues the undelivered pallet to the forks all the way to
    the new destination — the pallet "rides home" instead of being staged. By deferring, the
    truck finishes delivering to staging first, then (now empty, at idle) picks up the newer
    mission on the next tick. The bus keeps returning the latest command, so nothing is lost.

    EXCEPTION: a mission whose FIRST leg is a drop is a spill-reroute of the load we are
    already carrying to a clear bay — adopt it immediately so the truck diverts instead of
    driving into the now-blocked bay (the load still ends up staged, just elsewhere).
    """
    cmd = _BUS.get_command(name)
    if cmd.seq == c["seq"] or not cmd.legs:
        return
    if c.get("drop_pending"):
        return
    if c.get("carrying") and cmd.legs[0].action != "drop":
        return                       # finish the drop first; adopt once empty
    c["seq"] = cmd.seq
    c["legs"] = cmd.legs
    c["leg_i"] = 0
    c["wp_i"] = 0
    c["pick_engaged"] = None
    c["pick_insert_m"] = 0.0
    c["loaded_exit_m"] = 0.0
    c["loaded_exit_goal"] = 0.0
    c["speed_cmd"] = 0.0
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
        if leg.action == "pick":
            return "lifting"
        if leg.action == "drop":
            return "dropping"
        return "navigating"
    if c["phase"] == "return":
        return "returning"
    return "idle"


def _publish(name, c, x, y, yaw, speed, target, route):
    _update_battery(c, x, y)
    # UI/bridge yaw should represent forklift FORWARD/travel heading. The rigged
    # asset's articulation-root yaw is offset by MODEL_YAW_OFFSET (about -90°), so
    # convert back here before publishing telemetry.
    travel_yaw = _wrap(yaw + MODEL_YAW_OFFSET)
    route = _active_route_node_ids(c, x, y)
    _BUS.update_telemetry(
        name, x=x, y=y, yaw=travel_yaw, speed=speed,
        phase=_phase_label(c), carrying=c["carrying"], lift_height=c["lift"],
        target=target, route=route, goal_kind=None,
        object_detected="None", object_distance=0.0,
        path_blocked=False,
        battery=round(c["battery"], 1),
    )


def _update_battery(c, x, y):
    """Drain charge with distance driven since the last publish; trickle-charge when the
    truck is sitting on (near) its home charger. Kept purely in the commanded frame so it
    is deterministic and matches the route the planner costed."""
    last = c.get("last_xy")
    moved = math.hypot(x - last[0], y - last[1]) if last else 0.0
    batt = c.get("battery", BATTERY_FULL) - moved * BATTERY_DRAIN_PER_M
    # Recharge only while parked (negligible motion) within reach of the home charger.
    hx, hy = c["home"]
    if moved < 1e-3 and math.hypot(x - hx, y - hy) <= BATTERY_CHARGE_RADIUS:
        batt += BATTERY_CHARGE_PER_S * c.get("_dt", 0.0)
    c["battery"] = max(0.0, min(BATTERY_FULL, batt))



def _has_right_of_way(name, c, other_name, other_c):
    """Strict right-of-way order used ONLY as a tie-break for a true head-on (both trucks
    driving straight into each other). A LOADED truck outranks an empty one; if both are
    the same, AMR_1 outranks AMR_2. Antisymmetric, so exactly one truck yields."""
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


def _travel_heading(oc):
    """A controller's current world travel heading (radians) toward its active waypoint,
    or None if it isn't navigating (idle / lifting / no remaining waypoint)."""
    if oc.get("phase") != "navigate":
        return None
    legs = oc.get("legs") or []
    li, wi = oc.get("leg_i", 0), oc.get("wp_i", 0)
    if li >= len(legs):
        return None
    wps = legs[li].waypoints
    if wi >= len(wps):
        return None
    tx, ty = wps[wi]
    dx, dy = tx - oc["cx"], ty - oc["cy"]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def _should_i_stop(name, c, cx, cy, my_travel, on, oc):
    """Decide WHICH of the two trucks gives way — evaluated identically by both trucks so
    they always agree (exactly one stops, no deadlock, no mutual clip).

    Key idea: the truck that is driving MORE DIRECTLY INTO the other is the one whose path
    is blocked, so IT stops; the other truck's path is comparatively clear, so it keeps
    going and drives away — which then opens the gap and lets the stopped truck resume.
    A true head-on (both aimed at each other) is symmetric, so we break that tie with the
    strict right-of-way order.
    """
    ox, oy = oc["cx"], oc["cy"]
    d = math.hypot(ox - cx, oy - cy) or 1.0
    other_travel = _travel_heading(oc)
    # my_ahead: how directly the OTHER sits in front of MY heading (1 = dead ahead).
    my_ahead = -2.0
    if my_travel is not None:
        my_ahead = (math.cos(my_travel) * (ox - cx) + math.sin(my_travel) * (oy - cy)) / d
    # other_ahead: how directly I sit in front of the OTHER's heading.
    other_ahead = -2.0
    if other_travel is not None:
        other_ahead = (math.cos(other_travel) * (cx - ox)
                       + math.sin(other_travel) * (cy - oy)) / d
    EPS = 0.15
    if my_ahead > other_ahead + EPS:
        return True                     # I'm driving into them more -> I stop
    if other_ahead > my_ahead + EPS:
        return False                    # they're driving into me more -> they stop, I go
    # Head-on / ambiguous: strict priority decides. Lower-priority truck stops.
    return not _has_right_of_way(name, c, on, oc)


def _yield_check(name, c, cx, cy, cyaw, travel, dt):
    """Reactive collision avoidance vs the other forklift, evaluated every step.

    Returns (handled, cx, cy, cyaw, speed):
      * handled=False -> no conflict (or WE have the clear path), caller drives normally.
      * handled=True  -> this truck is giving way; it has STOPPED in place (speed 0) and
                         holds until the other clears. It never moves sideways, so it can't
                         be shoved into the other's path.
    Which truck stops is chosen geometrically by `_should_i_stop` (the one blocked by the
    other), so the truck that continues is always the one with the clear lane — it drives
    off and the gap reopens, letting the stopped truck resume automatically.
    """
    c["yielding"] = False
    near = _nearest_other(name, cx, cy)
    if not near:
        return (False, cx, cy, cyaw, 0.0)
    on, ox, oy, oc, dist = near
    if dist >= YIELD_DIST:
        return (False, cx, cy, cyaw, 0.0)

    # Only give way to a truck that is actually MOVING. A parked/idle truck (e.g. sitting
    # on its home charger) is already arced around by the planned route (the bridge punches
    # out its footprint at dispatch), so reactively stopping for it would deadlock forever —
    # the parked truck never clears. Dynamic conflicts are between two navigating trucks.
    if _travel_heading(oc) is None:
        return (False, cx, cy, cyaw, 0.0)

    # Ignore a truck clearly behind us — it isn't in our way.
    to_ox, to_oy = (ox - cx), (oy - cy)
    tod = math.hypot(to_ox, to_oy) or 1.0
    if travel is not None:
        cosang = math.cos(travel) * (to_ox / tod) + math.sin(travel) * (to_oy / tod)
        if cosang < YIELD_BEHIND:
            return (False, cx, cy, cyaw, 0.0)

    if not _should_i_stop(name, c, cx, cy, travel, on, oc):
        return (False, cx, cy, cyaw, 0.0)

    # We are the blocked truck: STOP and hold this pose (no lateral motion). Resume
    # automatically next step once the other has driven clear / out of range.
    c["yielding"] = True
    _set_base_pose(c, cx, cy, cyaw)
    return (True, cx, cy, cyaw, 0.0)


def _step_one(name, c, dt):
    art, idx = c["art"], c["idx"]
    c["_dt"] = dt

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
        c["speed_cmd"] = 0.0
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
        if (REALISTIC_PALLET_PHYSICS and c.get("carrying") and
                c.get("loaded_exit_m", 0.0) < c.get("loaded_exit_goal", 0.0)):
            fwd = cyaw + MODEL_YAW_OFFSET
            remaining = c["loaded_exit_goal"] - c["loaded_exit_m"]
            target_speed = min(
                LOADED_EXIT_SPEED,
                c.get("speed_cmd", 0.0) + LOADED_EXIT_ACCEL * dt,
            )
            step = min(target_speed * dt, remaining)
            cx -= math.cos(fwd) * step
            cy -= math.sin(fwd) * step
            c["loaded_exit_m"] += step
            c["cx"], c["cy"] = cx, cy
            c["speed_cmd"] = step / max(dt, 1e-3)
            speed = c["speed_cmd"]
            _set_base_pose(c, cx, cy, cyaw)
            if c["loaded_exit_m"] >= c["loaded_exit_goal"] - 1e-4:
                c["speed_cmd"] = 0.0
                _log(f"[FleetMind] {name} loaded straight-exit complete "
                     f"({c['loaded_exit_m']:.2f} m)")
            route = [leg.target] if leg.target else []
            _publish(name, c, cx, cy, cyaw, speed, leg.target, route)
            _carry_follow(c, cx, cy, cyaw)
            c["last_xy"] = (cx, cy)
            return cx, cy, cyaw
        if c["wp_i"] >= len(wps):
            _set_base_pose(c, cx, cy, cyaw)
            c["speed_cmd"] = 0.0
            c["pick_insert_m"] = 0.0
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
                c["speed_cmd"] = 0.0
                route = [leg.target] if leg.target else []
                _publish(name, c, cx, cy, cyaw, yspeed, leg.target, route)
                _carry_follow(c, cx, cy, cyaw)
                c["last_xy"] = (cx, cy)
                return cx, cy, cyaw
            is_last = (c["wp_i"] == len(wps) - 1)
            if is_last:
                reach = PICK_ARRIVE if leg.action in ("pick", "drop") else ARRIVE_DIST
            elif REALISTIC_PALLET_PHYSICS and c["carrying"]:
                reach = LOADED_WAYPOINT_DIST
            else:
                reach = WAYPOINT_DIST
            if dist < reach:
                c["wp_i"] += 1
                if REALISTIC_PALLET_PHYSICS and c["carrying"]:
                    c["speed_cmd"] = 0.0
            else:
                # Step the commanded pose toward the waypoint at constant speed (clamped
                # so we never overshoot) and slew heading toward travel so the forks
                # lead. Deterministic: arrival is exact, no orbit/stall/ram/runaway.
                target_yaw = _wrap(travel - MODEL_YAW_OFFSET)
                turn_err = min(abs(_wrap(target_yaw - cyaw)), TURN_SLOW_FULL)
                min_turn_scale = CARRY_TURN_SLOW_MIN if c["carrying"] else TURN_SLOW_MIN
                turn_scale = 1.0 - (1.0 - min_turn_scale) * (turn_err / TURN_SLOW_FULL)

                pick_scale = 1.0
                if leg.action == "pick" and is_last:
                    d = min(max(dist, 0.0), PICK_SLOW_DIST)
                    pick_scale = PICK_SLOW_MIN + (1.0 - PICK_SLOW_MIN) * (d / PICK_SLOW_DIST)

                target_speed = KIN_SPEED * turn_scale * pick_scale
                if REALISTIC_PALLET_PHYSICS and c["carrying"]:
                    target_speed *= CARRY_SPEED_SCALE
                    prev_speed = c.get("speed_cmd", 0.0)
                    target_speed = min(target_speed, prev_speed + CARRY_ACCEL * dt)
                step = min(target_speed * dt, dist)
                cx += step * math.cos(travel)
                cy += step * math.sin(travel)
                turn_rate = CARRY_TURN_RATE if (REALISTIC_PALLET_PHYSICS and c["carrying"]) else TURN_RATE
                cyaw = _slew(cyaw, target_yaw, turn_rate * dt)
                c["cx"], c["cy"], c["cyaw"] = cx, cy, cyaw
                _set_base_pose(c, cx, cy, cyaw)
                speed = step / max(dt, 1e-3)
                c["speed_cmd"] = speed
            if leg.action == "pick":
                # Keep the forks down on the inbound pick approach so the truck slides
                # in under the pallet instead of approaching with raised forks.
                c["lift"] = max(c["lift_down"], c["lift"] - LIFT_RATE)
                _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
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
            # Pick sequence: (1) force forks fully down, (2) creep forward so forks
            # insert into the pallet, (3) engage, then (4) raise the forks.
            if c["pick_engaged"] is None:
                c["lift"] = max(c["lift_down"], c["lift"] - LIFT_RATE)
                _apply_cmd(art, idx, 0.0, 0.0, c["lift"])

            if at_target and c["carrying"] is None and c["pick_engaged"] is None:
                p = _BUS.pallets.get(leg.target, {}) if leg.target else {}
                px = p.get("x")
                py = p.get("y")

                inserted = False
                fx, fy = _fork_xy(cx, cy, cyaw)
                if px is not None and py is not None:
                    inserted = (math.hypot(float(px) - fx, float(py) - fy) <= PICK_INSERT_TOL)

                if not inserted and c["pick_insert_m"] < PICK_INSERT_MAX:
                    fwd = cyaw + MODEL_YAW_OFFSET
                    step = min(PICK_INSERT_SPEED * dt, PICK_INSERT_MAX - c["pick_insert_m"])
                    cx += math.cos(fwd) * step
                    cy += math.sin(fwd) * step
                    c["pick_insert_m"] += step
                    c["cx"], c["cy"] = cx, cy
                    _set_base_pose(c, cx, cy, cyaw)
                    fx, fy = _fork_xy(cx, cy, cyaw)
                    if px is not None and py is not None:
                        inserted = (math.hypot(float(px) - fx, float(py) - fy) <= PICK_INSERT_TOL)

                # If pallet coords are unavailable, still complete the sequence after
                # a bounded insertion creep so the truck does not stall indefinitely.
                if (px is None or py is None) and c["pick_insert_m"] >= (0.5 * PICK_INSERT_MAX):
                    inserted = True
                elif c["pick_insert_m"] >= PICK_INSERT_MAX:
                    inserted = True
                    gap = math.hypot(float(px) - fx, float(py) - fy) if px is not None and py is not None else float("nan")
                    _log(f"[FleetMind] {name} forcing pick engagement for {leg.target} "
                         f"after max insertion (fork_gap={gap:.2f} m)")

                if inserted:
                    c["carry_path"] = leg.pallet_path or _PALLET_PATH.get(leg.target)
                    c["pick_engaged"] = leg.target
                    if c["carry_path"] and not REALISTIC_PALLET_PHYSICS:
                        _set_pallet_kinematic(_RT["stage"], c["carry_path"], True)
                    _log(f"[FleetMind] {name} ENGAGED {leg.target} (fork inserted)")
            if c["pick_engaged"] is not None:
                c["lift"] = min(c["lift_up"], c["lift"] + LIFT_RATE)
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            _carry_follow(c, cx, cy, cyaw)
            if REALISTIC_PALLET_PHYSICS and c["pick_engaged"] and c["carry_path"]:
                pose = _prim_world_xyz(_RT["stage"], c["carry_path"])
                lift_complete = c["lift"] >= c["lift_up"] - 1e-3
                lift_timed_out = lift_complete and c["k"] >= PICK_LIFT_TIMEOUT_STEPS
                if pose and pose[2] >= PALLET_LIFTED_Z:
                    c["carrying"] = c["pick_engaged"]
                    _BUS.set_pallet(c["carrying"], carried_by=name)
                elif lift_timed_out:
                    c["carrying"] = c["pick_engaged"]
                    _BUS.set_pallet(c["carrying"], carried_by=name)
                    z = pose[2] if pose else float("nan")
                    _log(f"[FleetMind] {name} continuing after lift timeout for "
                         f"{c['carrying']} (pallet_z={z:.3f}, lift={c['lift']:.3f})")
            elif (not REALISTIC_PALLET_PHYSICS) and c["pick_engaged"]:
                c["carrying"] = c["pick_engaged"]
                _BUS.set_pallet(c["carrying"], carried_by=name)

            can_advance = (c["carrying"] is not None) or (not REALISTIC_PALLET_PHYSICS)
            if c["k"] >= ACT_STEPS and can_advance and c["lift"] >= c["lift_up"] - 1e-3:
                if REALISTIC_PALLET_PHYSICS and c["carrying"]:
                    c["loaded_exit_m"] = 0.0
                    c["loaded_exit_goal"] = min(
                        LOADED_EXIT_MAX,
                        max(LOADED_EXIT_MIN, c.get("pick_insert_m", 0.0) + LOADED_EXIT_EXTRA),
                    )
                    c["speed_cmd"] = 0.0
                    _log(f"[FleetMind] {name} loaded straight-exit armed "
                         f"({c['loaded_exit_goal']:.2f} m) before route following")
                c["pick_engaged"] = None
                _advance_leg(c)
        elif leg.action == "drop":
            # Lower the fork first so the carried pallet descends with it, then release
            # it onto the staging cell, reverse to disengage the forks, then mark it
            # delivered — a smooth set-down rather than a snap-and-teleport.
            drop_done = False
            if c.get("drop_pending") is not None:
                if c["drop_backoff_m"] < DROP_BACKOFF_DIST:
                    fwd = cyaw + MODEL_YAW_OFFSET
                    step = min(DROP_BACKOFF_SPEED * dt, DROP_BACKOFF_DIST - c["drop_backoff_m"])
                    cx -= math.cos(fwd) * step
                    cy -= math.sin(fwd) * step
                    c["drop_backoff_m"] += step
                    c["cx"], c["cy"] = cx, cy
                    _set_base_pose(c, cx, cy, cyaw)
                else:
                    dx, dy = c["drop_pending_xy"] if c.get("drop_pending_xy") else (cx, cy)
                    if REALISTIC_PALLET_PHYSICS and c.get("drop_pending_path"):
                        pose = _prim_world_xyz(_RT["stage"], c["drop_pending_path"])
                        if pose:
                            dx, dy = pose[0], pose[1]
                    _BUS.set_pallet(c["drop_pending"], x=dx, y=dy,
                                    carried_by=None, delivered=True)
                    _log(f"[FleetMind] {name} DROPPED {c['drop_pending']} at ({dx:+.2f},{dy:+.2f})")
                    c["drop_pending"] = None
                    c["drop_pending_xy"] = None
                    c["drop_pending_path"] = None
                    c["drop_backoff_m"] = 0.0
                    drop_done = True
            elif at_target and c["carrying"] is not None:
                c["lift"] = max(c["lift_down"], c["lift"] - LIFT_RATE)
            _apply_cmd(art, idx, 0.0, 0.0, c["lift"])
            _carry_follow(c, cx, cy, cyaw)
            if (c.get("drop_pending") is None and at_target and
                    c["lift"] <= c["lift_down"] + 1e-3 and c["carrying"] is not None):
                if REALISTIC_PALLET_PHYSICS and c["carry_path"]:
                    pose = _prim_world_xyz(_RT["stage"], c["carry_path"])
                    if pose:
                        dx, dy = pose[0], pose[1]
                    else:
                        dx, dy = _fork_xy(cx, cy, cyaw)
                    c["drop_pending"] = c["carrying"]
                    c["drop_pending_xy"] = (dx, dy)
                    c["drop_pending_path"] = c["carry_path"]
                    c["drop_backoff_m"] = 0.0
                    _BUS.set_pallet(c["carrying"], x=dx, y=dy, carried_by=None,
                                    delivered=False)
                    c["carrying"], c["carry_path"] = None, None
                    c["pick_engaged"] = None
                else:
                    dx, dy = leg.drop_xy if leg.drop_xy else _fork_xy(cx, cy, cyaw)
                    dx, dy = _stage_slot(dx, dy)
                    if c["carry_path"]:
                        _move_prim_xy(_RT["stage"], c["carry_path"], dx, dy, PALLET_FLOOR_Z)
                    c["drop_pending"] = c["carrying"]
                    c["drop_pending_xy"] = (dx, dy)
                    c["drop_pending_path"] = c["carry_path"]
                    c["drop_backoff_m"] = 0.0
                    _BUS.set_pallet(c["carrying"], x=dx, y=dy, carried_by=None, delivered=False)
                    c["carrying"], c["carry_path"] = None, None
            if c["k"] >= ACT_STEPS and c["lift"] <= c["lift_down"] + 1e-3 and drop_done:
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
    c["drop_pending"] = None
    c["drop_pending_xy"] = None
    c["drop_pending_path"] = None
    c["drop_backoff_m"] = 0.0
    c["pick_engaged"] = None
    c["pick_insert_m"] = 0.0
    c["k"] = 0
    if c["leg_i"] >= len(c["legs"]):
        c["phase"] = "idle"
        c["legs"] = []
        c["leg_i"] = 0
        c["loaded_exit_m"] = 0.0
        c["loaded_exit_goal"] = 0.0
        c["speed_cmd"] = 0.0
    else:
        c["phase"] = "navigate"


def _fork_xy(x, y, yaw):
    """World XY of the fork tip: FORK_REACH ahead of the truck centre along its
    travel/forward heading (root_yaw + MODEL_YAW_OFFSET == travel direction)."""
    fwd = yaw + MODEL_YAW_OFFSET
    return x + math.cos(fwd) * FORK_REACH, y + math.sin(fwd) * FORK_REACH


# Placement slots (metres, relative to a bay-pad centre) so several pallets staged at one
# bay fan into a tidy cluster instead of landing on top of each other.
_STAGE_SLOTS = [(0.0, 0.0), (0.62, 0.32), (-0.62, 0.32), (0.62, -0.32), (-0.62, -0.32)]


def _stage_slot(x, y):
    """Offset a drop point by how many pallets are already staged on that pad, so a
    re-routed second load sits beside the first rather than clipping into it."""
    n = sum(1 for p in _BUS.pallets.values()
            if p.get("delivered") and math.hypot(p["x"] - x, p["y"] - y) < 1.5)
    ox, oy = _STAGE_SLOTS[n % len(_STAGE_SLOTS)]
    return x + ox, y + oy


def _carry_follow(c, x, y, yaw):
    if not c.get("carry_path"):
        return
    if REALISTIC_PALLET_PHYSICS:
        pose = _prim_world_xyz(_RT["stage"], c["carry_path"])
        if pose and (c.get("carrying") or c.get("pick_engaged")):
            _BUS.set_pallet(c.get("carrying") or c.get("pick_engaged"),
                            x=pose[0], y=pose[1],
                            carried_by=c.get("name") if c.get("carrying") else None)
        return
    if c["carrying"] or c.get("pick_engaged"):
        fx, fy = _fork_xy(x, y, yaw)
        pz = c["lift"] + FORK_CARRY_Z_OFFSET
        _move_prim_xy(_RT["stage"], c["carry_path"], fx, fy, pz)


def _apply_reset():
    """Controller half of a between-demo reset: teleport every pallet back to its rack
    cell and return each forklift to its home charger with a full battery. The bridge has
    already reset the shared bus; this restores the USD prims so the 3D scene visibly
    returns to its start state with no Isaac relaunch (the live stream stays up)."""
    stage = _RT["stage"]
    for i, (x, y) in enumerate(PALLETS):
        pp = f"/World/Pallets/Pallet_{i:02d}"
        _set_rigid_body_enabled(stage, pp, True)
        _set_collision_enabled(stage, pp, True)
        _set_pallet_kinematic(stage, pp, True)
        _set_rigid_body_velocities(stage, pp)
        _set_pose(stage, pp, (x, y, PALLET_FLOOR_Z), 0.0,
                  scale=_RT.get("pallet_scale", 1.0))
        _set_rigid_body_velocities(stage, pp)
    _RT["pallet_reset_rearm_steps"] = 3 if REALISTIC_PALLET_PHYSICS else 0
    _RT["reset_hold_steps"] = 4
    _log_pallet_physics_state(stage, "reset-staged")
    for name, c in _RT.get("ctrls", {}).items():
        hx, hy = c["home"]
        hyaw = c["home_yaw"]
        c["phase"] = "idle"
        c["cx"], c["cy"], c["cyaw"] = hx, hy, hyaw
        c["last_xy"] = None
        c["lift"] = c["lift_down"]
        c["carrying"] = None
        c["carry_path"] = None
        c["drop_pending"] = None
        c["drop_pending_xy"] = None
        c["drop_pending_path"] = None
        c["drop_backoff_m"] = 0.0
        c["pick_engaged"] = None
        c["pick_insert_m"] = 0.0
        c["loaded_exit_m"] = 0.0
        c["loaded_exit_goal"] = 0.0
        c["speed_cmd"] = 0.0
        c["legs"] = []
        c["leg_i"] = 0
        c["wp_i"] = 0
        c["k"] = 0
        c["battery"] = BATTERY_FULL
        # Sync our command cursor so the just-cleared bus mission is not re-adopted.
        c["seq"] = _BUS.get_command(name).seq
        _reset_forklift_articulation(c)
        _publish(name, c, hx, hy, hyaw, 0.0, None, [])
        _update_route_overlay(stage, name, [])
    _log("[FleetMind] Scene reset: pallets restocked, forklifts home, batteries full.")


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


def _active_route_node_ids(c, x, y):
    """Convert the remaining world-space route into roadmap node ids for the 2D map.

    The UI draws the forklift's current position as the first point, so we only publish
    the not-yet-reached waypoints here. This keeps the live line focused on the mission's
    full remaining path (e.g. forklift -> pallet -> stage) instead of just the current
    leg target or a stale home leg."""
    graph = _BUS.graph or {}
    nodes = graph.get("nodes", {})
    if not nodes:
        return []
    pts = _active_route_pts(c, x, y)
    if len(pts) < 2:
        return []

    def nearest_node(px, py):
        best = None
        best_d = float("inf")
        for nid, (nx, ny) in nodes.items():
            d = math.hypot(float(nx) - px, float(ny) - py)
            if d < best_d:
                best = nid
                best_d = d
        return best

    route = []
    for px, py in pts[1:]:
        nid = nearest_node(float(px), float(py))
        if not nid:
            continue
        if not route or route[-1] != nid:
            route.append(nid)
    return route


def _on_step(dt):
    _RT["warm"] += 1
    if _RT["warm"] < WARMUP_STEPS:
        return
    if "ctrls" not in _RT:
        _init_controllers()
        return
    # Between-demo reset: when the bridge bumps the bus epoch, snap every prim back to its
    # spawn pose (pallets on racks, forklifts home, batteries full) without an Isaac relaunch.
    ep = getattr(_BUS, "reset_epoch", 0)
    if _RT.get("reset_epoch", 0) != ep:
        _RT["reset_epoch"] = ep
        _apply_reset()
        return
    if REALISTIC_PALLET_PHYSICS and _RT.get("pallet_reset_rearm_steps", 0) > 0:
        _RT["pallet_reset_rearm_steps"] -= 1
        if _RT["pallet_reset_rearm_steps"] == 0:
            _rearm_reset_pallet_physics(_RT["stage"])
    if _RT.get("reset_hold_steps", 0) > 0:
        _RT["reset_hold_steps"] -= 1
        for name, c in _RT["ctrls"].items():
            hx, hy = c["home"]
            hyaw = c["home_yaw"]
            _reset_forklift_articulation(c)
            _publish(name, c, hx, hy, hyaw, 0.0, None, [])
            _update_follow_cam(_RT["stage"], name, hx, hy, hyaw)
            _update_route_overlay(_RT["stage"], name, [])
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
