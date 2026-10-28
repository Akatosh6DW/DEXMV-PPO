"""S12 volumetric elastic-cylinder physics, separate from frozen S10 inputs.

The object consists only of a tetrahedral flex: no invisible rigid core, object
actuator, grasp weld, or applied object force. A physical tray retracts before
the common Cartesian hand positioner lifts. Finger targets remain 17-D.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/orca_benchmark_suite/S12"
BASE = ROOT / "artifacts/orca_benchmark_suite/S10/inputs/model.xml"
REFERENCE = ROOT / "artifacts/orca_benchmark_suite/S09/inputs/reference.npz"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CENTER = np.array([.08, .055, .50])


@dataclass(frozen=True)
class SoftConfig:
    mesh_version: str = "conforming_prism_v2"
    young: float = 10000.
    poisson: float = .3
    damping: float = .002
    mass: float = .035
    radius: float = .026
    half_length: float = .026
    contact_radius: float = .0005
    sectors: int = 12
    layers: int = 3
    dof: str = "full"
    timestep: float = .001
    control_dt: float = .05
    approach_sec: float = 2.
    release_sec: float = 5.
    lift_start: float = 6.
    lift_sec: float = 2.
    lift_m: float = .05
    episode_sec: float = 24.
    hold_sec: float = 15.
    max_strain: float = .30
    max_penetration: float = .003
    max_force: float = 30.
    hand_offset_z: float = 0.


def numbers(a):
    return " ".join(format(float(x), ".12g") for x in np.asarray(a).ravel())


def tetra_cylinder(cfg):
    """Conforming staircase triangulation of disk triangles extruded in Z.

All adjacent prisms use the same global vertex ordering, hence the same face
diagonal. Generic Delaunay on this cospherical lattice followed by deletion of
zero-volume cells is NOT conforming and can leave artificial internal surfaces.
"""
    if cfg.mesh_version != "conforming_prism_v2":
        raise ValueError("Unsupported mesh version")
    theta = np.arange(cfg.sectors)*2*np.pi/cfg.sectors
    ring = np.c_[cfg.radius*np.cos(theta), cfg.radius*np.sin(theta)]
    disk = np.vstack(([0., 0.], ring))
    points = np.concatenate([np.c_[disk, np.full(len(disk), z)]
                             for z in np.linspace(-cfg.half_length, cfg.half_length, cfg.layers)])
    width = cfg.sectors+1
    tetra = []
    for layer in range(cfg.layers-1):
        for sector in range(cfg.sectors):
            a, b, c = np.sort([0, sector+1, (sector+1)%cfg.sectors+1])+layer*width
            tetra.extend([(a,b,c,c+width), (a,b,b+width,c+width),
                          (a,a+width,b+width,c+width)])
    return points, np.asarray(tetra, dtype=int)


def add_flex(parent, cfg, points, tetra):
    flex = ET.SubElement(parent, "flexcomp", name="s12_object", type="direct", dim="3",
                         dof=cfg.dof, point=numbers(points), element=" ".join(map(str, tetra.ravel())),
                         mass=str(cfg.mass), radius=str(cfg.contact_radius), rgba=".95 .40 .15 1")
    ET.SubElement(flex, "contact", contype="2", conaffinity="5", condim="3",
                  friction="1.35 .005 .001", solref=".004 1", solimp=".95 .99 .001",
                  selfcollide="auto" if cfg.dof == "full" else "none",
                  internal="true" if cfg.dof == "full" else "false")
    ET.SubElement(flex, "elasticity", young=str(cfg.young), poisson=str(cfg.poisson),
                  damping=str(cfg.damping))
    ET.SubElement(flex, "edge", equality="false", damping="0")
    return flex


def scene_xml(cfg=SoftConfig()):
    root = ET.parse(BASE).getroot()
    root.set("model", "S12 volumetric soft-cylinder grasp and lift")
    root.find("option").set("timestep", str(cfg.timestep))
    root.find("option").set("jacobian", "sparse")
    soft_default = ET.SubElement(root.find("default"), "default", {"class":"s12_soft"})
    ET.SubElement(soft_default, "joint", limited="false", damping="0", frictionloss="0", armature="0")
    world = root.find("worldbody")
    for name in ("cube", "cube_support"):
        world.remove(world.find(f"body[@name='{name}']"))
    for tag in ("tendon", "keyframe"):
        for elem in root.findall(tag):
            root.remove(elem)
    with np.load(REFERENCE) as bank:
        object_rot = Rotation.from_quat(bank["object_quaternion"][0][[1, 2, 3, 0]])
    geom_rot = Rotation.from_rotvec([np.pi/2, 0, 0])
    points, tetra = tetra_cylinder(cfg)
    points = (object_rot*geom_rot).apply(points)
    body = ET.SubElement(world, "body", name="s12_frame", pos=numbers(CENTER), childclass="s12_soft")
    add_flex(body, cfg, points, tetra)
    z = CENTER[2]+points[:, 2].min()-cfg.contact_radius-.002
    tray = ET.SubElement(world, "body", name="s12_tray", mocap="true", pos=numbers([*CENTER[:2], z]))
    ET.SubElement(tray, "geom", name="s12_tray_geom", type="box", size=".018 .018 .002",
                  contype="4", conaffinity="2", rgba=".25 .35 .4 1")
    return ET.tostring(root, encoding="unicode")


def tetra_metrics(vertices, elements, rest_inverse):
    edges = (vertices[elements[:, 1:]]-vertices[elements[:, :1]]).transpose(0, 2, 1)
    gradient = edges @ rest_inverse
    stretch = np.linalg.svd(gradient, compute_uv=False)
    determinant = np.linalg.det(gradient)
    return float(np.max(np.abs(stretch-1))), float(determinant.min()), float(determinant.mean())


class SoftGrasp:
    def __init__(self, cfg=SoftConfig(), xml=None):
        self.cfg = cfg
        self.xml = scene_xml(cfg) if xml is None else xml
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qids = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.dqids = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.low, self.high = m.actuator_ctrlrange.T.copy()
        names = [m.geom(i).name or "" for i in range(m.ngeom)]
        self.geom_finger = np.array([next((j for j, f in enumerate(FINGERS)
                                          if name.startswith(f"orca_{f}_")), -1) for name in names])
        self.geom_pad = np.array(["_tip_" in name or "_prox_" in name for name in names])
        self.geom_palm = np.array([name.startswith("orca_palm_") for name in names])
        self.tray_geom = m.geom("s12_tray_geom").id
        self.flex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, "s12_object")
        self.vid = int(m.flex_vertadr[self.flex])
        self.nvert = int(m.flex_vertnum[self.flex])
        self.vertex_slice = slice(self.vid, self.vid+self.nvert)
        self.soft_bodies = m.flex_vertbodyid[self.vertex_slice].copy()
        adr, n = int(m.flex_elemdataadr[self.flex]), int(m.flex_elemnum[self.flex])
        self.elements = m.flex_elem[adr:adr+4*n].reshape(-1, 4).copy()
        self.target_id = int(m.body_mocapid[m.body("s10_target").id])
        self.tray_id = int(m.body_mocapid[m.body("s12_tray").id])
        mount = m.body("right_mount").id
        self.original_pos = m.body_pos[mount].copy()
        self.original_quat = m.body_quat[mount].copy()
        self.rid = int(m.joint("s10_hand_free").qposadr[0])
        self.nsub = round(cfg.control_dt/cfg.timestep)
        if not np.isclose(self.nsub*cfg.timestep, cfg.control_dt):
            raise ValueError("Nonintegral substep count")
        with np.load(REFERENCE) as z:
            self.bank = {k:z[k].copy() for k in z.files}
        mujoco.mj_forward(m, self.data)
        rest = self.vertices.copy()
        rest_edges = (rest[self.elements[:, 1:]]-rest[self.elements[:, :1]]).transpose(0, 2, 1)
        self.rest_inverse = np.linalg.inv(rest_edges)
        self.tray_home = m.body_pos[m.body("s12_tray").id].copy()
        self.reset()

    @property
    def vertices(self):
        return self.data.flexvert_xpos[self.vertex_slice]

    def reset(self, seed=0, jitter=0., grip_scale=1., offset=(0., 0., 0.)):
        mujoco.mj_resetData(self.model, self.data)
        self.offset = (np.asarray(offset, float)+[0., 0., self.cfg.hand_offset_z]
                       +np.random.default_rng(seed).uniform(-jitter, jitter, 3))
        self.grip_scale = grip_scale
        self.home = self.original_pos + [0, -.20, 0] + self.offset
        self.goal = self.original_pos + self.offset
        self.data.qpos[self.rid:self.rid+7] = np.r_[self.home, self.original_quat]
        self.data.qpos[self.qids] = 0.
        self.data.ctrl[:] = 0.
        self.data.mocap_pos[self.target_id] = self.home
        self.data.mocap_quat[self.target_id] = self.original_quat
        self.data.mocap_pos[self.tray_id] = self.tray_home
        mujoco.mj_forward(self.model, self.data)
        self.hold = self.best = 0.
        self.max_strain = self.max_depth = self.peak_force = 0.
        self.ever_invalid = False
        return self.audit()

    def reference(self, time):
        phase = max(0., time-self.cfg.approach_sec)
        return self.grip_scale*np.array([np.interp(phase, self.bank["time"], self.bank["ctrl"][0, :, j])
                                         for j in range(17)])

    def targets(self, time):
        cfg = self.cfg
        u = np.clip(time/cfg.approach_sec, 0., 1.)
        p = self.home+(u*u*(3-2*u))*(self.goal-self.home)
        v = np.clip((time-cfg.lift_start)/cfg.lift_sec, 0., 1.)
        p = p + [0, 0, cfg.lift_m*v*v*(3-2*v)]
        self.data.mocap_pos[self.target_id] = p
        self.data.mocap_quat[self.target_id] = self.original_quat
        self.data.mocap_pos[self.tray_id] = self.tray_home-[0, 0, .15*np.clip((time-cfg.release_sec)/.5, 0, 1)]

    def audit(self, fast=True):
        m, d = self.model, self.data
        loads = np.zeros(5)
        peak = depth = support = forbidden = 0.
        contacts = d.contact[:d.ncon]
        # Elliptic contacts put normal force in the first constraint row. This
        # is identical to mj_contactForce(...)[0], checked against the slow path.
        if fast and d.ncon:
            selected = np.flatnonzero(np.any(contacts.flex == self.flex, axis=1) & (contacts.dist <= 0))
            if len(selected):
                adr = contacts.efc_address[selected]
                force = np.zeros(len(selected))
                active = adr >= 0
                force[active] = d.efc_force[adr[active]]
                peak = max(0., float(force.max()))
                depth = max(0., float(-contacts.dist[selected].min()))
                side = (contacts.flex[selected, 0] != self.flex).astype(int)
                gids = contacts.geom[selected, 1-side]
                is_geom = gids >= 0
                gids, force = gids[is_geom], force[is_geom]
                selected, side = selected[is_geom], side[is_geom]
                fingers = self.geom_finger[gids]
                tray = gids == self.tray_geom
                support = float(force[tray].sum())
                normal = contacts.frame[selected, :3] * np.where(side[:, None] == 1, 1., -1.)
                local = np.einsum("ni,nij->nj", normal, d.xmat[m.geom_bodyid[gids]].reshape(-1, 3, 3))
                pad = (fingers>=0)&self.geom_pad[gids]&(-local[:, 0]>=.4)&~tray
                loads = np.bincount(fingers[pad], weights=force[pad], minlength=5)
                bad = ~tray & (((fingers>=0)&~pad&(force>.02)) | ((fingers<0)&~self.geom_palm[gids]))
                forbidden = float(force[bad].sum())
        for i, c in enumerate(contacts if not fast else []):
            if self.flex not in c.flex or c.dist > 0:
                continue
            side = 0 if c.flex[0] == self.flex else 1
            gid = int(c.geom[1-side])
            depth = max(depth, -float(c.dist))
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            peak = max(peak, force[0])
            if gid < 0:
                continue
            name = m.geom(gid).name or ""
            finger = next((j for j, f in enumerate(FINGERS) if name.startswith(f"orca_{f}_")), None)
            if name == "s12_tray_geom":
                support += force[0]
            elif finger is not None:
                normal = c.frame[:3] * (1 if side == 1 else -1)
                local = normal @ d.xmat[m.geom_bodyid[gid]].reshape(3, 3)
                if ("_tip_" in name or "_prox_" in name) and -local[0] >= .4:
                    loads[finger] += force[0]
                elif force[0] > .02:
                    forbidden += force[0]
            elif not name.startswith("orca_palm_"):
                forbidden += force[0]
        strain, min_j, mean_j = tetra_metrics(self.vertices, self.elements, self.rest_inverse)
        return dict(time=float(d.time), loads=loads, peak_force_n=float(peak), penetration_m=depth,
                    support_force_n=float(support), forbidden_force_n=float(forbidden),
                    max_principal_strain=strain, min_volume_ratio=min_j, mean_volume_ratio=mean_j,
                    center=self.vertices.mean(0).copy(), bottom_m=float(self.vertices[:, 2].min()))

    def step(self, desired):
        d, cfg = self.data, self.cfg
        d.ctrl[:] = np.clip(np.clip(desired, self.low, self.high), d.ctrl-.1, d.ctrl+.1)
        peak = depth = support = forbidden = strain = 0.
        min_j = np.inf
        loads = np.full(5, np.inf)
        for _ in range(self.nsub):
            self.targets(d.time+cfg.timestep)
            mujoco.mj_step(self.model, d)
            # Audit the contact loads and deformation throughout each control step.
            row = self.audit()
            loads = np.minimum(loads, row["loads"])
            peak = max(peak, row["peak_force_n"])
            depth = max(depth, row["penetration_m"])
            support = max(support, row["support_force_n"])
            forbidden = max(forbidden, row["forbidden_force_n"])
            strain = max(strain, row["max_principal_strain"])
            min_j = min(min_j, row["min_volume_ratio"])
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError("S12 simulation warning or nonfinite state")
        row.update(max_principal_strain=strain, min_volume_ratio=min_j)
        released = d.time >= cfg.release_sec+.5
        lifted = row["center"][2] >= CENTER[2]+.035
        alive = row["center"][2] > CENTER[2]-.10 and row["min_volume_ratio"]>.05
        paired = loads[0]>.02 and np.any(loads[1:]>.02)
        quality = (row["max_principal_strain"]<=cfg.max_strain and depth<=cfg.max_penetration
                   and peak<=cfg.max_force and forbidden<.02)
        valid = released and lifted and alive and paired and quality and support<1e-5
        self.hold = self.hold+cfg.control_dt if valid else 0.
        self.best = max(self.best, self.hold)
        self.max_strain = max(self.max_strain, row["max_principal_strain"])
        self.max_depth = max(self.max_depth, depth)
        self.peak_force = max(self.peak_force, peak)
        self.ever_invalid |= (released and not quality)
        row.update(loads=loads, peak_force_n=peak, penetration_m=depth, support_force_n=support,
                   forbidden_force_n=forbidden, alive=bool(alive), valid=bool(valid), lifted=bool(lifted),
                   released=bool(released), hold_sec=self.hold, best_hold_sec=self.best,
                   success15=bool(self.hold>=cfg.hold_sec-1e-8 and not self.ever_invalid))
        return row


def run_reference(sim, seed=0, seconds=None, grip_scale=1., jitter=0., offset=(0, 0, 0)):
    sim.reset(seed, jitter=jitter, grip_scale=grip_scale, offset=offset)
    rows = []
    for _ in range(round((seconds or sim.cfg.episode_sec)/sim.cfg.control_dt)):
        row = sim.step(sim.reference(sim.data.time+sim.cfg.control_dt))
        row.update(qpos=sim.data.qpos.copy(), qvel=sim.data.qvel.copy(), ctrl=sim.data.ctrl.copy(),
                   mocap=sim.data.mocap_pos.copy(), mocap_quat=sim.data.mocap_quat.copy(),
                   vertices=sim.vertices.copy())
        rows.append(row)
        if not row["alive"]:
            break
    trace = {k:np.array([r[k] for r in rows]) for k in rows[0]}
    report = dict(config=asdict(sim.cfg), seed=seed, grip_scale=grip_scale, offset=list(offset),
                  duration_sec=float(sim.data.time), success15=rows[-1]["success15"],
                  best_hold_sec=sim.best, final_center=rows[-1]["center"].tolist(),
                  max_principal_strain=sim.max_strain, max_penetration_m=sim.max_depth,
                  peak_force_n=sim.peak_force, min_volume_ratio=float(trace["min_volume_ratio"].min()),
                  final_loads=rows[-1]["loads"].tolist(), final_support_force_n=rows[-1]["support_force_n"])
    return report, trace
