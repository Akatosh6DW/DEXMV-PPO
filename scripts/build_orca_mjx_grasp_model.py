#!/usr/bin/env python3
"""Build an MJX-friendly ORCA cube-grasp model with proxy collisions."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ORCA_V2 = (
    ROOT
    / "Aero-Hand"
    / "sim_rl"
    / "mujoco_playground"
    / "mujoco_playground"
    / "_src"
    / "manipulation"
    / "orca_hand"
    / "xmls"
    / "orcahand_description"
    / "v2"
)
MJCF_DIR = ORCA_V2 / "models" / "mjcf"
ASSET_DIR = ORCA_V2 / "models" / "assets"


def _remove_mesh_geoms(root: ET.Element) -> None:
  for parent in root.iter():
    for child in list(parent):
      if child.tag == "geom":
        parent.remove(child)


def _find_body(root: ET.Element, name: str) -> ET.Element:
  for body in root.iter("body"):
    if body.get("name") == name:
      return body
  raise KeyError(f"Body not found: {name}")


def _add_geom(body: ET.Element, **attrib: str) -> None:
  ET.SubElement(body, "geom", attrib)


def _proxy_capsule(
    root: ET.Element,
    body_name: str,
    geom_name: str,
    fromto: str,
    size: float,
    rgba: str,
) -> None:
  _add_geom(
      _find_body(root, body_name),
      name=geom_name,
      type="capsule",
      fromto=fromto,
      size=f"{size:.4f}",
      contype="1",
      conaffinity="2",
      condim="4",
      friction="1.4 0.02 0.003",
      group="3",
      rgba=rgba,
  )


def _proxy_sphere(
    root: ET.Element,
    body_name: str,
    geom_name: str,
    pos: str,
    size: float,
    rgba: str,
) -> None:
  _add_geom(
      _find_body(root, body_name),
      name=geom_name,
      type="sphere",
      pos=pos,
      size=f"{size:.4f}",
      contype="1",
      conaffinity="2",
      condim="4",
      friction="1.6 0.025 0.004",
      group="3",
      rgba=rgba,
  )


def _proxy_box(
    root: ET.Element,
    body_name: str,
    geom_name: str,
    pos: str,
    size: str,
    rgba: str,
    *,
    contype: str = "1",
    conaffinity: str = "2",
) -> None:
  _add_geom(
      _find_body(root, body_name),
      name=geom_name,
      type="box",
      pos=pos,
      size=size,
      contype=contype,
      conaffinity=conaffinity,
      condim="4",
      friction="1.2 0.02 0.003",
      group="3",
      rgba=rgba,
  )


def build_body() -> None:
  tree = ET.parse(MJCF_DIR / "orcahand_right_body.xml")
  root = tree.getroot()
  _remove_mesh_geoms(root)

  # Palm/wrist proxy. Keep it slightly small so finger contacts dominate.
  _proxy_box(
      root,
      "right_R-Carpals_8d1f1041",
      "orca_palm_proxy",
      pos="0.004 0.000 0.022",
      size="0.034 0.020 0.024",
      rgba="0.10 0.35 0.95 0.42",
  )

  # Index.
  _proxy_sphere(root, "right_I-AP-R_d95d02d1", "orca_index_base_proxy", "0.0188 0 0.001", 0.0070, "0.15 0.45 1 0.50")
  _proxy_capsule(root, "right_I-PP_bacbd481", "orca_index_prox_proxy", "0 0 0 0.0015 0 0.034", 0.0070, "0.15 0.45 1 0.55")
  _proxy_capsule(root, "right_I-FingerTipAssembly_ec49c16c", "orca_index_tip_proxy", "0 0 0 0 0 0.028", 0.0075, "0.15 0.45 1 0.78")
  _proxy_sphere(root, "right_I-FingerTipAssembly_ec49c16c", "orca_index_tip_ball_proxy", "0 0 0.030", 0.0088, "0.15 0.45 1 0.88")

  # Middle.
  _proxy_sphere(root, "right_M-AP_e04a96f2", "orca_middle_base_proxy", "0.0193 0 0.001", 0.0074, "0.15 0.45 1 0.50")
  _proxy_capsule(root, "right_M-PP_08efa608", "orca_middle_prox_proxy", "0 0 0 0.002 0 0.037", 0.0075, "0.15 0.45 1 0.55")
  _proxy_capsule(root, "right_M-FingerTipAssembly_34afb748", "orca_middle_tip_proxy", "0 0 0 0 0 0.030", 0.0080, "0.15 0.45 1 0.78")
  _proxy_sphere(root, "right_M-FingerTipAssembly_34afb748", "orca_middle_tip_ball_proxy", "0 0 0.032", 0.0090, "0.15 0.45 1 0.88")

  # Ring.
  _proxy_sphere(root, "right_M-AP_6ec59111", "orca_ring_base_proxy", "0.0193 0 0.001", 0.0074, "0.15 0.45 1 0.50")
  _proxy_capsule(root, "right_M-PP_8660a1eb", "orca_ring_prox_proxy", "0 0 0 0.002 0 0.037", 0.0075, "0.15 0.45 1 0.55")
  _proxy_capsule(root, "right_M-FingerTipAssembly_424a8e75", "orca_ring_tip_proxy", "0 0 0 0 0 0.030", 0.0080, "0.15 0.45 1 0.78")
  _proxy_sphere(root, "right_M-FingerTipAssembly_424a8e75", "orca_ring_tip_ball_proxy", "0 0 0.032", 0.0090, "0.15 0.45 1 0.88")

  # Pinky.
  _proxy_sphere(root, "right_P-AP_f5e42b61", "orca_pinky_base_proxy", "0.0178 0 0.001", 0.0068, "0.15 0.45 1 0.50")
  _proxy_capsule(root, "right_P-PP_1d411b9b", "orca_pinky_prox_proxy", "0 0 0 0.001 0 0.029", 0.0068, "0.15 0.45 1 0.55")
  _proxy_capsule(root, "right_P-FingerTipAssembly_cd219176", "orca_pinky_tip_proxy", "0 0 0 0 0 0.024", 0.0072, "0.15 0.45 1 0.78")
  _proxy_sphere(root, "right_P-FingerTipAssembly_cd219176", "orca_pinky_tip_ball_proxy", "0 0 0.026", 0.0080, "0.15 0.45 1 0.88")

  # Thumb.
  _proxy_capsule(root, "right_T-TP-R_1c2b802d", "orca_thumb_base_proxy", "0 0 0 0.030 0 0", 0.0082, "0.62 0.15 0.95 0.58")
  _proxy_capsule(root, "right_R-T-AP_a9723101", "orca_thumb_mid_proxy", "0 0 0 0.030 0 0", 0.0080, "0.62 0.15 0.95 0.64")
  _proxy_capsule(root, "right_T-PP_68395e98", "orca_thumb_prox_proxy", "0 0 0 0.001 0 0.030", 0.0082, "0.62 0.15 0.95 0.72")
  _proxy_capsule(root, "right_T-DP_b7429e50", "orca_thumb_tip_proxy", "0 0 0 0 0 0.024", 0.0082, "0.62 0.15 0.95 0.86")
  _proxy_sphere(root, "right_T-DP_b7429e50", "orca_thumb_tip_ball_proxy", "0 0 0.026", 0.0094, "0.62 0.15 0.95 0.94")

  ET.indent(tree, space="  ")
  tree.write(MJCF_DIR / "orcahand_right_mjx_grasp_body.xml", encoding="utf-8", xml_declaration=True)


def build_mjcf() -> None:
  root = ET.Element("mujoco", {"model": "orcahand_right_mjx_grasp"})
  default = ET.SubElement(root, "default")
  ET.SubElement(default, "geom", {
      "condim": "4",
      "friction": "1.4 0.02 0.003",
      "solref": "0.008 1",
      "solimp": "0.92 0.98 0.002",
  })
  ET.SubElement(default, "joint", {
      "type": "hinge",
      "limited": "true",
      "damping": "0.08",
      "armature": "0.001",
      "frictionloss": "0.0005",
  })
  ET.SubElement(default, "position", {
      "ctrllimited": "true",
      "forcelimited": "true",
      "forcerange": "-3 3",
      "kp": "7.0",
      "kv": "0.08",
  })

  source = ET.parse(MJCF_DIR / "orcahand_right.mjcf").getroot()
  actuator = source.find("actuator")
  if actuator is None:
    raise RuntimeError("Missing actuator block in ORCA right MJCF.")
  root.append(actuator)
  ET.indent(root, space="  ")
  ET.ElementTree(root).write(MJCF_DIR / "orcahand_right_mjx_grasp.mjcf", encoding="utf-8", xml_declaration=True)


def build_options() -> None:
  root = ET.Element("mujoco", {"model": "orcahand_mjx_grasp_options"})
  ET.SubElement(root, "compiler", {"angle": "radian", "eulerseq": "XYZ"})
  ET.SubElement(root, "option", {
      "timestep": "0.01",
      "integrator": "implicitfast",
      "cone": "pyramidal",
      "iterations": "8",
      "ls_iterations": "8",
  })
  default = ET.SubElement(root, "default")
  ET.SubElement(default, "geom", {
      "condim": "4",
      "friction": "1.4 0.02 0.003",
      "solref": "0.008 1",
      "solimp": "0.92 0.98 0.002",
  })
  ET.SubElement(default, "joint", {
      "type": "hinge",
      "limited": "true",
      "damping": "0.08",
      "armature": "0.001",
      "frictionloss": "0.0005",
  })
  ET.SubElement(default, "position", {
      "ctrllimited": "true",
      "forcelimited": "true",
      "forcerange": "-3 3",
      "kp": "7.0",
      "kv": "0.08",
  })
  ET.indent(root, space="  ")
  ET.ElementTree(root).write(ASSET_DIR / "options_mjx_grasp.xml", encoding="utf-8", xml_declaration=True)


def build_scene() -> None:
  scene = """<?xml version="1.0" ?>
<mujoco model="ORCA hand right MJX cube grasp scene">
  <include file="models/assets/options_mjx_grasp.xml"/>
  <include file="models/mjcf/orcahand_right_mjx_grasp.mjcf"/>

  <statistic extent="0.35" center="0.08 0.02 0.22"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.60" rgb2="0.04 0.05 0.07" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.18 0.22 0.28" rgb2="0.08 0.10 0.13" width="256" height="256"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.12"/>
    <material name="cube_mat" rgba="0.95 0.35 0.12 1"/>
    <material name="support_mat" rgba="0.20 0.20 0.22 0.85"/>
  </asset>

  <worldbody>
    <light pos="0.1 0.4 1.1" dir="0 -0.3 -1" diffuse="0.85 0.85 0.85"/>
    <camera name="side" pos="0.30 -0.36 0.26" xyaxes="1 0 0 0 0.45 0.89"/>
    <camera name="top" pos="0.08 0.02 0.62" xyaxes="1 0 0 0 1 0"/>
    <camera name="front" pos="0.08 -0.34 0.28" xyaxes="1 0 0 0 0.45 0.89"/>
    <geom name="floor" type="plane" pos="0 0 0" size="0 0 0.05" material="groundplane" contype="4" conaffinity="2" group="0"/>

    <body name="right_mount" pos="0.09363 0.00332 0.058" euler="1.5708 0 0">
      <include file="models/mjcf/orcahand_right_mjx_grasp_body.xml"/>
    </body>

    <body name="cube" pos="0.082 0.026 0.248">
      <freejoint name="cube_freejoint"/>
      <geom name="cube" type="box" size="0.018 0.018 0.018" mass="0.035" material="cube_mat"
            contype="2" conaffinity="5" condim="4" friction="1.35 0.025 0.004" group="1"/>
    </body>

    <body name="cube_support" mocap="true" pos="0.082 0.026 0.226">
      <geom name="cube_support" type="box" size="0.038 0.038 0.004" material="support_mat"
            contype="4" conaffinity="2" condim="4" friction="1.0 0.01 0.002" group="1"/>
    </body>
  </worldbody>
</mujoco>
"""
  (ORCA_V2 / "scene_right_mjx_grasp_cube.xml").write_text(scene)


def main() -> None:
  build_body()
  build_mjcf()
  build_options()
  build_scene()
  print(f"Wrote {ORCA_V2 / 'scene_right_mjx_grasp_cube.xml'}")
  print(f"Wrote {MJCF_DIR / 'orcahand_right_mjx_grasp_body.xml'}")
  print(f"Wrote {MJCF_DIR / 'orcahand_right_mjx_grasp.mjcf'}")
  print(f"Wrote {ASSET_DIR / 'options_mjx_grasp.xml'}")


if __name__ == "__main__":
  main()
