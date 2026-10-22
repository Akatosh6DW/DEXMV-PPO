"""Export episode statistics and inspect saved S05 contacts/control composition."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from s05_pad_contacts import FINGERS, is_distal_pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.run / "run_config.json").read_text())
    config = json.loads((args.run / "env_config.json").read_text())
    directory = args.run / f"step_{args.step:09d}"
    episodes = json.loads((directory / "episodes.json").read_text())
    with (args.output / "episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    with np.load(directory / "preview_trajectories.npz") as archive:
        traces = {k: archive[k] for k in archive.files}
    model = mujoco.MjModel.from_xml_path(meta["model"])
    data = mujoco.MjData(model)
    cube = model.geom("cube").id
    rows, control_errors = [], []
    for episode in range(len(traces["qpos"])):
        trace = {k: v[episode] for k, v in traces.items()}
        if meta["method"] == "residual":
            low, high = model.actuator_ctrlrange.T
            residual = np.clip(trace["action"], -1, 1) * .08
            target = np.clip(trace["video_base"] + residual, low, high)
            previous = np.vstack([config["initial_ctrl"], trace["ctrl"][:-1]])
            ctrl = np.clip(target, previous - config["max_ctrl_delta"],
                           previous + config["max_ctrl_delta"])
            active = trace["active"].astype(bool)
            control_errors.append(dict(
                requested_max_error=float(np.max(np.abs(residual[active] - trace["requested_residual"][active]))),
                ctrl_max_error=float(np.max(np.abs(ctrl[active] - trace["ctrl"][active]))),
                residual_rms_rad=float(np.sqrt(np.mean(residual[active] ** 2))),
                max_residual_rad=float(np.max(np.abs(residual[active])))))
        # Saved-state reconstruction gives interpretable normals/locations;
        # the environment's original substep audit remains the success authority.
        for frame in range(59, len(trace["qpos"]), 2):
            if not trace["active"][frame]:
                continue
            for key in ("qpos", "qvel", "ctrl"):
                getattr(data, key)[:] = trace[key][frame]
            data.mocap_pos[:] = trace["mocap"][frame]
            mujoco.mj_forward(model, data)
            for ci, contact in enumerate(data.contact[:data.ncon]):
                if cube not in contact.geom or contact.dist > 0:
                    continue
                gid = contact.geom2 if contact.geom1 == cube else contact.geom1
                name = model.geom(int(gid)).name
                finger = next((i for i, f in enumerate(FINGERS)
                               if name.startswith(f"orca_{f}_")), None)
                if finger is None:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                if force[0] <= 1e-5:
                    continue
                body = model.geom_bodyid[gid]
                rotation = data.xmat[body].reshape(3, 3)
                point = (contact.pos - data.xpos[body]) @ rotation
                normal = contact.frame[:3] * (1 if contact.geom1 == gid else -1) @ rotation
                rows.append(dict(episode=episode, time_sec=(frame + 1) * .05,
                                 finger=FINGERS[finger], geom=name, force_n=float(force[0]),
                                 local_z_m=float(point[2]), nx=float(normal[0]), ny=float(normal[1]),
                                 nz=float(normal[2]), pad=bool(is_distal_pad(name, point, normal, finger))))
    if rows:
        with (args.output / "reconstructed_contacts.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(run=str(args.run), step=args.step, episodes=len(episodes),
                   successes=sum(e["success30"] for e in episodes),
                   saved_trajectory_count=len(traces["qpos"]),
                   contact_scope="All saved trajectories, 100ms reconstructed contacts; not original substeps",
                   residual_composition=control_errors, contacts={})
    for finger in FINGERS:
        points = [r for r in rows if r["finger"] == finger]
        summary["contacts"][finger] = dict(
            count=len(points), pad_fraction=float(np.mean([r["pad"] for r in points])) if points else None,
            mean_normal=np.mean([[r["nx"], r["ny"], r["nz"]] for r in points], axis=0).tolist() if points else None)
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    if control_errors:
        assert max(c["ctrl_max_error"] for c in control_errors) < 2e-6, "Residual composition mismatch"
        assert max(c["requested_max_error"] for c in control_errors) < 2e-6
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    times = (np.arange(traces["cube_z"].shape[1]) + 1) * .05
    for i in range(min(4, len(traces["qpos"]))):
        label = f"seed {episodes[i]['seed']}"
        axes[0].plot(times, 1000 * (traces["cube_z"][i] - traces["spawn_z"][i]), label=label)
        axes[1].plot(times, traces["hold"][i])
        axes[2].plot(times, np.sqrt(np.mean(traces["requested_residual"][i] ** 2, axis=1)))
    axes[0].axhline(-10, color="red", linestyle="--", label="height failure threshold")
    for ax, label in zip(axes, ("Height change (mm)", "Continuous valid hold (s)", "Residual RMS (rad)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.tight_layout()
    fig.savefig(args.output / "height_hold_residual.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
