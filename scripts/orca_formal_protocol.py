"""Frozen recipes for the HIC53 confirmatory study; no outcome-based selection."""

SEEDS = (301, 302, 303, 304, 305)
STEPS = 655360
MODEL = "artifacts/orca_collision_refit/O20_hic53_elliptic/cube.xml"
REFERENCE = "artifacts/orca_collision_refit/O20_hic53_elliptic/reference.npz"
PHYSICS_GATE = "artifacts/orca_collision_refit/O21_hic53_elliptic_validation/report.json"
DATASET = "artifacts/orca_collision_refit/M01_hic53_native_pilot/teacher/trajectories.npz"
GATES = {("video", .08): "artifacts/orca_formal/G01_video_bound08/report.json",
         ("video", .15): "artifacts/orca_formal/G02_video_bound15/report.json",
         ("endpoint", .15): "artifacts/orca_formal/G03_endpoint_bound15/report.json",
         ("initial", .15): "artifacts/orca_formal/G04_initial_bound15/report.json"}
SOURCE_NAMES = (
    "train_orca_matched.py", "orca_matched_env.py", "orca_hic53_env.py", "orca_refit_env.py",
    "orca_residual_env.py", "orca_benchmark_env.py", "orca_residual_controls.py", "orca_residual_metrics.py",
    "train_orca_comparison.py", "orca_guidance_recheck.py", "train_orca_guidance_recheck.py",
    "prepare_hic_orca_object.py", "orca_contact_forces.py", "orca_solver_runtime.py", "orca_native_physics.py",
    "orca_formal_protocol.py", "audit_orca_formal_run.py", "run_orca_formal.py",
    "run_orca_matched_pilot.py", "diagnose_orca_refit.py", "audit_orca_trace_geometry.py",
)


def recipes():
    rows = []
    def add(phase, seed, method, tactile, mode="video", std=.7, label=None, bc=None):
        label = label or f"{method}_{tactile}"
        rows.append(dict(phase=phase, seed=seed, method=method, tactile=tactile,
                         reference_mode=mode, std=std, bound=.15 if method == "residual" else .08,
                         steps=0 if method == "bc" else STEPS, label=f"s{seed}_{label}", bc=bc))
    for seed in SEEDS:
        for method in ("bc", "ppo", "guided", "residual"):
            for tactile in ("off", "on"):
                add("main", seed, method, tactile,
                    bc=f"s{seed}_bc_{tactile}" if method == "guided" else None)
    for seed in SEEDS:
        for mode in ("endpoint", "initial"):
            for tactile in ("off", "on"):
                add("reference", seed, "residual", tactile, mode, label=f"residual_{mode}_{tactile}")
    for seed in SEEDS:
        for tactile in ("off", "on"):
            add("exploration", seed, "bc", tactile, std=.03, label=f"bc_smallstd_{tactile}")
            add("exploration", seed, "guided", tactile, std=.03, label=f"guided_smallstd_{tactile}",
                bc=f"s{seed}_bc_smallstd_{tactile}")
    return rows


def scenarios():
    return {
        "nominal": {}, "xy5mm": {"xy_jitter_m": .005}, "xy10mm": {"xy_jitter_m": .01},
        "yaw10deg": {"yaw_deg": 10.}, "yaw20deg": {"yaw_deg": 20.},
        "mass05": {"mass_scale": .5}, "mass15": {"mass_scale": 1.5},
        "friction07": {"friction_scale": .7}, "friction13": {"friction_scale": 1.3},
        "noise02": {"noise_std_n": .2}, "push01": {"push_force_n": .1},
        "push02": {"push_force_n": .2},
    }
