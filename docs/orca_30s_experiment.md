# OrcaHand 30 s grasp: experimental protocol

Status: F01 training and held-out comparison completed; see the Chinese
report `orca_ppo_guidance_comparison.md`. Started 2026-10-02.
The historical Aero result is not an Orca result. No historical PPO weights
will initialize either main experimental arm.

## Question and literature

Does demonstration initialization reduce the environment interaction needed
for OrcaHand to acquire and continuously hold an unsupported cube for 30 s?
A favorable answer is a hypothesis, not a required interpretation of the data.

- [DAPG, Rajeswaran et al.](https://arxiv.org/abs/1709.10087) motivates using
  behavior cloning (BC, supervised imitation of demonstrated actions) before
  reinforcement learning. BC followed by PPO is not the original DAPG algorithm.
- [PPO, Schulman et al.](https://arxiv.org/abs/1707.06347) supplies the shared
  on-policy update algorithm. Both arms collect their own simulation samples.
- [Residual RL, Johannink et al.](https://arxiv.org/abs/1812.03201) combines a
  base controller with learned corrections. This is a separate intervention
  from initialization and must not be silently mixed into the main comparison.
- [Agarwal et al.](https://arxiv.org/abs/2108.13264) motivates reporting
  variability and uncertainty rather than selecting one favorable seed.

## Common task and safety against confounding

Both arms use the same Orca model, 17 controls, action parameterization,
observations, reward, reset distribution, support removal schedule, network,
PPO hyperparameters, environment step budget, and evaluation episodes.
The object starts on a removable support with the hand not already grasping.
There is no object weld, gravity cancellation, or hidden supporting force.
Every evaluation lasts at least 35 s and never auto-resets after failure.

Success requires a continuous 30 s interval after support removal with finite
state, no drop, actual thumb and another finger contact, no palm/support
contact, and object height at least spawn height minus 1 cm. Save per-step
contact and height evidence, not just summed training-wrapper success metrics.
Native MuJoCo physical checks are development diagnostics; final policies must
pass the actual training simulator evaluation, plus a visual inspection.

The existing simplified collision model is a simulation approximation, not
hardware validation. Any geometry repair applies equally to both arms and is
recorded before freezing the final protocol.

## Arms

1. Scratch PPO: random policy initialization, no demonstration labels.
2. BC + PPO: identical architecture, supervised initialization using an explicit
   demonstration dataset, then the same PPO updates without imitation rewards.

The video source is HIC sequence 20 and its published tracking. Any geometric
optimization or simulation refinement must be labeled as such. A scripted
expert cannot be described as a policy extracted solely from RGB video.
Record dataset provenance, failed demonstrations, refinement simulation cost,
and BC wall time. A residual-controller arm can be added as a separately named
secondary experiment; it cannot replace the main scratch-vs-guided comparison.

## Development and confirmatory evaluation

First establish physical reachability and run short development probes. Tune
the shared task using development seeds only. Record all failed probes.
Freeze the final task, dataset, budget and hyperparameters before confirmatory
training. Plan five paired independent training seeds (0--4), with shared
evaluation seeds 1000--1015 for curves and reserved seeds 2000--2031 for the
final policies. Repeated evaluation episodes are not independent training runs.

Primary convergence measure: environment steps to >=80% 30 s success on two
consecutive evaluations. Runs not reaching it are censored at the common
budget, not assigned an invented convergence time. Also report success-vs-step
area, all individual curves, wall time both excluding and including guidance
preparation, maximum continuous hold, and failure modes. Report paired seed
differences and uncertainty; five seeds alone do not establish broad universality.

## Development log

### O00: audit

The previous Orca probes had zero continuous hold. The old 380-step episode at
0.05 s cannot measure 30 s. Baseline and residual modes used different action
definitions and reset poses. Baseline observations/reward also contained a
synthetic target pose. None of those runs is a fair confirmatory baseline.

Next controlled question: can bounded Orca fingertip IK produce a physically
stable, unsupported cube grasp in the existing proxy scene? Run a native
MuJoCo geometry/closure diagnostic before spending PPO training budget.

### O01: fixed-wrist fingertip IK and physical replay

Command: `python scripts/diagnose_orca_grasp.py` in `aero_rl`.
No checkpoint; bounded SciPy least-squares followed by actual position-actuator
simulation, gravity enabled, support lowered after 2 s. 27 cube placements.
Artifact: `artifacts/orca30/geometry_o01/report.json` and `best_trace.json`.
Hold first/last/max: 0/0/0 s; best_step: not applicable (not training).
Some configurations remain aloft by palm contact; others fall on release.
Thus the positive-looking non-drop cases do NOT establish unsupported grasp.
The next controlled change is geometry/reset: search wrist preorientation and
cube location with the same actuator model, IK method, and success criterion.

### O02: wrist/location expansion

Command: `python scripts/diagnose_orca_grasp.py --expanded --output artifacts/orca30/geometry_o02`.
144 candidates; hold first/last/max remains 0/0/0 s. No training checkpoint.
Review of fingertip trajectories revealed an IK modeling error: the requested
opposing faces were reversed relative to natural closure. This makes fingers
sweep through the object to reach nominally accurate endpoints. O03 changes
only the opposition direction (thumb toward +Y, fingers toward -Y); it is a
geometric diagnostic correction, not evidence that PPO or demonstrations work.
To reproduce O01/O02 with the corrected script, add `--opposition -1`.

### O03: corrected opposing faces

Command: `python scripts/diagnose_orca_grasp.py --output artifacts/orca30/geometry_o03`.
27 placements; hold first/last/max = 0/6.80/12.37 s; best candidate index 8
(zero-based), cube [0.055, 0.050, 0.270], wrist 0. No PPO checkpoint.
This is the first positive native physics signal, but the best case later drops.
The IK target is imperfect due to joint bounds (19 mm aggregate fingertip
error); a low IK error alone was not a useful physical success predictor.
O04 repeats the wrist/location expansion with corrected face orientation.

### O04: physical reachability candidate

Command: `python scripts/diagnose_orca_grasp.py --expanded --output artifacts/orca30/geometry_o04`.
144 candidates; best continuous hold 33 s of a 36 s rollout, with no palm or
support contact counted. Selected cube [0.055,0.025,0.265], wrist -0.25 rad.
This is native scripted-control feasibility, NOT a trained PPO success.
The selection is development-only; confirmatory task randomization and policy
training are still pending. Next gate: run the same controller in MJX with
small common reset perturbations, save all episodes and render the grasp.

The new `scripts/orca_benchmark_env.py` defines a separate common task rather
than changing historical environments. Its absolute normalized actions cover
the same joint limits for both arms, with 0.1 rad/step slew limiting. The policy
observes proprioception, relative cube state and task timing, but no reference
pose. Contact tests require nonpositive distance and positive normal force.
The task has no imitation reward or demonstration-specific reset.

### O05: MJX replay

Command: `python scripts/check_orca_benchmark.py` with the playground PYTHONPATH
and the aero_rl interpreter. Eight independent reset seeds 0--7, 40 s each.
All eight hold 37.05 s continuously under the strict common criterion.
Artifact: `artifacts/orca30/mjx_o05/` (report, all traces, rollout.mp4 and PNG).
Visual inspection of the 30 s frame confirms the cube is between the fingers,
well above the withdrawn support. This validates the benchmark controller, not
a learned policy or video benefit. Dataset provenance is geometric controller.

### O06: BC/PPO entry smoke

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --output artifacts/orca30/dev_o06_bc_seed10`.
The first attempt fit BC but failed before any PPO update because the network
factory accepted keywords only while Brax passes positional dimensions.
No PPO metrics or convergence claim. The corrected retry has suffix `_retry`.
This development probe deliberately uses the geometric demonstrations from O05;
it is not a video-derived demonstration experiment. HIC guidance and provenance
must be integrated and compared separately before any video-related claim.

O06 corrected-entry result: success30 first/last/max = 1/0/1, best_step=0;
mean continuous hold 36.78125 -> 2.19375 -> 0 s at steps 0/40960/81920.
The BC network can reproduce the geometric controller, but the RL training
wrapper default does not reset task-specific info. After a drop, the support
timer remains advanced although the physical scene is reset. This is an
invalid training reset, not a fair convergence measurement. Keep the failed
run, fix `full_reset=True`, and rerun exactly the same BC/PPO settings as O07.
Add a GPU smoke test that forces an early drop and checks clock/support reset.

### O07: correct full reset, same BC/PPO hyperparameters

GPU reset smoke passed. Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --output artifacts/orca30/dev_o07_bc_fullreset_seed10`.
Success30 first/last/max=1/0/1, best_step=0; mean hold 36.76875/0/0.43125 s
at 0/40960/81920. Thus the reset bug was real but did not explain all policy
degradation. BC competence alone is not enough for stable fine-tuning.
Next controlled change is PPO learning rate: 3e-4 to 3e-5, everything else fixed.
Do not retroactively describe O07 as successful PPO training.

### O08: smaller PPO learning rate

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --learning-rate 3e-5 --output artifacts/orca30/dev_o08_bc_lr3e5_seed10`.
Only learning rate changes relative to O07. Success30 first/last/max=1/1/1,
best_step=0 (tie); mean hold first/last/max=36.73125/37.05/37.05 s,
best mean-hold step=81920. All eight development evaluations at the final
checkpoint exceed 30 s. This is a positive BC+PPO development result, not a
completed multi-seed comparison. Next: same-budget scratch PPO at the same LR.

### O09: HIC timing, native diagnostic

Command: `python scripts/check_orca_hic_native.py`.
Replace the linear closure schedule with monotone normalized thumb-index
distance from HIC frames 0--60, retimed from 2 s to 1.5 s; retain the same
geometry-optimized endpoint. Native continuous hold is 37 s, peak contact
force 5.20 N. Artifact `artifacts/orca30/hic_native_o09/`.
This is a hybrid: the video contributes closure timing, NOT the optimized
Orca endpoint pose. It is not evidence that raw video or DexMV alone can grasp.
To attribute benefit specifically to video, compare against the geometric-only
BC arm, in addition to the scratch arm. This attribution remains open.

### O10: matched scratch PPO short probe

Command: `python scripts/train_orca_comparison.py --arm scratch --steps 81920 --evals 3 --learning-rate 3e-5 --output artifacts/orca30/dev_o10_scratch_lr3e5_seed10`.
Same reset, reward, actor/critic initial seed, network, optimizer and sampling
budget as O08, with no BC. Success30 and continuous hold first/last/max=0/0/0;
best_step=0 (tie). Return improves 59.26 -> 69.73 -> 122.36 without a valid
hold. The initialization advantage is real on this development seed, but this
short budget cannot determine scratch convergence or cross-seed reliability.
Next: extend the scratch learning budget to 1,310,720 steps from a fresh seed-10
initialization, keeping all training parameters unchanged (O11).

### O11: scratch PPO long development run

Command: `python scripts/train_orca_comparison.py --arm scratch --steps 1310720 --evals 9 --learning-rate 3e-5 --output artifacts/orca30/dev_o11_scratch_long_seed10`.
Success30 first/last/max=0/0/0.875, best_step=983040; mean hold
first/last/max=0/1.0/35.79375 s, best_step=983040. Seven of eight episodes at
that checkpoint exceed 30 s (the eighth holds 29.3 s), but the next evaluation
regresses. The two-consecutive-evaluation convergence criterion is NOT met.
The task is learnable from scratch, yet training is unstable.

Inspection of installed Brax reveals `init_noise_std=.2` is ignored by the
`tanh_normal` MLP branch. Actual scratch exploration standard deviation starts
near 0.7, and fitting the shared BC trunk can also change the untrained scale
outputs. O12 explicitly initializes scale-head weights to zero and biases to
inverse-softplus(0.2-0.001), for both arms. This preserves the network architecture
and actuator action space while making the initial exploration parameter real.
Add a unit test for the effective standard deviation before further comparison.

### O12: explicit exploration standard deviation

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --learning-rate 3e-5 --init-std .2 --output artifacts/orca30/dev_o12_bc_std02_seed10`.
Six focused tests pass, including actual policy scale=0.2 and continuous rather
than accumulated hold counting. Success30 first/last/max=1/0/1, best_step=0;
mean hold 36.74375 -> 31.84375 -> 23.4625 s. Explicitly fixing initial noise
does not prevent gradual BC forgetting. This rules out claiming robust BC-only
initialization from O08 alone. Next run the matched long scratch arm (O13),
and investigate demonstration-loss regularization as a separately named method,
motivated by DAPG's distinction between pretraining and continued guidance.
Any formal comparison must disclose that added loss; it is not merely BC init.

## Development amendment before confirmatory runs

BC-only fine-tuning has now failed retention in several controlled probes.
Add a third explicit method: BC initialization + PPO + demonstration MSE loss
(actor means against demonstrated normalized actions in pre-tanh coordinates).
The original Brax PPO loss remains intact; the extra scalar loss is added by
a process-local wrapper restored after training. The environment reward,
observations, resets, model and action space remain identical across methods.
This is demonstration-regularized PPO inspired by DAPG, not exact DAPG and not
an action-residual architecture. Preserve BC-only as an ablation, and evaluate
geometric-only vs video-timed demonstrations to avoid attributing IK benefits
to video. No confirmatory runs have begun under the earlier provisional plan.

Guidance preparation costs so far: O01--O04 used 299809 native 0.01 s physics
steps (59961.8 control-step equivalents), about 25 s measured diagnostic wall
time excluding interpreter startup. O05 collected 6400 control-step samples.
Report these costs rather than calling BC's zero PPO steps zero total cost.

### O13: matched scratch run with explicit std=0.2

Command: `python scripts/train_orca_comparison.py --arm scratch --steps 1310720 --evals 9 --learning-rate 3e-5 --init-std .2 --output artifacts/orca30/dev_o13_scratch_std02_seed10`.
Success30 first/last/max=0/0/0; mean hold first/last/max=0/0/4.425 s,
best_step=655360. Compared with the legacy ~0.7-noise scratch run O11, this is
worse exploration, not a stronger baseline. Do not use this weaker setting to
claim the largest possible demonstration benefit. Test explicit std=0.7 before
freezing the common formal configuration.

### O14: DexMV-timed demonstration replay

Command: `python scripts/check_orca_benchmark.py --timing dexmv_pinch --output artifacts/orca30/mjx_o14_dexmv_timing`.
Eight of eight MJX episodes exceed 30 s; seven hold 37.05 s and one 35.45 s.
The input is HIC -> DexMV PositionOptimizer fitted thumb/index positions,
frames 0--95. Normalize monotone pinch closure and retime it to 1.5 s;
the final Orca pose remains the geometry-calibrated O04 endpoint. No raw-video
recognition was rerun and no claim is made that video specifies the full pose.
O14 briefly overlapped O13 on the GPU, so their wall times are development
diagnostics, not a clean wall-clock speed benchmark.

Next controlled intervention O15: add weight-10 demonstration MSE to O12's
BC+PPO objective, leaving the geometric dataset and all other settings fixed.

### O15: continued demonstration regularization, weight 10

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --learning-rate 3e-5 --init-std .2 --demo-weight 10 --output artifacts/orca30/dev_o15_bc_demo10_seed10`.
Success30 first/last/max=1/0/1, best_step=0 (tie with 40960);
mean hold first/last/max=36.75/18.20625/37.05 s, best_step=40960.
The regularizer delays but does not eliminate forgetting. Next objective-weight
probe O17 increases only the MSE coefficient to 1000; report this coefficient
explicitly rather than presenting it as standard DAPG.

### O16: full 3D video trajectory retargeting diagnostic

Command: `python scripts/bridge_dexmv_to_orca_trajectory.py`.
Transfer DexMV's three object-relative fingertip trajectories using a rigid
alignment and cube-size ratio, then bounded per-frame Orca IK. This carries
3D motion, not merely the scalar pinch timing in O14. The target endpoint is
still geometry-calibrated and this must remain part of its provenance.
Mean aggregate tip error 9.96 mm; native replay holds only 4.87 s and drops at
9.03 s. Therefore the full-trajectory version is NOT yet a successful video
demonstration dataset. Preserve its failure trace and do not replace O14 with
it unless physics validation passes after a documented correction.

### O17: stronger demonstration loss

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --learning-rate 3e-5 --init-std .2 --demo-weight 1000 --output artifacts/orca30/dev_o17_bc_demo1000_seed10`.
Success30 first/last/max=1/1/1, best_step=0 (tie); mean hold
36.7625/36.7625/36.8625 s, best_step=40960. Retention passes this short test.
This large coefficient strongly anchors the actor to the demonstrations; do
not claim that PPO has independently discovered or improved the grasp.
O19 changes only initial exploration std to 0.7, since O13 showed 0.2 was a
weak scratch baseline. Long retention and multi-seed evidence remain required.

### O18: video hold-endpoint correction

Command: `python scripts/refine_orca_video_endpoint.py`.
45 fingertip endpoint offsets after the unchanged full-video acquisition;
maximum native hold only 5.74 s. The cube has already moved left by about 2 cm
during acquisition, so endpoint-only changes do not resolve the failure.
Next geometry hypothesis: constrain penetration during per-frame retargeting,
rather than relying on inaccurate free-space fingertip matching alone.

### O19: stronger common exploration, guided short run

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --learning-rate 3e-5 --init-std .7 --demo-weight 1000 --output artifacts/orca30/dev_o19_bc_std07_seed10`.
Success30 first/last/max=1/1/1, best_step=0 (tie); mean hold
36.80625/36.675/36.80625 s. Strong guidance remains viable with std=0.7.

### O20--O21: 3D video acquisition diagnostics

O20 adds penetration penalties during IK (`--collision-weight 10`): hold
falls to 2.15 s, so this version is not adopted.
O21 restores O16's IK and changes only the initial acquisition blend from
0.4 to 1.5 s (`--acquisition-sec 1.5`). Native continuous hold reaches 33.6 s,
but the object eventually drops at 37.62 s. This transfers full 3D fingertip
motion and is a genuine positive video-trajectory candidate, not yet an MJX
or PPO success. Next run strict MJX replay with the common perturbations.

### O22: full video trajectory in MJX

Command: `python scripts/check_orca_benchmark.py --reference artifacts/orca30/dexmv_slow_acquisition_o21/reference.npz --output artifacts/orca30/mjx_o22_video3d`.
Seven of eight reach 30 s; continuous durations are
36.60/32.65/31.55/37.05/28.30/35.80/37.05/37.05 s. Three eventually drop
within 40 s. Do not turn 7/8 threshold successes into a claim of no drops.
`scripts/select_orca_demos.py` retains the five complete non-dropping episodes
(indices 0,3,5,6,7), 4000 samples, in `artifacts/orca30/video3d_demos_v1/`.
All eight original episodes and their preparation cost (6400 steps) remain in
the source artifact. These are simulated variants of ONE human video, not five
independent human demonstrations.

Next O23 changes O19's demonstration source from geometric-only to the full
DexMV 3D trajectory dataset, keeping PPO/BC settings fixed.

### O23: full-video demonstration-guided PPO

Command: `python scripts/train_orca_comparison.py --arm bc --steps 81920 --evals 3 --learning-rate 3e-5 --init-std .7 --demo-weight 1000 --dataset artifacts/orca30/video3d_demos_v1/trajectories.npz --output artifacts/orca30/dev_o23_video3d_seed10`.
Success30 first/last/max=0/1/1, best_step=40960. Mean continuous hold
0/35.95/36.34375 s, best_step=40960. Every evaluation episode at both updated
checkpoints exceeds 30 s. This is a positive learned full-video-trajectory
guidance result, unlike merely replaying a scripted controller. It is still
one development training seed and does not establish the final convergence
comparison. More BC updates alone are another possible explanation for part
of the improvement; a compute-matched BC-only ablation would isolate PPO's
additional contribution, whereas the main comparison tests demonstration use.

## Directional hypothesis requested by the user

Prespecify H1: demonstration guidance accelerates acquisition of Orca's 30 s
unsupported cube grasp relative to scratch PPO. The null is no improvement.
This is the expected conclusion to TEST, not permission to discard negative
results or announce it before evidence. Freeze the confirmatory protocol and
keep all training seeds, checkpoints, failed episodes and censored runs.
Use identical 17-dimensional absolute normalized actions, actuator limits,
slew limits, environment rewards and reset distributions in the main arms.
Inspect multi-view reset/acquisition/hold images when geometric alignment is
in doubt; check finger-object contact, palm clearance and support removal.
The final report must distinguish evidence for demonstrations as a whole from
evidence for video's incremental value over geometric calibration.

## Visual alignment check

O23 checkpoint 81920, evaluation episode 0 (seed 100), rendered from saved
physical states using the same collision model. At 0.05 s the cube is on the
support and not already enclosed by the hand. At 30.05 s opposing camera views
show the cube between the thumb and curled fingers, with clearance from the
palm; the support has moved downward. The blue/purple shapes are the explicit
proxy collision model, not a claim of full-mesh or hardware validation.
Images corroborate the contact metrics but do not replace force/contact tests.

![Initial alignment](../artifacts/orca30/dev_o23_video3d_seed10/multiview/rollout_step000_az45.png)
![Thirty-second grasp, first view](../artifacts/orca30/dev_o23_video3d_seed10/multiview/rollout_step600_az45.png)
![Thirty-second grasp, opposite view](../artifacts/orca30/dev_o23_video3d_seed10/multiview/rollout_step600_az225.png)

### O24: explicit std=0.7 scratch long run

Command: `python scripts/train_orca_comparison.py --arm scratch --steps 1310720 --evals 9 --learning-rate 3e-5 --init-std .7 --output artifacts/orca30/dev_o24_scratch_std07_seed10`.
The first submission never launched because automatic permission review timed
out; the permitted retry completed. Success30 first/last/max=0/0/1,
best_step=1146880. All eight episodes at that checkpoint hold 37.05 s, but the
last checkpoint regresses to mean hold 0.725 s. Scratch PPO can solve the same
task and is not an intentionally incapable baseline; stable convergence under
the two-consecutive-evaluation rule remains unproven. Next O25 extends the
successful full-video guided configuration to the same 1,310,720-step budget.

## Existing refined-model inventory

Search covered this integration workspace and `/home/ll/SRTP/ORCAHAND`.

- Aero V2: `Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand_v2_vertical_coacd.xml`
  and `right_hand_v2_vertical_coacd_qbr.xml`, with scene entry points and
  `right_hand_v2_vertical_coacd_report.txt`. Despite the filename, the report
  describes a current 25-geom approximation (10 boxes, 15 capsules), including
  an eight-box palm and multiple distal capsules. Separate CoACD STL assets
  exist under `Aero-Hand/handinformation/URDF/右/qbr/meshes/coacd_v2/`.
- Orca V2 original: `Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/orca_hand/xmls/orcahand_description/v2/scene_right.xml`.
  This includes detailed CAD-derived mesh geometry, with noncolliding skin
  meshes and colliding structural meshes. A detailed visual is not by itself
  a refined MJX collision model.
- Orca V1: the sibling `v1/assets/mjcf/right/collision/` contains separate
  reduced-face collision STL files; it is a legacy model, not a drop-in V2
  refinement. The source README documents separate visual/collision mesh sizes.
- Current experimental model: `v2/scene_right_mjx_grasp_cube.xml` and
  `v2/models/mjcf/orcahand_right_mjx_grasp_body.xml`, retaining V2 joints but
  using explicit primitives. No separately hand-refined V2 collision version
  was identified in the inspected folders; do not silently mix V1/V2 physics.

The complete V2 `orcahand_right_body.xml` is byte-identical in this workspace,
`/home/ll/SRTP/ORCAHAND/orcahand_description/`, and
`/home/ll/SRTP/ORCAHAND/orca_sim/src/orca_sim/models/`:
SHA256 `aa80571e58bf305fe6ff3bafdb0555fe9370f1ce93a63afb287b11adee61c1a7`.
No external model directory was modified. The existing Orca full-mesh previews
are in `Aero-Hand/v2_iteration_docs/scene_previews/orca_right_*_preview.png`.

### O25: full-video guided long run

Command: `python scripts/train_orca_comparison.py --arm bc --steps 1310720 --evals 9 --learning-rate 3e-5 --init-std .7 --demo-weight 1000 --dataset artifacts/orca30/video3d_demos_v1/trajectories.npz --output artifacts/orca30/dev_o25_video3d_long_seed10`.
Completed all 1,310,720 steps. Success30 first/last/max=0/0.875/1,
best_step=163840; mean hold first/last/max=0/34.875/37.05 s,
best_step=163840. Seven consecutive nonzero checkpoints have 8/8 successes;
the final checkpoint has one 28.55 s failure. The two-evaluation convergence
criterion is met at onset 163840, confirmation 327680. This is development
seed 10 only, not the confirmatory comparison. Strict contact criteria remain
unchanged. Next intervention changes training seeds under frozen settings,
not reward or geometry.

## Frozen confirmatory protocol F01

Frozen before starting training seeds 0--4. The main comparison is scratch PPO
versus BC initialization plus PPO plus continued demonstration MSE (weight
1000). This supersedes the initial BC-initialization-only arm description;
O07/O12/O15 document why continued guidance was introduced. It is not exact
DAPG, nor an action-residual controller. No historical policy initializes
either arm. Geometry-calibrated full-video demonstrations are the intervention
as a whole; this experiment alone cannot identify video's incremental benefit
over geometry or isolate PPO from additional BC updates.

- Five paired seeds 0,1,2,3,4; alternate which arm runs first within each pair.
- Each run: 1,310,720 PPO steps, 17 evaluations including step zero, every
  81,920 steps; 16 curve episodes, seeds 1000--1015.
- Common learning rate 3e-5, explicit initial std 0.7, 128 environments,
  128x128 networks, absolute normalized 17D actions, unchanged common task.
- Guided arm: 2000 BC updates, dataset `video3d_demos_v1/trajectories.npz`,
  continued demonstration coefficient 1000. Scratch receives neither.
- Primary criterion remains >=80% strict 30 s success on two consecutive
  evaluations; report both onset and confirmation. Report censored runs as
  not converged within budget, not as having converged at the budget.
- Select one checkpoint per run by highest curve success, then highest mean
  hold, then earliest step. Evaluate every selected checkpoint on untouched
  seeds 2000--2031, and also evaluate final checkpoints to expose regression.
  Do not select a training seed using held-out performance.
- Run jobs serially. Record wall clock and BC cost; shared compilation cache
  makes wall time secondary. Report demo preparation and geometry-search costs
  separately, never call them free samples. Save all curves and failures.
- Record source/dataset hashes before launch. No source, reward, geometry,
  dataset, or hyperparameter changes during F01. Any later change is a new
  experiment and does not overwrite this one.

The experimental unit is the training seed, not each evaluation episode.
Use paired AUC differences with a seed bootstrap and disclose five-seed
uncertainty. Negative or mixed findings remain valid outcomes of H1's test.

### F01 completed-run log (updated as runs finish)

All commands and per-checkpoint metrics are retained under
`artifacts/orca30/confirmatory_f01/`; `completed.jsonl` records each completion
before launching the next serial run. Seven environment/algorithm invariants
and three convergence-statistics tests passed before interpreting results.

| Arm | Seed | First / last / max success30 | First best step | Mean hold first / last / max (s) | Convergence onset / confirmation | Next controlled run |
| --- | ---: | --- | ---: | --- | --- | --- |
| Scratch | 0 | 0 / .75 / .875 | 573440 | 0 / 29.600 / 32.759 | Not reached | Guided seed 0, same task and PPO settings |
| Guided | 0 | 0 / .5 / .9375 | 245760 | .984 / 27.222 / 35.444 | 491520 / 573440 | Guided seed 1, change random seed only |
| Guided | 1 | 0 / 0 / .0625 | 163840 | 1.138 / 0 / 4.491 | Not reached | Scratch seed 1, same task and PPO settings |
| Scratch | 1 | 0 / 0 / .5625 | 655360 | 0 / 5.747 / 26.112 | Not reached | Scratch seed 2, change random seed only |
| Scratch | 2 | 0 / 0 / 1 | 901120 | .091 / 6.144 / 37.050 | Not reached | Guided seed 2, same task and PPO settings |
| Guided | 2 | 0 / 1 / 1 | 81920 | 0 / 36.694 / 36.728 | 81920 / 163840 | Guided seed 3, change random seed only |
| Guided | 3 | 1 / 1 / 1 | 0 | 36.216 / 37.050 / 37.050 | 163840 / 245760 | Scratch seed 3, same task and PPO settings |
| Scratch | 3 | 0 / 0 / 1 | 1064960 | 0 / 0 / 36.122 | Not reached | Scratch seed 4, change random seed only |
| Scratch | 4 | 1 / 1 / 1 | 0 | 37.050 / 37.050 / 37.050 | 983040 / 1064960 | Guided seed 4, same task and PPO settings |
| Guided | 4 | 1 / .9375 / 1 | 0 | 36.450 / 36.162 / 36.678 | 0 / 81920 | Frozen independent held-out evaluation |

Scratch seed 0 sometimes reaches the threshold, but not on two consecutive
evaluations. This is retention failure, not proof of physical impossibility.
No task or reward change is justified during the frozen comparison. The next
run changes only the prespecified demonstration intervention. All successes
use strict force-bearing finger contact and exclude non-finger support.

Pair 0 normalized AUC is .30078125 scratch versus .546875 guided. Guided
passes the prespecified acquisition criterion sooner, but terminal success is
worse (.5 versus .75). Preserve that negative retention result. The selected
guided checkpoint is 245760 by the frozen rule, not the later checkpoint with
highest mean hold. Next run changes seed only; no hyperparameter adaptation
is allowed within F01. GPU health check during training: 60 C, 1547/8188 MiB.

Guided seed 1 fails despite small demonstration fitting loss. Its AUC is
.0078125, a negative result that prevents generalizing from development seed
10 or pair 0 alone. A covariate-shift/limited-coverage explanation is plausible
but unproven; do not change the dataset mid-experiment. Next evaluate the
paired scratch initialization under the frozen budget.

Pair 1 normalized AUC is .0859375 scratch versus .0078125 guided: this pair
contradicts a universal guidance advantage. Neither arm converges under the
prespecified rule. No post-hoc exclusion is made. Next run changes training
seed to 2 under exactly the same frozen task and optimization settings.

Scratch seed 2 reaches 16/16 at step 901120, but does not meet the two-adjacent
evaluation criterion and regresses. AUC=.125. A progress update based only on
the tail initially missed that earlier success; the full-curve summary above
is authoritative and the update was corrected. Retention, not reachability,
is the main failure mode. The paired guided run is now using the same seed.

Pair 2 normalized AUC is .125 scratch versus .83203125 guided. Guided has two
complete failures at steps 573440 and 655360, so its early convergence does
not imply uninterrupted retention. The terminal policy recovers to 16/16.
Selected guided step=819200 under the frozen success/mean-hold rule. Next run
changes only training seed to 3; all intermediate failures remain available.

Guided seed 3 AUC=.75. BC alone initially achieves 16/16, but the first PPO
checkpoint regresses; therefore convergence onset is 163840, not zero under
the two-evaluation rule. Selected step=737280 (earliest tied perfect mean
hold), terminal also has 16/16. Next run changes only the demonstration arm,
retaining seed 3 and all shared settings.

Pair 3 AUC=.07421875 scratch versus .75 guided. Scratch reaches a single
16/16 checkpoint at 1064960, then regresses through 3/16 to zero. The selected
checkpoint still demonstrates physical task reachability and will be evaluated
alongside the terminal checkpoint. Next run changes training seed to 4 only.

Scratch seed 4's untrained random initialization already succeeds on 16/16;
training first destroys that behavior, then recovers and passes the two-eval
criterion at 983040/1064960. AUC=.38671875. The frozen selection rule chooses
step zero, which must be labeled UNTRAINED, not presented as a learned PPO
policy. Its terminal trained policy also achieves 16/16. This reinforces the
limited, fixed-placement nature of the task. Next run changes only guidance.

Before opening held-out results, choose training seed 2 for illustrative
paired videos: both selected checkpoints are post-training and both reached
16/16 on the curve set. Render held-out episode seed 2000 for each; if it
fails, retain that video and optionally add a clearly labeled successful
example. Do not use the held-out set to choose or tune a training checkpoint.

Guided seed 4 AUC=.708984375 versus scratch .38671875. Its initial BC policy
and first PPO checkpoint meet the adjacent-evaluation rule, but later failures
remain in the curve. Selected step=737280; terminal 15/16. All ten F01 runs
completed the fixed budget. Next action is the preregistered 32-episode
independent evaluation of all selected and terminal policies, with no further
training or parameter changes before reporting F01.

### F01 aggregate and held-out audit

All ten source/model configurations match; dataset hashes match the frozen
manifest; every run has the exact 17-point schedule. Mean AUC is .19453125
scratch and .569140625 guided, paired difference .374609375, 95% paired seed
bootstrap interval [.131640625, .617578125]. Four of five guided runs meet
the operational convergence criterion, versus one scratch run. This supports
the finite-task sample-efficiency hypothesis, not universal superiority.

All twenty selected/terminal policies were reloaded on seeds 2000--2031.
Independent continuous counters match environment counters in all 640
episodes. Selected-policy mean success is .85625 scratch versus .74375
guided, a negative final-quality result. Terminal mean success is .33125
scratch versus .675 guided. Neither negative seeds nor regression checkpoints
were excluded. Seed 2 selected policies both achieve 32/32; scratch minimum
37.05 s, guided minimum 31.55 s. Exact per-seed counts are in the Chinese
report and `confirmatory_f01/heldout_summary.json`.

### Delivery audit

`artifacts/orca30/final_f01/` contains all ten selected and terminal policy
archives, not just positive seeds. Its 132 registered files pass SHA256 checks.
Both seed-2 archived policies were reloaded for another 32 episodes: scratch
32/32 at 37.05 s; guided 32/32, minimum 31.40 s, mean 36.6625 s. The small
difference from the original guided replay (minimum 31.55 s) is retained;
do not claim bitwise trajectory determinism.

Both paired videos are 40 s, 800 frames, 20 FPS, 960x720. Their 30-second
frames show the cube held clear of palm and support. Original CAD visual-only
overlays preserve masses, inertias, joints and actuator parameters; inspection
shows approximate, not exact, agreement of collision primitives with CAD.
The refined Aero models were located; no separate refined Orca V2 MJX model
was found. These are proxy-simulation results, not full-mesh/hardware claims.

Seven environment/algorithm tests and three statistics tests pass. Training,
held-out evaluation, archive reloads and render processes have completed.
No Git commit or push was made. The finite-task sample-efficiency hypothesis
is supported by the paired AUC evidence; the negative selected-policy-quality
comparison and failed guidance seed remain explicit in the final report.
