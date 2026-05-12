# DEXMV-PPO

This repository is the integration workspace for combining the Aero-Hand PPO project with DexMV demonstrations.

## Repository layout

- `Aero-Hand/`: PPO subproject, tracked as a Git submodule.
- `DexMV/`: reserved location for the DexMV project, to be added after download.
- `bridges/`: planned conversion scripts between DexMV outputs and PPO demonstration datasets.
- `datasets/`: generated datasets such as `demo_dataset.npz`; ignored by Git.
- `docs/`: integration notes and interface records.

## Planned integration flow

1. Inspect PPO interfaces in `Aero-Hand`.
2. Inspect DexMV interfaces after the DexMV project is downloaded.
3. Generate bridge scripts.
4. Generate `demo_dataset.npz`.
5. Load `demo_dataset.npz` from the PPO project.
6. Add BC pretraining, demo loss, pose rewards, and residual PPO.

## Submodule usage

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```
