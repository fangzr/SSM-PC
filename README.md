# Shared Spatial Memory Through Predictive Coding

This repository bundles the open-source implementation that accompanies the paper
**"Shared Spatial Memory Through Predictive Coding"**.  The project integrates
modules for grid-cell representation learning, bird's-eye-view (BEV) prediction,
and (upcoming) shared spatial memory components. 

## Paper Overview

Sharing and reconstructing a coherent map is notoriously challenging in partially observed, bandwidth-limited environments. The associated paper reframes cooperation as minimizing mutual predictive uncertainty: agents communicate only the bits that most reduce their partners' prediction error. The core ingredients are:

- A grid-like metric scaffold that emerges from self-supervised motion forecasting and supports accurate self-localization.
- Social place-cell (SPC) populations that encode teammates’ trajectories, enabling agents to reason about where peers are likely to be.
- A hierarchical RL policy that decides *when* communication is worth the bandwidth, guided by an information-bottleneck objective.

On the Memory-Maze benchmark this predictive-coding strategy remains robust even when the link budget is slashed from 128 to 4 bits per step (success rate drops gently from 73.5% to 64.4%), while a full-broadcast baseline collapses from 67.6% to 28.6%.

![Overview of the predictive coding framework](image/method.png)

*Figure 1. Predictive coding framework for shared spatial memory. Panel (a) depicts the cooperative navigation problem; panel (b) highlights the self-localization scaffold built from emergent grid codes; panel (c) summarizes the communication-aware decision loop that prioritizes uncertainty reduction.*

<p align="center">
  <img src="image/video 5.2a.png" alt="Early training: unstable path integration" width="22%">
  <img src="image/video 5.2b.png" alt="Mid training: improving position estimates" width="22%">
  <img src="image/video 5.2c.png" alt="Late training: refined grid structure" width="22%">
  <img src="image/video 5.2d.png" alt="Converged model: stable trajectory prediction" width="22%">
</p>

*Figure 2. Emergence of grid-cell-like predictions across training. (a) Early checkpoints show diffuse, drifting trajectories. (b) Mid-training models begin to align the latent lattice with actual coordinates. (c) Near convergence the bottleneck units form a hexagonal code. (d) Mature models maintain tightly overlapped predicted and ground-truth paths, mirroring the accuracy gains reported in the paper.*


---

## Repository Layout

```
SSM-PC/
├── grid_cell/           # Grid-cell training, inference, and visualization
├── bev_generation/      # BEV predictor training and visualization utilities
├── TODO.md              # Pending modules slated for open-sourcing
├── LICENSE
└── README.md
```

The sections below summarize each released module and how to reproduce key
results.  Additional modules requested by the author (SPC, maze data capture,
SLAM reconstruction, MAPPO, and IB-based compression) remain TODOs and are
tracked explicitly so future releases can extend the codebase without
ambiguity.

---

## Dataset Preparation

Both sub-projects expect datasets to be placed inside repository-local folders
or supplied via environment variables that override the defaults.

### Grid-Cell Module

- Default location: `grid_cell/data/self-navigation-maze-frame-only/`
- Override via env vars: set `GRID_DATA_ROOT` and optionally
  `GRID_DATASET_PATTERN` (defaults to `D*_P*`).

Each dataset folder should resemble `D123_P1/000000` containing
`frame_info.json`, optionally sharded `frame_info_*.json` metadata, and image
assets used by `SingleMazeDataset`.

### BEV Module

- Default location: `bev_generation/data/`
- Override via env vars: set `BEV_DATA_ROOT` and optionally
  `BEV_DATASET_PATTERN` (defaults to `D*`).

A dataset directory holds `fps_views/`, `bev_views/`, metadata JSONs, and
`env_config.json`.  The loader normalizes legacy and new formats so long as the
required fields are present.

---

## 1. Grid-Cell Module

Located in `grid_cell/`, this module contains the training scripts, supporting
models, evaluation tooling, and manuscript video reproduction pipelines.

### Key Scripts

| Purpose | Command Skeleton |
| --- | --- |
| Train main model | `python train_grid_cells.py --gpus 0,1` (set `GRID_DATA_ROOT` first) |
| Run gridness analysis | `python run_gridness_analysis.py` (interactive directory selection) |
| Visualize grid cells | `python grid_cell_visualization.py --model_path <ckpt> [--split val]` |
| Trajectory comparison figure | `python vis_cell.py --checkpoint <ckpt> --trajectory <traj_dir> --output comparison.png` |
| Supplementary Video 5 | `python generate_video5.py --checkpoint_dirs <dir ...> --output_dir demo-output/video5` |
| Trajectory comparison video | `python generate_traj_comparison_video.py --checkpoint_dir <dir> --trajectory_dir <traj> --epochs 0 5 50 150` |

#### Training

`train_grid_cells.py` exposes flags:

- `--gpus`: comma-separated GPU IDs (defaults to `0`).
- `--resume`: optional checkpoint to continue training.
- `--disable_grid_loss`: remove grid regularization for ablation studies.

The script automatically handles single- or multi-GPU training via DDP and will
store run outputs beneath `grid_cell/log_train_grid_cells/`.

#### Evaluation & Visualizations

- `grid_cell_visualization.py` renders rate maps and grid alignment statistics.
- `generate_video5.py` composes the dual-panel supplementary video (heatmap
  evolution + trajectories).  Specify one or more checkpoint directories (e.g.
  `log_train_grid_cells/YYYYMMDD_HHMMSS/checkpoints`).
- `generate_traj_comparison_video.py` compares multiple epochs side-by-side.
- `vis_cell.py` generates a static comparison plot for a single trajectory.

Each script accepts CLI options to change fonts, datasets, trajectory limits,
and output destinations.  All hard-coded absolute paths from the original
environment were removed; rely on CLI arguments or environment variables.

### Requirements

Install dependencies via:

```bash
pip install -r grid_cell/requirements.txt
```

This covers PyTorch, NumPy/SciPy, Matplotlib, and OpenCV (used when videos are
rendered).

---

## 2. BEV Generation Module

The `bev_generation/` directory contains the BEV predictor architecture,
training harness, and visualization pipeline used to generate BEV trajectories
from first-person views.

### Key Scripts

| Purpose | Command Skeleton |
| --- | --- |
| Train BEV predictor | `python train.py` (uses `Config`; override via env vars) |
| Multi-GPU training | `python train_multi_gpu.py --config <cfg_json>` |
| Train target detector | `python train_target_detector.py` |
| Run inference + videos | `python visualization_output.py --bev_model <ckpt> --target_model <ckpt> --dataset <seq> --output_dir out --clean_output` |
| Mapping / SLAM viz | `python run_visualization.py --dataset_path <seq> --model_path <bev_ckpt>` |

`visualization_output.py` now exposes CLI flags instead of hard-coded paths and
also honors environment variables (`BEV_MODEL_PATH`, `TARGET_MODEL_PATH`,
`BEV_DATASET_PATH`, `BEV_OUTPUT_DIR`).

`run_visualization.py` provides an entry point for interactive BEV mapping or
SLAM reconstruction videos (set `BEV_MODEL_PATH` or pass `--model_path`).

`datasets/maze_dataset.py` includes a debug utility that can be launched with
custom maze directories:

```bash
python datasets/maze_dataset.py \
  --base_dir /path/to/D*/ \
  --maze_ids 000004 000005 \
  --num_frames 50 \
  --save_dir debug/rotation
```

### Requirements

```bash
pip install -r bev_generation/requirements.txt
```

This adds Matplotlib and OpenCV to the original predictor requirements so the
visualization pipeline works out of the box.

---

## TODO Modules (Planned Releases)

The following modules are pending cleanup and integration.  They are listed here
per the request so that the GitHub project documents future work clearly.

- [ ] Spatial memory reconstruction
- [ ] HRL-ICM -based maze policies
- [ ] Information Bottleneck (IB) compression pipeline

---

## Reproducing Paper Results

1. Prepare datasets according to the instructions above.
2. Train the grid-cell module or use existing checkpoints.
3. Run gridness analyses and generate the supplementary videos (Video 5 plus the
   trajectory comparison) using the provided commands.
4. Train or reuse BEV predictor weights, then render BEV videos via
   `visualization_output.py`.
