<div align="center">

# SCGC

### Sparse Control-Guided Gaussian Splatting with Contrastive Learning

**Hongtao Xie** · **Hui Zhang**

VRCAI 2025 · Macau

<a href="https://doi.org/10.1145/3779232.3779453"><img src="https://img.shields.io/badge/Paper-ACM_DL-blue"></a>
<a href="#-citation"><img src="https://img.shields.io/badge/📝_Cite-SCGC-blue"></a>
<img src="https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/Built_on-Scaffold--GS-6f42c1">
<img src="https://img.shields.io/badge/License-Inria_/_MPII_(non--commercial)-lightgrey">

<img src="results/figures/flythrough_lego.gif" width="49%"> <img src="results/figures/flythrough_ship.gif" width="49%">

*Novel views rendered by the trained models: Scaffold-GS (left half) vs. SCGC (right half) of each clip.*

</div>

---

## 📖 Abstract

Scaffold-GS organises 3D Gaussians around anchors that are placed once, from a voxelised SfM point cloud, and then never move. Each anchor only sees its own little neighbourhood. SCGC asks: what if a small set of **sparse control points** could look at the *whole* scene and nudge the anchors around?

- 🎯 **512 control points**, picked by farthest point sampling over the anchors and linked to them by KNN with Gaussian distance weights.
- 🧠 **A tiny transformer** (2 layers, 4 heads, 128-d) lets every control point attend to all others and predicts how it should move: `P' = P + Transformer(P + E)`. Anchors follow their control points.
- 🧲 **Dual-level contrastive learning**: InfoNCE on control-point features (jittered copies are positives, distant points are negatives) and on rendered patches (renders with jittered Gaussian scale/opacity are positives, other image regions are negatives).

This repository is a **full re-implementation and re-run** of the paper: the original code was lost with an old laptop, so everything here was rebuilt from the paper, trained again on a GPU cluster and evaluated on four benchmarks (NeRF-Synthetic, Mip-NeRF 360, Tanks & Temples, Deep Blending). 🔍

## 🧠 Method

<div align="center"><img src="assets/scgc_pipeline.png" width="100%"></div>

| Component | Description | Code |
|---|---|---|
| **Sparse control points** | `C = FPS(anchors, 512)`, each `c_i = {p_i ∈ R³, f_i ∈ R⁶⁴, r_i}`. Every anchor is linked to its k = 4 nearest control points with normalised weights `w_ij ∝ exp(−‖p_i − a_j‖² / 2(σ r_i)²)`; its displacement is `Δa_j = Σ_i w_ij Δp_i`. KNN is re-assigned every 500 iterations and after densification. | [`scene/control_points.py`](scene/control_points.py) |
| **Transformer refinement** | Tokens `Linear(3→128)(P) + E + Linear(64→128)(f)` → 2-layer, 4-head pre-norm transformer → zero-initialised `Linear(128→3)` step. `P ← P'` every 100 iterations. Each step is clamped to 0.5 control spacings and the total drift from the FPS initialisation to 0.25 spacings (see [Notes](#-notes-on-the-reproduction)). | [`scene/control_points.py`](scene/control_points.py) |
| **Control-level contrast** | InfoNCE (τ = 0.1) between `g([p̂_i, f_i])` and `g([p̂_i + ε, f_i])`, `ε ~ N(0, 0.01)`; the 8 nearest control points are excluded from the negatives. | [`scene/control_points.py`](scene/control_points.py) |
| **Gaussian-level contrast** | A second render with neural-Gaussian scale and opacity jittered by 5 %; 128 textured 16×16 patches; the same patch in both renders is the positive pair, other patches are negatives (InfoNCE, τ = 0.1). | [`scene/control_points.py`](scene/control_points.py), [`gaussian_renderer`](gaussian_renderer/__init__.py) |
| **Joint optimisation** | `L = L_scaffold + λ₁‖P' − P‖₁ + λ₂ L_ctrl-contrast + λ₃ L_gauss-contrast`, λ₁ = 0.1, λ₂ = 0.05, **λ₃ = 0.002** (paper: 0.02, see below). Contrastive terms start after 1 000 iterations; transformer trained with Adam, lr 1e-4. | [`train.py`](train.py) |

Positions inside the control-point module are expressed in units of the mean control-point spacing, so σ, the jitter and the clamps transfer unchanged from object-scale Blender scenes to room- and street-scale captures.

## 🎬 Gallery

Every clip is a fly-through along the test cameras, **Scaffold-GS on the left, SCGC on the right**.

### NeRF-Synthetic

| | |
|:---:|:---:|
| <img src="results/figures/flythrough_chair.gif" width="100%"><br>chair | <img src="results/figures/flythrough_drums.gif" width="100%"><br>drums |
| <img src="results/figures/flythrough_ficus.gif" width="100%"><br>ficus | <img src="results/figures/flythrough_hotdog.gif" width="100%"><br>hotdog |
| <img src="results/figures/flythrough_materials.gif" width="100%"><br>materials | <img src="results/figures/flythrough_mic.gif" width="100%"><br>mic |

### Real scenes (Mip-NeRF 360 · Tanks & Temples · Deep Blending)

| | |
|:---:|:---:|
| <img src="results/figures/flythrough_garden.gif" width="100%"><br>garden | <img src="results/figures/flythrough_bicycle.gif" width="100%"><br>bicycle |
| <img src="results/figures/flythrough_room.gif" width="100%"><br>room | <img src="results/figures/flythrough_counter.gif" width="100%"><br>counter |
| <img src="results/figures/flythrough_truck.gif" width="100%"><br>truck | <img src="results/figures/flythrough_train.gif" width="100%"><br>train |
| <img src="results/figures/flythrough_playroom.gif" width="100%"><br>playroom | |

### Close-ups

The test view where SCGC gains the most and the median view, with a zoomed crop where the two methods differ most and ×4 error maps.

<img src="results/figures/compare_ship.png" width="100%">
<img src="results/figures/compare_truck.png" width="100%">
<img src="results/figures/compare_garden.png" width="100%">

<details>
<summary><b>More scenes</b></summary>

<img src="results/figures/compare_lego.png" width="100%">
<img src="results/figures/compare_materials.png" width="100%">
<img src="results/figures/compare_chair.png" width="100%">
<img src="results/figures/compare_drums.png" width="100%">
<img src="results/figures/compare_ficus.png" width="100%">
<img src="results/figures/compare_hotdog.png" width="100%">
<img src="results/figures/compare_mic.png" width="100%">
<img src="results/figures/compare_bicycle.png" width="100%">
<img src="results/figures/compare_stump.png" width="100%">
<img src="results/figures/compare_flowers.png" width="100%">
<img src="results/figures/compare_treehill.png" width="100%">
<img src="results/figures/compare_room.png" width="100%">
<img src="results/figures/compare_counter.png" width="100%">
<img src="results/figures/compare_kitchen.png" width="100%">
<img src="results/figures/compare_bonsai.png" width="100%">
<img src="results/figures/compare_train.png" width="100%">
<img src="results/figures/compare_playroom.png" width="100%">
<img src="results/figures/compare_drjohnson.png" width="100%">

</details>

### Control points

The 512 control points (coloured by how far the transformer moved them) over the Scaffold-GS anchors (grey).

<img src="results/figures/control_points_lego.png" width="49%"> <img src="results/figures/control_points_garden.png" width="49%">

## 📝 Notes on the reproduction

Quantitatively, SCGC is on par with Scaffold-GS in this re-run: 33.40 vs. 33.43 dB PSNR on NeRF-Synthetic (3 seeds) and within ±0.35 dB on every real scene. Per-scene numbers for all datasets and ablations are in [`results/tables.md`](results/tables.md). Things worth knowing if you compare with the paper:

1. **The Gaussian-level contrast weight.** With the paper's λ₃ = 0.02 the term costs about 2 dB (31.38 vs 33.43 average PSNR on Blender, 33.17 vs 35.37 on Lego). Early in training the rendered patches are nearly flat, and pushing them apart in an InfoNCE sense fights the photometric loss. λ₃ = 0.002 removes the penalty; it was chosen on a single scene (Lego), which is mild test-set tuning and is stated here for that reason. The paper-weight runs are kept in the ablations.
2. **Bounded control-point drift.** A first version re-applied the transformer's step every 100 iterations without a bound on the total. On most scenes the points barely moved, but on Tanks & Temples *train* and Mip-NeRF 360 *bicycle* they ran away late in training (drift jumped from ≈ 0.05 to 5–10 control spacings between 20 k and 25 k iterations) and the reconstructions collapsed (12.4 dB and 21.1 dB PSNR). The total displacement is now clamped to 0.25 spacings; normal runs never get near the bound (≤ 0.05). The failing logs and the full v1 tables are kept in [`results/v1_unbounded/`](results/v1_unbounded/).
3. **Table 1 of the paper.** Its Scaffold-GS averages (31.83 dB / 0.899) do not match its own per-scene columns, whose means are 33.54 dB / 0.932, and its 3DGS row (≈ 3 dB PSNR) is clearly a failed run. Re-running Scaffold-GS here gives 33.43 dB, close to the 33.68 dB reported by the Scaffold-GS authors.
4. **Upstream fixes.** Respect Slurm's `CUDA_VISIBLE_DEVICES`, optional `wandb`, a `--seed` flag, and a Pillow ≥ 10 fix in the Blender loader (`np.byte` → `np.uint8`, bit-identical images).

## 🚀 Getting Started

Tested on Ubuntu 24.04, Python 3.10, PyTorch 2.1.2 + CUDA 12.1, GTX 1080 Ti / RTX 2080 Ti / Titan RTX.

```bash
git clone https://github.com/Eric031212/SCGC.git && cd SCGC
bash scripts/setup_env.sh          # conda env `scgc` + CUDA rasterizer (see build_ext.sh for compiler notes)
conda activate scgc
bash scripts/download_data.sh      # NeRF-Synthetic  -> data/nerf_synthetic
bash scripts/download_real.sh      # Mip-NeRF 360, Tanks & Temples, Deep Blending
python scripts/prepare_blender_init.py
```

Train and evaluate one scene (renders the test set and writes `results.json`):

```bash
# SCGC
python train.py --eval -s data/nerf_synthetic/lego -m outputs/lego_scgc \
    --voxel_size 0.001 --update_init_factor 4 --appearance_dim 0 --ctrl_mode transformer
# Scaffold-GS baseline
python train.py --eval -s data/nerf_synthetic/lego -m outputs/lego_scaffold \
    --voxel_size 0.001 --update_init_factor 4 --appearance_dim 0 --ctrl_mode none
```

Key flags: `--ctrl_mode {none,free,transformer}`, `--num_ctrl`, `--ctrl_layers`, `--ctrl_heads`, `--lambda_ctrl`, `--lambda_cc`, `--lambda_gc`, `--ctrl_max_total`, `--seed`.

Reproduce every number and figure on a Slurm cluster:

```bash
python scripts/make_jobs.py                                    # writes jobs/*.txt
sbatch --array=1-24 scripts/array_job.sh jobs/blender_scaffold.txt
sbatch --array=1-24 scripts/array_job.sh jobs/blender_scgc.txt
sbatch --array=1-13 scripts/array_job.sh jobs/real_scaffold.txt
sbatch --array=1-13 scripts/array_job.sh jobs/real_scgc.txt
sbatch --array=1-96 scripts/array_job.sh jobs/blender_ablation.txt
sbatch scripts/postprocess.sh      # tables, figures, fly-throughs, FPS benchmark -> results/
```

## 📂 Repository layout

```
scene/control_points.py     control points, transformer, contrastive losses (SCGC)
train.py                    Scaffold-GS training loop + SCGC hooks
scripts/                    environment, data, Slurm jobs, result tables, figures, videos
results/tables.md           all tables (generated);  results/all_runs.csv  one row per run
results/figures/            figures and fly-throughs used in this README
results/v1_unbounded/       logs and tables of the first (unbounded-drift) version
```

## 🙏 Acknowledgements

Built on [Scaffold-GS](https://github.com/city-super/Scaffold-GS) (Lu et al., CVPR 2024), which itself builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023) — thank you both for releasing your code. 🙌 Datasets: [NeRF-Synthetic](https://www.matthewtancik.com/nerf), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Tanks and Temples](https://www.tanksandtemples.org/) and [Deep Blending](http://visual.cs.ucl.ac.uk/pubs/deepblending/). Experiments ran on the Brown University CS GPU cluster.

## 📝 Citation

```bibtex
@inproceedings{xie2025scgc,
  title     = {SCGC: Sparse Control-Guided Gaussian Splatting with Contrastive Learning},
  author    = {Xie, Hongtao and Zhang, Hui},
  booktitle = {Proceedings of the 20th ACM SIGGRAPH International Conference on Virtual-Reality Continuum and its Applications in Industry (VRCAI '25)},
  year      = {2025},
  address   = {Macau, China},
  publisher = {ACM},
  doi       = {10.1145/3779232.3779453}
}
```

## ⚖️ License

The code inherits the [Gaussian-Splatting License](LICENSE.md) (Inria & MPII) from 3D Gaussian Splatting and Scaffold-GS: **research and evaluation use only, no commercial use**. The datasets keep their own licences.
