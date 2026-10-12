#!/usr/bin/env bash
# 3k-iteration end-to-end check of the baseline and full SCGC on lego (densification, KNN reassignment,
# both contrastive terms, commit, save, render, metrics).
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgc
python scripts/prepare_blender_init.py data/nerf_synthetic
for exp in "none" "transformer"; do
  python train.py --eval -s data/nerf_synthetic/lego -m outputs/smoke/lego_$exp --voxel_size 0.001 --update_init_factor 4 \
    --appearance_dim 0 --iterations 3000 --test_iterations 3000 --save_iterations 3000 --update_until 2500 \
    --contrast_from 500 --ctrl_mode $exp --port $((7000 + RANDOM % 1000))
  cat outputs/smoke/lego_$exp/results.json
done
