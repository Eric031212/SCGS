#!/usr/bin/env bash
# Fly-through GIFs (Scaffold-GS | SCGC) for the given scenes:  bash scripts/render_flythroughs.sh nerf_synthetic/lego ...
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgc
for s in "$@"; do
  python scripts/render_video.py outputs/$s --out results/figures/flythrough_$(basename $s).gif --max_frames 60 --width 320 2>&1 | grep -E "wrote|Error"
done
