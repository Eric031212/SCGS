#!/usr/bin/env bash
# Runs after all training arrays finish:  tables, figures, fly-through GIFs, single-GPU FPS benchmark.
#SBATCH -p gpus
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH -o logs/%x-%j.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgc
cd "${SLURM_SUBMIT_DIR:-.}"
echo "[$(date)] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python scripts/collect_results.py
python scripts/make_figures.py
for s in nerf_synthetic/lego nerf_synthetic/ship nerf_synthetic/materials mipnerf360/garden mipnerf360/room mipnerf360/bicycle tandt/truck; do
  name=$(basename $s)
  python scripts/render_video.py outputs/$s --out results/figures/flythrough_${name}.gif --max_frames 60 --width 320 || echo "video failed: $s"
done
python scripts/benchmark_fps.py
echo "[$(date)] postprocess done"
ls -la results results/figures
