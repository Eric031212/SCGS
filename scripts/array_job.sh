#!/usr/bin/env bash
# sbatch --array=1-N%32 [--gres=gpu:geforce_gtx_2080_ti:1] scripts/array_job.sh jobs/<group>.txt
# Runs line $SLURM_ARRAY_TASK_ID. Training writes to node-local /ltmp; only metrics/configs/logs are copied
# back to outputs/ (plus test renders + point clouds for seed 0 of the main comparison, used by the figures).
#SBATCH -p gpus
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%A_%a.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgc
cd "${SLURM_SUBMIT_DIR:-.}"
ulimit -n 4096

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
out=$(echo "$CMD" | grep -o -- '-m [^ ]*' | cut -d' ' -f2)
TMP=/ltmp/$USER/scgc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"
CMD=${CMD/-m outputs\//-m $TMP/outputs/}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "[$(date)] $(hostname) GPU: $GPU"
echo "+ $CMD"
start=$(date +%s)
eval "$CMD"
status=$?

tmp_out="$TMP/$out"
if [ -f "$tmp_out/results.json" ]; then
  printf '{"gpu": "%s", "train_eval_seconds": %d}\n' "$GPU" $(( $(date +%s) - start )) > "$tmp_out/job_info.json"
  keep=(--include='*/' --include='results.json' --include='per_view.json' --include='cfg_args'
        --include='outputs.log' --include='job_info.json' --include='events.out.tfevents*')
  case "$out" in
    */scaffold/seed0|*/scgc/seed0)
      keep+=(--include='test/*/renders/*' --include='test/*/gt/*' --include='point_cloud/**') ;;
  esac
  mkdir -p "$out"
  rsync -a "${keep[@]}" --exclude='*' --prune-empty-dirs "$tmp_out/" "$out/"
  echo "copied back to $out ($(du -sh "$out" | cut -f1))"
fi
exit $status
