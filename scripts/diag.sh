#!/usr/bin/env bash
# 3k-iteration lego diagnostics: which SCGC component costs PSNR?   sbatch --array=1-6 scripts/diag.sh
#SBATCH -p gpus
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --exclude=gpu[1601-1605]
#SBATCH -o logs/%x-%A_%a.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgc
cfgs=(
  "L30_base|--ctrl_mode none"
  "L30_tf_cc|--ctrl_mode transformer --lambda_gc 0"
  "L30_full|--ctrl_mode transformer"
  "L30_full_gc0002|--ctrl_mode transformer --lambda_gc 0.002"
  "L30_full_gcdetach|--ctrl_mode transformer --gc_detach"
)
IFS='|' read -r name flags <<< "${cfgs[$((SLURM_ARRAY_TASK_ID-1))]}"
out=/ltmp/$USER/diag_$name; rm -rf $out
python train.py --eval -s data/nerf_synthetic/lego -m $out --voxel_size 0.001 --update_init_factor 4 \
  --appearance_dim 0 --iterations ${ITERS:-3000} --test_iterations ${ITERS:-3000} --save_iterations ${ITERS:-3000} --update_until ${UNTIL:-2500} \
  --contrast_from ${CFROM:-500} --port $((7000 + SLURM_ARRAY_TASK_ID)) $flags > /dev/null 2>&1
mkdir -p outputs/diag/$name; cp $out/results.json $out/outputs.log outputs/diag/$name/ 2>/dev/null
echo "$name $(python -c "import json;r=next(iter(json.load(open('outputs/diag/$name/results.json')).values()));print('PSNR %.2f SSIM %.4f LPIPS %.4f'%(r['PSNR'],r['SSIM'],r['LPIPS']))")"
grep "control drift" outputs/diag/$name/outputs.log | sed 's/.*INFO: //'
rm -rf $out
