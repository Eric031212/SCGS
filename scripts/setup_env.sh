#!/usr/bin/env bash
# Creates the conda env `scgc` and builds the CUDA extensions. Run on a GPU node:
#   srun -p gpus --gres=gpu:1 -c 8 --mem=32G bash scripts/setup_env.sh
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^scgc '; then
  conda create -y -n scgc python=3.10
fi
conda activate scgc
python -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir plyfile tqdm einops opencv-python-headless lpips laspy jaxtyping colorama \
    "numpy<2" matplotlib pandas imageio imageio-ffmpeg gdown tensorboard
python -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
bash scripts/build_ext.sh
