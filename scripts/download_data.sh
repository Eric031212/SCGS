#!/usr/bin/env bash
# NeRF-Synthetic (Blender) scenes -> data/nerf_synthetic/<scene>
set -euo pipefail
mkdir -p data && cd data
[ -d nerf_synthetic/lego ] && { echo "already present"; exit 0; }
# official Google Drive file (often over quota) -> byte-identical-size backup on Hugging Face
gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O nerf_synthetic.zip || \
  wget -q https://huggingface.co/donydchen/matchnerf/resolve/main/nerf_synthetic.zip -O nerf_synthetic.zip
unzip -q nerf_synthetic.zip && rm nerf_synthetic.zip
ls nerf_synthetic
