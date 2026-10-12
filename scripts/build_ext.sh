#!/usr/bin/env bash
# Rebuild only the CUDA extensions (diff-gaussian-rasterization, simple-knn) in the `scgc` env.
# Uses conda's nvcc 12.1 (same CUDA as the torch wheel): the system nvcc 12.0 front end fails on
# pybind11's cast.h, and so does any nvcc with the system g++ 12.3/13 headers.
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda activate scgc
if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
  conda install -y -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-cccl
fi
if [ ! -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  conda install -y -c conda-forge gcc_linux-64=11.2 gxx_linux-64=11.2
fi
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6;8.9"
# conda gcc 11.2 only as nvcc's host compiler for .cu files (torch passes `-ccbin $CC` to nvcc);
# system g++-12 compiles .cpp and links (conda's ld cannot read the system libmvec).
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=g++-12
export LDSHARED="g++-12 -shared"
python -m pip install "setuptools<70" wheel ninja
# build from node-local disk: wheel packaging fails on NFS (.nfs* silly-rename files)
B=/ltmp/$USER/scgc_build_$$; rm -rf "$B"; mkdir -p "$B"; trap 'rm -rf "$B"' EXIT
cp -r submodules/diff-gaussian-rasterization submodules/simple-knn "$B"/
rm -rf "$B"/*/build "$B"/*/*.egg-info
python -m pip install --no-build-isolation "$B"/diff-gaussian-rasterization "$B"/simple-knn
python -c "import torch, diff_gaussian_rasterization, simple_knn._C; print('ok', torch.__version__, torch.cuda.get_device_name(0))"
