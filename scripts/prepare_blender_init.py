"""Writes the random 10k-point initialisation (seed 0) for every Blender scene once, so concurrent
jobs do not race on data/nerf_synthetic/<scene>/points3d.ply and all methods share the same init."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.dataset_readers import storePly
from utils.sh_utils import SH2RGB

root = sys.argv[1] if len(sys.argv) > 1 else "data/nerf_synthetic"
for scene in sorted(os.listdir(root)):
    path = os.path.join(root, scene, "points3d.ply")
    if os.path.isdir(os.path.join(root, scene)) and not os.path.exists(path):
        rng = np.random.default_rng(0)
        xyz = rng.random((10_000, 3)) * 2.6 - 1.3
        storePly(path, xyz, SH2RGB(rng.random((10_000, 3)) / 255.0) * 255)
        print("wrote", path)
