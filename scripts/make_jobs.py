"""Writes jobs/<group>.txt: one training command per line, consumed by scripts/array_job.sh."""
import os

BLENDER = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
M360_OUT = ["bicycle", "garden", "stump", "flowers", "treehill"]
M360_IN = ["room", "counter", "kitchen", "bonsai"]

# Scaffold-GS official per-dataset settings; Mip-NeRF 360 uses the 3DGS images_4 / images_2 protocol.
DATA = {f"nerf_synthetic/{s}": "--voxel_size 0.001 --update_init_factor 4" for s in BLENDER}
DATA.update({f"mipnerf360/{s}": "--voxel_size 0.001 --update_init_factor 16 -i images_4 -r 1" for s in M360_OUT})
DATA.update({f"mipnerf360/{s}": "--voxel_size 0.001 --update_init_factor 16 -i images_2 -r 1" for s in M360_IN})
DATA.update({f"tandt/{s}": "--voxel_size 0.01 --update_init_factor 16" for s in ["truck", "train"]})
DATA.update({f"db/{s}": "--voxel_size 0.005 --update_init_factor 16" for s in ["playroom", "drjohnson"]})

FULL = "--ctrl_mode transformer"
EXPS = {
    "scaffold": "--ctrl_mode none",
    "scgc": FULL,                                   # λ3 = 0.002 (default)
    "scgc_paper": FULL + " --lambda_gc 0.02",      # λ3 as printed in the paper
    # Table 5: progressive components
    "abl_cp": "--ctrl_mode free --lambda_cc 0 --lambda_gc 0",
    "abl_tf": "--ctrl_mode transformer --lambda_cc 0 --lambda_gc 0",
    "abl_cc": "--ctrl_mode transformer --lambda_gc 0",
    # Table 2: number of control points
    "ctrl256": FULL + " --num_ctrl 256",
    "ctrl1024": FULL + " --num_ctrl 1024",
    # Table 3: transformer size
    "tf_1l2h": FULL + " --ctrl_layers 1 --ctrl_heads 2",
    "tf_4l4h": FULL + " --ctrl_layers 4 --ctrl_heads 4",
    "tf_2l8h": FULL + " --ctrl_layers 2 --ctrl_heads 8",
    # Table 4: loss weights (λ1, λ2, λ3)
    "lw_low": FULL + " --lambda_ctrl 0.01 --lambda_cc 0.01 --lambda_gc 0.01",
    "lw_mid": FULL + " --lambda_ctrl 0.15 --lambda_cc 0.03 --lambda_gc 0.01",
    "lw_high": FULL + " --lambda_ctrl 0.30 --lambda_cc 0.10 --lambda_gc 0.05",
}


def cmd(scene, exp, seed):
    out = f"outputs/{scene}/{exp}/seed{seed}"
    return (f"[ -f {out}/results.json ] || python train.py --eval -s data/{scene} -m {out} {DATA[scene]} "
            f"--appearance_dim 0 --ratio 1 --seed {seed} --port {6000 + hash((scene, exp, seed)) % 50000} {EXPS[exp]}")


REAL_SCENES = [sc for sc in DATA if not sc.startswith("nerf")]
groups = {
    "blender_scaffold": [cmd(f"nerf_synthetic/{s}", "scaffold", k) for k in range(3) for s in BLENDER],
    "real_scaffold": [cmd(sc, "scaffold", 0) for sc in REAL_SCENES],
    "blender_scgc": [cmd(f"nerf_synthetic/{s}", "scgc", k) for k in range(3) for s in BLENDER],
    "real_scgc": [cmd(sc, "scgc", 0) for sc in REAL_SCENES],
    "blender_ablation": [cmd(f"nerf_synthetic/{s}", e, 0) for e in list(EXPS)[2:] for s in BLENDER],  # incl. scgc_paper
}
os.makedirs("jobs", exist_ok=True)
for g, lines in groups.items():
    with open(f"jobs/{g}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(g, len(lines))
