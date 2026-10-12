"""
Rendering speed / model size of Scaffold-GS vs SCGC on one GPU (seed-0 models), so that the
mixed-GPU training runs do not affect the comparison.  -> results/fps.json
"""
import glob, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_video import load  # noqa: E402
from gaussian_renderer import prefilter_voxel, render  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def fps(model_path, reps=3):
    g, scene, pipe, bg = load(model_path)
    cams = scene.getTestCameras()
    for c in cams[:5]:
        render(c, g, pipe, bg, visible_mask=prefilter_voxel(c, g, pipe, bg))
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        for c in cams:
            render(c, g, pipe, bg, visible_mask=prefilter_voxel(c, g, pipe, bg))
    torch.cuda.synchronize()
    out = {"fps": reps * len(cams) / (time.time() - t), "anchors": int(g.get_anchor.shape[0])}
    del g, scene
    torch.cuda.empty_cache()
    return out


def main():
    res = {"gpu": torch.cuda.get_device_name(0)}
    for d in sorted(glob.glob(os.path.join(ROOT, "outputs/*/*/scaffold/seed0"))):
        scene_dir = os.path.dirname(os.path.dirname(d))
        key = os.path.relpath(scene_dir, os.path.join(ROOT, "outputs"))
        for exp in ["scaffold", "scgc"]:
            p = os.path.join(scene_dir, exp, "seed0")
            if os.path.exists(os.path.join(p, "point_cloud")):
                try:
                    res.setdefault(key, {})[exp] = fps(p)
                    print(key, exp, res[key][exp], flush=True)
                except Exception as e:  # keep going; one broken model should not stop the benchmark
                    print("skip", p, e, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "fps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
