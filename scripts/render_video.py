"""
Side-by-side fly-through (Scaffold-GS | SCGC) along the ordered test (+train) cameras.
  python scripts/render_video.py outputs/nerf_synthetic/lego --out results/figures/lego.gif
"""
import argparse, os, sys
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import prefilter_voxel, render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402


def load(model_path):
    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    cfg = eval(open(os.path.join(model_path, "cfg_args")).read(), {"Namespace": argparse.Namespace})
    args = p.parse_args([])
    for k, v in vars(cfg).items():
        setattr(args, k, v)
    args.model_path = model_path
    ds = lp.extract(args)
    g = GaussianModel(ds.feat_dim, ds.n_offsets, ds.voxel_size, ds.update_depth, ds.update_init_factor,
                      ds.update_hierachy_factor, ds.use_feat_bank, ds.appearance_dim, ds.ratio,
                      ds.add_opacity_dist, ds.add_cov_dist, ds.add_color_dist)
    scene = Scene(ds, g, load_iteration=-1, shuffle=False)
    g.eval()
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    return g, scene, pp.extract(args), bg


@torch.no_grad()
def frames(g, cams, pipe, bg):
    for cam in cams:
        mask = prefilter_voxel(cam, g, pipe, bg)
        img = render(cam, g, pipe, bg, visible_mask=mask)["render"].clamp(0, 1)
        yield (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 9 * len(text) + 12, 22], fill=(0, 0, 0))
    dr.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--seed", default="seed0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=400)
    a = ap.parse_args()

    ga, sa, pipe, bg = load(os.path.join(a.scene_dir, "scaffold", a.seed))
    cams = sorted(sa.getTestCameras() + sa.getTrainCameras(), key=lambda c: c.image_name)
    if "nerf_synthetic" in a.scene_dir:   # Blender test cameras already form a smooth orbit
        cams = sorted(sa.getTestCameras(), key=lambda c: int("".join(filter(str.isdigit, c.image_name)) or 0))
    cams = cams[:: max(1, len(cams) // a.max_frames)][: a.max_frames]
    fa = list(frames(ga, cams, pipe, bg))
    del ga, sa
    torch.cuda.empty_cache()
    gb, _, _, _ = load(os.path.join(a.scene_dir, "scgc", a.seed))
    fb = list(frames(gb, cams, pipe, bg))

    out = []
    for x, y in zip(fa, fb):
        h, w, _ = x.shape
        nh = int(h * a.width / w)
        rs = lambda im: np.asarray(Image.fromarray(im).resize((a.width, nh), Image.LANCZOS))
        out.append(np.concatenate([label(rs(x), "Scaffold-GS"), np.full((nh, 4, 3), 255, np.uint8), label(rs(y), "SCGC")], 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    imageio.mimsave(a.out, out, duration=1 / 20, loop=0)
    if a.out.endswith(".gif"):
        imageio.mimsave(a.out[:-4] + ".mp4", out, fps=20, quality=8)
    print("wrote", a.out, len(out), "frames")


if __name__ == "__main__":
    main()
