"""
Figures for the README / paper from finished runs (run on the cluster, the PNGs are synced back):
  results/figures/compare_<scene>.png     GT | Scaffold-GS | SCGC with a zoomed crop and error maps
  results/figures/blender_delta.png       per-scene ΔPSNR (SCGC − Scaffold-GS), mean ± std over seeds
  results/figures/real_psnr.png           per-scene PSNR on the real datasets
  results/figures/ablations.png           progressive components / #control points / transformer size
  results/figures/control_points_<scene>.png   anchors + control points coloured by displacement
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
OUTP = os.path.join(ROOT, "outputs")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from collect_results import BLENDER, REAL, NAMES, index, load_runs  # noqa: E402

# reference categorical slots 1, 2 (validated light-mode palette); text stays in ink colours
C_BASE, C_OURS = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
                     "savefig.bbox": "tight", "legend.frameon": False})


def run_dir(dataset, scene, exp, seed=0):
    return os.path.join(OUTP, dataset, scene, exp, f"seed{seed}")


def test_dir(d):
    m = sorted(glob.glob(os.path.join(d, "test", "ours_*")))
    return m[-1] if m else None


def psnr(a, b):
    return -10 * np.log10(np.mean((a - b) ** 2) + 1e-12)


# --------------------------------------------------------------------------- qualitative
def compare(dataset, scene, crop=0.22):
    a, b = test_dir(run_dir(dataset, scene, "scaffold")), test_dir(run_dir(dataset, scene, "scgc"))
    if not (a and b):
        return
    pa = json.load(open(os.path.join(run_dir(dataset, scene, "scaffold"), "per_view.json")))
    pb = json.load(open(os.path.join(run_dir(dataset, scene, "scgc"), "per_view.json")))
    ka, kb = next(iter(pa)), next(iter(pb))
    names = sorted(pa[ka]["PSNR"])
    gain = np.array([pb[kb]["PSNR"][n] - pa[ka]["PSNR"][n] for n in names])
    # one view where SCGC helps most and the view with the median gain (shown honestly side by side)
    picks = [("largest gain", names[int(np.argmax(gain))]), ("median view", names[int(np.argsort(gain)[len(gain) // 2])])]

    load = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    fig, axes = plt.subplots(len(picks), 5, figsize=(13, 2.75 * len(picks)), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.18})
    for r, (label, n) in enumerate(picks):
        gt, ra, rb = load(os.path.join(a, "gt", n)), load(os.path.join(a, "renders", n)), load(os.path.join(b, "renders", n))
        H, W, _ = gt.shape
        ch, cw = int(H * crop), int(W * crop)
        ea, eb = np.abs(ra - gt).mean(-1), np.abs(rb - gt).mean(-1)
        # crop where the methods differ most, tie-broken by total error (never on empty background)
        score = np.abs(ea - eb) + 0.25 * (ea + eb)
        cs = np.cumsum(np.cumsum(np.pad(score, ((1, 0), (1, 0))), 0), 1)
        box = cs[ch:, cw:] - cs[:-ch, cw:] - cs[ch:, :-cw] + cs[:-ch, :-cw]
        y, x = np.unravel_index(np.argmax(box), box.shape)
        panels = [(gt, "Ground truth"), (ra, f"Scaffold-GS  {psnr(ra, gt):.2f} dB"), (rb, f"SCGC  {psnr(rb, gt):.2f} dB"),
                  (np.clip(4 * ea, 0, 1), "|error| ×4, Scaffold-GS"), (np.clip(4 * eb, 0, 1), "|error| ×4, SCGC")]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="magma" if img.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.add_patch(Rectangle((x, y), cw, ch, fill=False, ec="#eda100", lw=1.2))
            ax.set_title(title, color=INK, fontsize=8.5)
            if c < 3:
                inset = ax.inset_axes([0.6, 0.0, 0.4, 0.4])
                inset.imshow(img[y:y + ch, x:x + cw]); inset.set_xticks([]); inset.set_yticks([])
                for sp in inset.spines.values():
                    sp.set_edgecolor("#eda100"); sp.set_linewidth(1.2)
        axes[r, 0].set_ylabel(f"{scene} · view {n[:-4]}\n({label})", color=INK2, fontsize=8.5)
    fig.savefig(os.path.join(FIG, f"compare_{scene}.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- quantitative
def blender_delta(deltas):
    scenes = [s for s in BLENDER if s in deltas]
    if not scenes:
        return
    m = np.array([np.mean(deltas[s]["PSNR"]) for s in scenes])
    e = np.array([np.std(deltas[s]["PSNR"], ddof=1) if len(deltas[s]["PSNR"]) > 1 else 0 for s in scenes])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar(scenes, m, width=0.55, color=C_OURS, yerr=e, capsize=3, ecolor=INK2, error_kw={"lw": 1})
    ax.axhline(0, color=INK, lw=0.8)
    for i, v in enumerate(m):
        ax.annotate(f"{v:+.2f}", (i, v), xytext=(0, 4 if v >= 0 else -11), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    n = len(deltas[scenes[0]]["PSNR"])
    ax.set_ylabel("ΔPSNR (dB)")
    ax.set_title(f"SCGC − Scaffold-GS on NeRF-Synthetic (mean ± std over {n} paired seeds)", color=INK, fontsize=9.5, loc="left")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "blender_delta.png"))
    plt.close(fig)


def real_psnr(d):
    scenes = [(ds, s) for ds, ss in REAL.items() for s in ss if d.get((ds, s, "scaffold", "PSNR")) and d.get((ds, s, "scgc", "PSNR"))]
    if not scenes:
        return
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(9, 3))
    for off, exp, c in [(-0.19, "scaffold", C_BASE), (0.19, "scgc", C_OURS)]:
        ax.bar(x + off, [np.mean(d[(ds, s, exp, "PSNR")]) for ds, s in scenes], width=0.36, color=c,
               label=NAMES[exp].replace(" (ours)", ""), edgecolor="white", linewidth=1)
    lo = min(np.mean(d[(ds, s, e, "PSNR")]) for ds, s in scenes for e in ["scaffold", "scgc"])
    ax.set_ylim(np.floor(lo) - 1, None)
    ax.set_xticks(x, [s for _, s in scenes], rotation=30, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.grid(axis="x", visible=False)
    # dataset separators
    pos = 0
    for ds, ss in REAL.items():
        k = sum(1 for dd, _ in scenes if dd == ds)
        if k:
            ax.text(pos + (k - 1) / 2, ax.get_ylim()[1], {"mipnerf360": "Mip-NeRF 360", "tandt": "T&T", "db": "DB"}[ds],
                    ha="center", va="bottom", color=INK2, fontsize=8)
            if pos:
                ax.axvline(pos - 0.5, color=GRID, lw=1)
        pos += k
    fig.savefig(os.path.join(FIG, "real_psnr.png"))
    plt.close(fig)


def ablations(d):
    def avg(exp, m="PSNR"):
        v = [d[("nerf_synthetic", s, exp, m)][0] for s in BLENDER if d.get(("nerf_synthetic", s, exp, m))]
        return np.mean(v) if len(v) == len(BLENDER) else np.nan

    panels = [("Progressive components", ["scaffold", "abl_cp", "abl_tf", "abl_cc", "scgc"]),
              ("# control points", ["ctrl256", "scgc", "ctrl1024"]),
              ("Transformer size", ["tf_1l2h", "scgc", "tf_4l4h", "tf_2l8h"]),
              ("Loss weights λ1/λ2/λ3", ["lw_low", "scgc", "lw_mid", "lw_high"])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
    vals_all = [avg(e) for _, es in panels for e in es]
    if np.all(np.isnan(vals_all)):
        plt.close(fig)
        return
    lo, hi = np.nanmin(vals_all), np.nanmax(vals_all)
    for ax, (title, exps) in zip(axes, panels):
        v = [avg(e) for e in exps]
        labels = [{"scgc": "SCGC\n(default)", "scaffold": "Scaffold\n-GS"}.get(e, NAMES[e].replace("+ ", "+").replace(" / ", "/\n")) for e in exps]
        ax.bar(range(len(exps)), v, width=0.55, color=[C_BASE if e == "scaffold" else C_OURS for e in exps])
        for i, y in enumerate(v):
            if not np.isnan(y):
                ax.annotate(f"{y:.2f}", (i, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(exps)), labels, fontsize=7.5)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(lo - 0.15, hi + 0.12)
    axes[0].set_ylabel("Avg PSNR, NeRF-Synthetic (dB)")
    fig.savefig(os.path.join(FIG, "ablations.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- control points
def control_points(dataset, scene):
    import torch
    from plyfile import PlyData
    d = run_dir(dataset, scene, "scgc")
    its = sorted(glob.glob(os.path.join(d, "point_cloud", "iteration_*")), key=lambda p: int(p.split("_")[-1]))
    if not its or not os.path.exists(os.path.join(its[-1], "control_points.pth")):
        return
    st = torch.load(os.path.join(its[-1], "control_points.pth"), map_location="cpu")["state"]
    p0, p1 = st["pos_init"].numpy(), st["pos"].numpy()
    v = PlyData.read(os.path.join(its[-1], "point_cloud.ply"))["vertex"]
    anchors = np.stack([v["x"], v["y"], v["z"]], 1)
    anchors = anchors[np.random.default_rng(0).choice(len(anchors), min(30000, len(anchors)), replace=False)]
    disp = np.linalg.norm(p1 - p0, axis=1) / float(st["spacing"])

    fig = plt.figure(figsize=(9.5, 4.2))
    for i, (elev, azim) in enumerate([(20, 45), (60, 135)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(*anchors.T, s=0.2, c="#b8b7b1", alpha=0.35, linewidths=0)
        sc = ax.scatter(*p1.T, s=9, c=disp, cmap="viridis", linewidths=0.3, edgecolors="white")
        ax.view_init(elev, azim)
        ax.set_axis_off()
    cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
    cb.set_label("control-point displacement (× spacing)", color=INK2)
    fig.suptitle(f"{scene}: {len(anchors)} sampled anchors (grey) and {len(p1)} control points after training",
                 color=INK, fontsize=9.5, x=0.45)
    fig.savefig(os.path.join(FIG, f"control_points_{scene}.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    d = index(load_runs())
    df = os.path.join(ROOT, "results", "blender_deltas.json")
    if os.path.exists(df):
        blender_delta(json.load(open(df)))
    real_psnr(d)
    ablations(d)
    for s in BLENDER:
        compare("nerf_synthetic", s)
    for ds, ss in REAL.items():
        for s in ss:
            compare(ds, s)
    for s in ["lego", "materials", "drums", "ship"]:
        control_points("nerf_synthetic", s)
    for s in ["bicycle", "garden", "room"]:
        control_points("mipnerf360", s)
    print("figures ->", FIG, sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
