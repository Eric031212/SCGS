"""
Collects outputs/<dataset>/<scene>/<exp>/seed<k>/results.json into
  results/all_runs.csv     one row per run
  results/tables.md        paper-style tables (mean ± std over seeds where available)
  results/summary.json     dataset averages per method
"""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
BLENDER = ["lego", "chair", "drums", "ficus", "hotdog", "materials", "mic", "ship"]  # paper's column order
REAL = {"mipnerf360": ["bicycle", "garden", "stump", "flowers", "treehill", "room", "counter", "kitchen", "bonsai"],
        "tandt": ["truck", "train"], "db": ["playroom", "drjohnson"]}
NAMES = {"scaffold": "Scaffold-GS", "scgc": "SCGC (ours)", "scgc_paper": "SCGC, paper λ3 = 0.02", "abl_cp": "+ Control Points", "abl_tf": "+ Transformer",
         "abl_cc": "+ Control Contrast", "ctrl256": "256 pts", "ctrl1024": "1024 pts", "tf_1l2h": "1L-2H",
         "tf_4l4h": "4L-4H", "tf_2l8h": "2L-8H", "lw_low": "0.01 / 0.01 / 0.01", "lw_mid": "0.15 / 0.03 / 0.01",
         "lw_high": "0.30 / 0.10 / 0.05"}
METRICS = ["PSNR", "SSIM", "LPIPS"]


def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "outputs/*/*/*/seed*/results.json"))):
        dataset, scene, exp, seed = f.split(os.sep)[-5:-1]
        res = next(iter(json.load(open(f)).values()))
        info_f = os.path.join(os.path.dirname(f), "job_info.json")
        info = json.load(open(info_f)) if os.path.exists(info_f) else {}
        rows.append(dict(dataset=dataset, scene=scene, exp=exp, seed=int(seed[4:]),
                         PSNR=res["PSNR"], SSIM=res["SSIM"], LPIPS=res["LPIPS"],
                         gpu=info.get("gpu", ""), seconds=info.get("train_eval_seconds", "")))
    return rows


def index(rows):
    d = defaultdict(list)  # (dataset, scene, exp, metric) -> values over seeds
    for r in rows:
        for m in METRICS:
            d[(r["dataset"], r["scene"], r["exp"], m)].append(r[m])
    return d


def fmt(vals, m, std=True):
    if not vals:
        return "–"
    p = 2 if m == "PSNR" else 3
    s = f"{np.mean(vals):.{p}f}"
    if std and len(vals) > 1:
        s += f" ± {np.std(vals, ddof=1):.{p}f}"
    return s


def scene_mean(d, dataset, scenes, exp, m, seed_avg=True):
    """Dataset average of per-scene means; None unless every scene is present."""
    v = [np.mean(d[(dataset, s, exp, m)]) for s in scenes if d.get((dataset, s, exp, m))]
    return float(np.mean(v)) if len(v) == len(scenes) else None


def per_seed_avg(d, dataset, scenes, exp, m):
    """Dataset average computed separately for each seed (for a std of the average)."""
    n = min((len(d.get((dataset, s, exp, m), [])) for s in scenes), default=0)
    return [float(np.mean([d[(dataset, s, exp, m)][k] for s in scenes])) for k in range(n)]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_runs()
    with open(os.path.join(OUT, "all_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        w.writeheader()
        w.writerows(rows)
    d = index(rows)
    md, summary = [], {}

    # ---- Table 1: NeRF-Synthetic per scene
    md.append("## NeRF-Synthetic (per scene, mean ± std over seeds)\n")
    for m in METRICS:
        md.append(f"**{m}** {'↑' if m != 'LPIPS' else '↓'}\n")
        md.append("| Method | " + " | ".join(s.capitalize() for s in BLENDER) + " | Avg |")
        md.append("|---" * (len(BLENDER) + 2) + "|")
        for exp in ["scaffold", "scgc"]:
            avg = per_seed_avg(d, "nerf_synthetic", BLENDER, exp, m)
            md.append(f"| {NAMES[exp]} | " + " | ".join(fmt(d.get(("nerf_synthetic", s, exp, m), []), m, std=False)
                                                       for s in BLENDER) + f" | {fmt(avg, m)} |")
        md.append("")

    # ---- paired difference per scene (same seed, same init)
    md.append("### SCGC − Scaffold-GS, paired by seed\n")
    md.append("| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | seeds where SCGC PSNR is higher |")
    md.append("|---|---|---|---|---|")
    deltas = {}
    for s in BLENDER + ["avg"]:
        if s == "avg":
            a = {m: per_seed_avg(d, "nerf_synthetic", BLENDER, "scaffold", m) for m in METRICS}
            b = {m: per_seed_avg(d, "nerf_synthetic", BLENDER, "scgc", m) for m in METRICS}
        else:
            a = {m: d.get(("nerf_synthetic", s, "scaffold", m), []) for m in METRICS}
            b = {m: d.get(("nerf_synthetic", s, "scgc", m), []) for m in METRICS}
        n = min(len(a["PSNR"]), len(b["PSNR"]))
        if n == 0:
            continue
        dd = {m: [b[m][k] - a[m][k] for k in range(n)] for m in METRICS}
        deltas[s] = dd
        wins = sum(x > 0 for x in dd["PSNR"])
        md.append(f"| {s.capitalize() if s != 'avg' else '**Average**'} | "
                  + " | ".join(("+" if np.mean(dd[m]) >= 0 else "") + fmt(dd[m], m) for m in METRICS)
                  + f" | {wins}/{n} |")
    md.append("")
    json.dump(deltas, open(os.path.join(OUT, "blender_deltas.json"), "w"), indent=1)

    # ---- real scenes
    for ds, scenes in REAL.items():
        title = {"mipnerf360": "Mip-NeRF 360", "tandt": "Tanks & Temples", "db": "Deep Blending"}[ds]
        md.append(f"## {title}\n")
        md.append("| Method | " + " | ".join(scenes) + " | Avg PSNR | Avg SSIM | Avg LPIPS |")
        md.append("|---" * (len(scenes) + 4) + "|")
        for exp in ["scaffold", "scgc"]:
            cells = [fmt(d.get((ds, s, exp, "PSNR"), []), "PSNR", std=False) for s in scenes]
            avgs = [scene_mean(d, ds, scenes, exp, m) for m in METRICS]
            md.append(f"| {NAMES[exp]} | " + " | ".join(cells) + " | "
                      + " | ".join(fmt([v], m) if v is not None else "–" for v, m in zip(avgs, METRICS)) + " |")
            summary.setdefault(ds, {})[exp] = dict(zip(METRICS, avgs))
        md.append("\n(per-scene cells are PSNR; one seed)\n")

    for exp in ["scaffold", "scgc"]:
        summary.setdefault("nerf_synthetic", {})[exp] = {m: scene_mean(d, "nerf_synthetic", BLENDER, exp, m) for m in METRICS}

    # ---- ablations (seed 0, NeRF-Synthetic average over the 8 scenes)
    def seed0(exp, m):
        v = [d[("nerf_synthetic", s, exp, m)][0] for s in BLENDER if d.get(("nerf_synthetic", s, exp, m))]
        return float(np.mean(v)) if len(v) == len(BLENDER) else None

    def abl_table(title, exps, cols=METRICS):
        md.append(f"## {title}\n")
        md.append("| Config | " + " | ".join(cols) + " |")
        md.append("|---" * (len(cols) + 1) + "|")
        for e in exps:
            vals = [seed0(e, m) for m in cols]
            md.append(f"| {NAMES[e]} | " + " | ".join(fmt([v], m) if v is not None else "–" for v, m in zip(vals, cols)) + " |")
        md.append("")

    md.append("# Ablations (NeRF-Synthetic, 8-scene average, seed 0)\n")
    abl_table("Progressive components (paper Table 5)", ["scaffold", "abl_cp", "abl_tf", "abl_cc", "scgc"])
    abl_table("Number of control points (paper Table 2)", ["ctrl256", "scgc", "ctrl1024"])
    abl_table("Transformer size (paper Table 3)", ["tf_1l2h", "scgc", "tf_4l4h", "tf_2l8h"])
    abl_table("Loss weights λ1 / λ2 / λ3 (paper Table 4; paper default 0.10 / 0.05 / 0.02, ours 0.10 / 0.05 / 0.002)",
              ["lw_low", "scgc_paper", "lw_mid", "lw_high", "scgc"])

    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"{len(rows)} runs -> results/tables.md")


if __name__ == "__main__":
    main()
