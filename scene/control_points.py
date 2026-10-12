#
# SCGC: sparse control points, transformer refinement and dual-level contrastive learning
# on top of Scaffold-GS anchors (Xie & Zhang, VRCAI 2025).
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def farthest_point_sampling(points, n, max_input=200_000):
    """Greedy FPS on the GPU. Very large anchor sets are randomly subsampled first."""
    if points.shape[0] > max_input:
        points = points[torch.randperm(points.shape[0], device=points.device)[:max_input]]
    n = min(n, points.shape[0])
    idx = torch.zeros(n, dtype=torch.long, device=points.device)
    dist = torch.full((points.shape[0],), float("inf"), device=points.device)
    idx[0] = torch.randint(points.shape[0], (1,), device=points.device)
    for i in range(1, n):
        dist = torch.minimum(dist, (points - points[idx[i - 1]]).pow(2).sum(-1))
        idx[i] = torch.argmax(dist)
    return points[idx]


@torch.no_grad()
def knn(query, ref, k, chunk=65536):
    """Indices and squared distances of the k nearest `ref` points for every `query` point."""
    out_d, out_i = [], []
    for s in range(0, query.shape[0], chunk):
        d = torch.cdist(query[s:s + chunk], ref).pow(2)
        dk, ik = d.topk(k, dim=1, largest=False)
        out_d.append(dk)
        out_i.append(ik)
    return torch.cat(out_d), torch.cat(out_i)


def info_nce(z, z_pos, tau, neg_mask=None, normalize=True):
    """InfoNCE (Eq. 9): row i's positive is z_pos[i]; neg_mask[i, j]=False drops pair (i, j) from the denominator."""
    if normalize:
        z, z_pos = F.normalize(z, dim=-1), F.normalize(z_pos, dim=-1)
    logits = z @ z_pos.t() / tau
    if neg_mask is not None:
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~(neg_mask | eye), float("-inf"))
    return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


class ControlPoints(nn.Module):
    """
    512 sparse control points c_i = {p_i, f_i, r_i} initialised by FPS over the anchors.

    mode='transformer':  P' = P + Transformer(Linear(P) + E + Linear(f))      (Eq. 7)
    mode='free':         P' = P + Δ with a free learnable Δ (ablation "+ Control Points")

    Every anchor is linked to its k nearest control points; its displacement is the
    normalised Gaussian-weighted average of their displacements (Eq. 8):
        w_ij ∝ exp(-||p_i - a_j||² / 2(σ r_i)²),   Δa_j = Σ_i w_ij (p'_i - p_ref_i)
    Positions are handled in a normalised frame (anchor centre, unit = control spacing s),
    so σ, the perturbation std and the clamps are scale free across datasets.

    Stability: each update is clamped to `max_step` spacings and the total displacement from the FPS
    initialisation to `max_total` spacings. Anchors are not re-baked at every control update (only before
    densification), so the anchor displacement is bounded by the same clamp. Re-applying unbounded
    increments made the control points run away late in training on Tanks&Temples/train and bicycle.
    """

    def __init__(self, num=512, feat_dim=64, d_model=128, n_layers=2, n_heads=4, k=4,
                 sigma=1.0, mode="transformer", max_step=0.5, max_total=0.25):
        super().__init__()
        self.num, self.feat_dim, self.k, self.sigma, self.mode = num, feat_dim, k, sigma, mode
        self.max_step, self.max_total = max_step, max_total
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.register_buffer("pos", torch.zeros(num, 3))       # current control positions (world)
        self.register_buffer("pos_init", torch.zeros(num, 3))  # FPS initialisation
        self.register_buffer("pos_ref", torch.zeros(num, 3))   # control positions the anchors were last baked at
        self.register_buffer("center", torch.zeros(3))
        self.register_buffer("spacing", torch.ones(()))         # mean NN distance between control points
        self.anchor_idx = None                                   # [N, k] control indices per anchor

    def init_from_anchors(self, anchors):
        dev = anchors.device
        pos = farthest_point_sampling(anchors.detach(), self.num)
        self.num = pos.shape[0]
        self.pos = pos
        self.pos_init = pos.clone()
        self.pos_ref = pos.clone()
        self.center = anchors.detach().mean(0)
        nn_d, _ = knn(pos, pos, 2)
        self.spacing = nn_d[:, 1].sqrt().mean().clamp_min(1e-6)

        self.feat = nn.Parameter(torch.randn(self.num, self.feat_dim, device=dev))
        self.log_radius = nn.Parameter(torch.zeros(self.num, device=dev))
        if self.mode == "transformer":
            self.in_proj = nn.Linear(3, self.d_model)
            self.feat_proj = nn.Linear(self.feat_dim, self.d_model)
            self.pos_emb = nn.Parameter(torch.randn(self.num, self.d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, dim_feedforward=2 * self.d_model,
                                               dropout=0.0, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, self.n_layers)
            self.head = nn.Linear(self.d_model, 3)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        elif self.mode == "free":
            self.delta = nn.Parameter(torch.zeros(self.num, 3))
        # contrastive projection g([p̂, f]) -> 64
        self.proj = nn.Sequential(nn.Linear(3 + self.feat_dim, 128), nn.ReLU(True), nn.Linear(128, 64))
        self.to(dev)
        self.assign(anchors)

    # ---------------------------------------------------------------- geometry
    def normalise(self, x):
        return (x - self.center) / self.spacing

    def refined(self):
        """Refined positions P' (world units) and the per-point step in normalised units."""
        if self.mode == "transformer":
            tok = self.in_proj(self.normalise(self.pos) / 10.0) + self.pos_emb + self.feat_proj(self.feat)
            step = self.head(self.transformer(tok[None]))[0]
        else:
            step = self.delta
        step = self.max_step * torch.tanh(step / self.max_step)  # conservative update
        # bound the total displacement from the FPS initialisation (smooth radial clamp)
        total = (self.pos - self.pos_init) / self.spacing + step
        n = total.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        total = total * (self.max_total * torch.tanh(n / self.max_total) / n)
        return self.pos_init + total * self.spacing, step

    @torch.no_grad()
    def assign(self, anchors):
        """KNN assignment of anchors to control points (re-run every 500 its and after densification)."""
        _, self.anchor_idx = knn(anchors.detach(), self.pos, self.k)

    def anchor_delta(self, anchors):
        """Differentiable displacement for every anchor, and the control step used for L_ctrl."""
        new_pos, step = self.refined()
        nb = self.pos[self.anchor_idx]                                    # [N, k, 3]
        d2 = (anchors.detach()[:, None] - nb).pow(2).sum(-1) / self.spacing ** 2
        sig = self.sigma * self.log_radius.exp()[self.anchor_idx]
        w = torch.softmax(-d2 / (2 * sig ** 2), dim=1)                   # normalised over neighbours
        disp = (new_pos - self.pos_ref)[self.anchor_idx]                  # [N, k, 3]
        return (w[..., None] * disp).sum(1), step

    @torch.no_grad()
    def commit(self):
        """Control point update (every 100 its): P <- P'."""
        new_pos, _ = self.refined()
        self.pos.copy_(new_pos)

    @torch.no_grad()
    def rebase(self):
        """Called right after the anchors were baked (before densification / at the end)."""
        new_pos, _ = self.refined()
        self.pos_ref.copy_(new_pos)

    # ---------------------------------------------------------- contrastive
    def control_contrast(self, tau=0.1, perturb=0.01, n_exclude=8):
        """Level 1: positives = jittered copies (std 0.01 ≈ σ=0.01 on Blender), negatives = all but the 8 nearest points."""
        p = self.normalise(self.refined()[0]) / 10.0
        z = self.proj(torch.cat([p, self.feat], -1))
        z_pos = self.proj(torch.cat([p + perturb * torch.randn_like(p), self.feat], -1))
        with torch.no_grad():
            _, near = knn(p, p, n_exclude + 1)
            neg_mask = torch.ones(p.shape[0], p.shape[0], dtype=torch.bool, device=p.device)
            neg_mask.scatter_(1, near, False)
        return info_nce(z, z_pos, tau, neg_mask)

    def capture(self):
        return {"state": self.state_dict(), "anchor_idx": self.anchor_idx}


def gaussian_contrast(image, image_pert, gt, tau=0.1, n_patches=128, patch=16, min_std=0.02):
    """
    Level 2: patches of the normal render vs. the same patches rendered with jittered Gaussian
    scale/opacity are positives; patches from other image regions are negatives. Only textured
    regions (GT patch std > min_std) are sampled so flat background cannot dominate.
    """
    with torch.no_grad():
        g = gt.mean(0, keepdim=True)[None]
        std = (F.avg_pool2d(g * g, patch, patch) - F.avg_pool2d(g, patch, patch) ** 2).clamp_min(0).sqrt()[0, 0]
        cand = torch.nonzero(std.flatten() > min_std).flatten()
        if cand.numel() < 8:
            return image.new_zeros(())
        cand = cand[torch.randperm(cand.numel(), device=cand.device)[:n_patches]]
    def patches(img):
        p = F.unfold(img[None], patch, stride=patch)[0].t()[cand]          # [M, 3*patch*patch]
        p = p - p.mean(1, keepdim=True)
        # cosine features with a floor at the texture threshold: a flat (not yet fitted) render patch
        # gets a ~0 feature instead of a normalised noise vector whose gradient scales like 1/||p||
        return p / (p.pow(2).sum(1, keepdim=True) + p.shape[1] * min_std ** 2).sqrt()

    return info_nce(patches(image), patches(image_pert), tau, normalize=False)
