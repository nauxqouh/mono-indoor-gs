""" 
Multi-view consistency loss for 2DGS
followed 2DGS-Room (https://arxiv.org/abs/2412.03428)
"""

import math
from random import choice

import torch
import torch.nn.functional as F

def get_camera_intrinsics(cam):
    """Return (fx, fy, cx, cy, H, W) from a 3DGS/2DGS Camera object."""
    H = cam.image_height
    W = cam.image_width
    fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
    fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
    return fx, fy, W / 2.0, H / 2.0, H, W

def build_neighbor_table(cameras, k: int = 5):
    """
    Pre-compute k-nearest neighbour indices for every training camera.
    Proximity is Euclidean distance between camera centres in world space.
 
    Call once before the training loop and reuse every iteration.
 
    Returns
    -------
    table : list[list[int]]
        table[i] = [j1, j2, …, jk]  (indices into `cameras`, sorted by distance)
    """
    centers = []
    for cam in cameras:
        R = torch.tensor(cam.R, dtype=torch.float32)   # [3, 3]
        T = torch.tensor(cam.T, dtype=torch.float32)   # [3]
        # C2W: cam centre = (0 − T) @ Rᵀ  (row-vector form) = −R @ T
        center = (-T.unsqueeze(0) @ R.T).squeeze(0)    # [3]
        centers.append(center)
 
    centers = torch.stack(centers)   # [N, 3]
    N = len(cameras)
    k = min(k, N - 1)
 
    table = []
    for i in range(N):
        dists = torch.norm(centers - centers[i : i + 1], dim=1)
        dists[i] = float("inf")                          # exclude self
        _, idx = torch.topk(dists, k, largest=False)
        table.append(idx.tolist())
    return table

def sample_neighbor(cameras, cam_idx: int, neighbor_table):
    """
    Return a randomly sampled neighbouring camera and its index.
 
    Parameters
    ----------
    cameras        : list of Camera  (from scene.getTrainCameras())
    cam_idx        : int             index of the reference camera in `cameras`
    neighbor_table : list[list[int]] output of build_neighbor_table()
    """
    neigh_idx = choice(neighbor_table[cam_idx])
    return cameras[neigh_idx], neigh_idx


# ─────────────────────────────────────────────────────────────────────────────
# Forward warp  (reference view → neighbour view)
# ─────────────────────────────────────────────────────────────────────────────
 
def _forward_warp(depth_r, cam_r, cam_n):
    """
    Project every pixel of the reference view into the neighbour view
    using the rendered surface depth from the reference view (differentiable).
 
    Parameters
    ----------
    depth_r : Tensor [1, H, W]  rendered surf_depth of reference view
    cam_r   : Camera            reference camera
    cam_n   : Camera            neighbour camera
 
    Returns
    -------
    grid_fwd : Tensor [1, H, W, 2]  normalised (-1…1) grid for F.grid_sample
    u_n, v_n : Tensor [H, W]        pixel coords in neighbour image (unnormalised)
    Z_proj   : Tensor [H, W]        depth of projected points in neighbour cam space
    valid    : BoolTensor [H, W]    True where projection is visible & in-bounds
    H_n, W_n : int                  neighbour image dimensions
    """
    fx_r, fy_r, cx_r, cy_r, H, W   = get_camera_intrinsics(cam_r)
    fx_n, fy_n, cx_n, cy_n, H_n, W_n = get_camera_intrinsics(cam_n)
 
    # R_r = torch.tensor(cam_r.R, dtype=torch.float32, device="cuda")
    # T_r = torch.tensor(cam_r.T, dtype=torch.float32, device="cuda")
    # R_n = torch.tensor(cam_n.R, dtype=torch.float32, device="cuda")
    # T_n = torch.tensor(cam_n.T, dtype=torch.float32, device="cuda")
    R_r = cam_r.R_cuda
    T_r = cam_r.T_cuda
    R_n = cam_n.R_cuda
    T_n = cam_n.T_cuda
 
    d = depth_r[0]   # [H, W]
 
    rows, cols = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device="cuda"),
        torch.arange(W, dtype=torch.float32, device="cuda"),
        indexing="ij",
    )
 
    # 1. Unproject ref pixels to ref camera space
    X = (cols - cx_r) / fx_r * d
    Y = (rows - cy_r) / fy_r * d
    pts_cam_r = torch.stack([X, Y, d], dim=-1).reshape(-1, 3)  # [H*W, 3]
 
    # 2. Ref cam to world   X_world = (X_cam − T) @ R.T
    pts_w = (pts_cam_r - T_r) @ R_r.T   # [H*W, 3]
 
    # 3. World to neighbour cam   X_n = X_world @ R_n + T_n
    pts_n = pts_w @ R_n + T_n           # [H*W, 3]
 
    Z_n   = pts_n[:, 2].reshape(H, W)
    Z_safe = Z_n.clamp(min=1e-6)
    u_n = fx_n * pts_n[:, 0].reshape(H, W) / Z_safe + cx_n
    v_n = fy_n * pts_n[:, 1].reshape(H, W) / Z_safe + cy_n
 
    # Validity: positive depth + inside neighbour image bounds
    valid = (Z_n > 0.05) & (u_n >= 0) & (u_n < W_n) & (v_n >= 0) & (v_n < H_n)
 
    # Normalise to [−1, 1] for F.grid_sample  (x = u / col, y = v / row)
    u_norm = 2.0 * u_n / (W_n - 1) - 1.0
    v_norm = 2.0 * v_n / (H_n - 1) - 1.0
    grid_fwd = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]
 
    return grid_fwd, u_n, v_n, Z_n, valid, H_n, W_n
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main loss
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_multiview_consistency_loss(
    cam_r,
    cam_n,
    render_pkg_r,
    render_pkg_n,
    gt_image_r,           # Tensor [3, H, W]  — cam_r.original_image (on CUDA)
    gt_image_n,           # Tensor [3, H_n, W_n] — cam_n.original_image (on CUDA)
    patch_size: int = 7,
):
    """
    Compute multi-view geometric + photometric consistency losses.
 
    Geometric loss
    --------------
    Forward-backward reprojection cycle:
      1.  Warp ref pixel p_r  →  p_n  via rendered depth d_r.
      2.  Sample depth d_n at p_n  from the neighbour render.
      3.  Back-project p_n (using d_n)  →  world  →  p_r'.
      4.  Penalise  ‖p_r - p_r'‖  +  |d_r - Z_r'| / d_r.
 
    Photometric loss  (NCC)
    -----------------------
    Convert both GT images to grayscale.
    Warp the neighbour grayscale image to the reference frame (via depth).
    Compute sliding-window NCC; loss = mean( 1 - NCC ) over valid pixels.
 
    Parameters
    ----------
    cam_r / cam_n        : Camera  reference / neighbour camera
    render_pkg_r / _n    : dict    output of gaussian_renderer.render()
    gt_image_r / _n      : Tensor  ground-truth RGB images on CUDA
    patch_size           : int     NCC patch window size (should be odd)
 
    Returns
    -------
    L_geo : scalar Tensor (differentiable w.r.t. Gaussian parameters)
    L_pho : scalar Tensor (differentiable w.r.t. Gaussian parameters)
    """
    depth_r = render_pkg_r["surf_depth"]    # [1, H, W]
    depth_n = render_pkg_n["surf_depth"]    # [1, H_n, W_n]
    alpha_r = render_pkg_r["rend_alpha"]    # [1, H, W]
    alpha_n = render_pkg_n["rend_alpha"]    # [1, H_n, W_n]
 
    H = cam_r.image_height
    W = cam_r.image_width
 
    # ── 1. Forward warp  ref to neighbour ────────────────────────────────────
    grid_fwd, u_n, v_n, Z_proj, valid_fwd, H_n, W_n = _forward_warp(
        depth_r, cam_r, cam_n
    )
 
    # Sample depth & alpha from the neighbour render at warped locations
    def gs(src, grid):
        return F.grid_sample(
            src, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze()
 
    sampled_d_n = gs(depth_n, grid_fwd)   # [H, W]
    sampled_a_n = gs(alpha_n, grid_fwd)   # [H, W]
 
    # Combined validity mask  (both surfaces rendered + positive sampled depth)
    valid = (
        valid_fwd
        & (alpha_r[0] > 0.5)
        & (sampled_a_n > 0.5)
        & (sampled_d_n > 0.05)
    ).detach()
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
 
    # ── 2. Geometric consistency ─────────────────────────────────────────────
    fx_r, fy_r, cx_r, cy_r, _, _ = get_camera_intrinsics(cam_r)
    fx_n, fy_n, cx_n, cy_n, _, _ = get_camera_intrinsics(cam_n)
 
    # R_r = torch.tensor(cam_r.R, dtype=torch.float32, device="cuda")
    # T_r = torch.tensor(cam_r.T, dtype=torch.float32, device="cuda")
    # R_n = torch.tensor(cam_n.R, dtype=torch.float32, device="cuda")
    # T_n = torch.tensor(cam_n.T, dtype=torch.float32, device="cuda")
    R_r = cam_r.R_cuda
    T_r = cam_r.T_cuda
    R_n = cam_n.R_cuda
    T_n = cam_n.T_cuda
 
    # Back-project warped neighbour pixels (with sampled depth) to world to ref cam
    Xn = (u_n - cx_n) / fx_n * sampled_d_n
    Yn = (v_n - cy_n) / fy_n * sampled_d_n
    pts_n_cam  = torch.stack([Xn, Yn, sampled_d_n], dim=-1).reshape(-1, 3)
 
    pts_w_back = (pts_n_cam - T_n) @ R_n.T    # neighbour cam → world
    pts_r_back = pts_w_back @ R_r + T_r        # world → ref cam
 
    Z_back = pts_r_back[:, 2].reshape(H, W).clamp(min=1e-6)
    u_back = fx_r * pts_r_back[:, 0].reshape(H, W) / Z_back + cx_r
    v_back = fy_r * pts_r_back[:, 1].reshape(H, W) / Z_back + cy_r
 
    rows, cols = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device="cuda"),
        torch.arange(W, dtype=torch.float32, device="cuda"),
        indexing="ij",
    )
 
    # Pixel reprojection error (L2 distance in pixels)
    px_err = torch.sqrt(
        ((cols - u_back) ** 2 + (rows - v_back) ** 2).clamp(min=0) + 1e-8
    )
    px_err_normalized = px_err / math.sqrt(H**2 + W**2)
    # Relative depth error  (normalised by reference depth)
    depth_err = torch.abs(depth_r[0] - Z_back) / (depth_r[0].clamp(min=1e-6) + 1e-6)
 
    L_geo = ((px_err_normalized + depth_err) * valid_f).sum() / n_valid
 
    # ── 3. Photometric consistency (NCC) ─────────────────────────────────────
    def to_gray(rgb):
        """[3, H, W] → [1, 1, H, W]  (BT.601 luminance weights)"""
        g = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return g.unsqueeze(0).unsqueeze(0)
 
    gray_r = to_gray(gt_image_r)   # [1, 1, H, W]
    gray_n = to_gray(gt_image_n)   # [1, 1, H_n, W_n]
 
    # Warp neighbour grayscale to reference image space via the computed warp
    gray_n_w = F.grid_sample(
        gray_n, grid_fwd, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [1, 1, H, W]   (differentiable w.r.t. depth through grid_fwd)
 
    # Sliding-window NCC via convolution  (O(H·W), single pass)
    pad = patch_size // 2
    k_w = torch.ones(1, 1, patch_size, patch_size, device="cuda") / (patch_size ** 2)
 
    def local_mean_var(x):
        """x: [1, 1, H, W] → mu, var each [1, 1, H, W]"""
        x_p = F.pad(x, [pad] * 4, mode="replicate")
        mu  = F.conv2d(x_p, k_w)
        mu2 = F.conv2d(x_p ** 2, k_w)
        var = (mu2 - mu ** 2).clamp(min=1e-8)
        return mu, var
 
    mu_r, var_r = local_mean_var(gray_r)
    mu_n, var_n = local_mean_var(gray_n_w)
 
    # Cross-covariance:  E[R·N] − E[R]·E[N]
    r_p = F.pad(gray_r,   [pad] * 4, mode="replicate")
    n_p = F.pad(gray_n_w, [pad] * 4, mode="replicate")
    cov = F.conv2d(r_p * n_p, k_w) - mu_r * mu_n
 
    ncc = (cov / (torch.sqrt(var_r * var_n) + 1e-8)).clamp(-1.0, 1.0)
    ncc = ncc.squeeze()   # [H, W]
 
    L_pho = ((1.0 - ncc) * valid_f).sum() / n_valid
 
    return L_geo, L_pho