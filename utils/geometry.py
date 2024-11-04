# Utility functions for geometric transformations and projections.
from typing import Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def transform_points(points, transform_matrix):
    """
    Apply a 4x4 transformation matrix to 3D points.

    Args:
        points: (N, 3) tensor of 3D points
        transform_matrix: (4, 4) transformation matrix

    Returns:
        (N, 3) tensor of transformed 3D points
    """
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homo_points = torch.cat([points, ones], dim=1)  # N x 4
    transformed_points = torch.matmul(homo_points, transform_matrix.T)
    return transformed_points[:, :3]

def get_corners(l: float, w: float, h: float):
    """
    Get 8 corners of a 3D bounding box centered at origin.

    Args:
        l, w, h: length, width, height of the box

    Returns:
        (3, 8) array of corner coordinates
    """
    return np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2],
    ])
    
def project_camera_points_to_image(points_cam, cam_intrinsics):
    """
    Project 3D points from camera space to 2D image space.

    Args:
        points_cam (np.ndarray): Shape (N, 3), points in camera space.
        cam_intrinsics (np.ndarray): Shape (3, 3), intrinsic matrix of the camera.

    Returns:
        tuple: (projected_points, depths)
            - projected_points (np.ndarray): Shape (N, 2), projected 2D points in image space.
            - depths (np.ndarray): Shape (N,), depth values of the projected points.
    """
    points_img = cam_intrinsics @ points_cam.T
    depths = points_img[2, :]
    projected_points = (points_img[:2, :] / (depths + 1e-6)).T
    
    return projected_points, depths

def cube_root(x):
    return torch.sign(x) * torch.abs(x) ** (1. / 3)

def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

def uniform_sample_sphere(num_samples, device, inverse=False):
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist)
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(0.02)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def chamfer_distance(
    source_pc: torch.Tensor,
    target_pc: torch.Tensor,
    chunk_size: Optional[int] = None,
    normalize_with_target: bool = False,
) -> torch.Tensor:
    """Computes chamfer distance (minimum squared distance from every point to the other point cloud) between two point clouds.

    Args:
        source_pc: Source point cloud [N,3].
        target_pc: Target point cloud [M,3].
        chunk_size: Chunk size to use for computing chamfer distance. If None, use full point clouds.
        normalize_with_target: Whether to normalize chamfer distance with target point cloud size.

    Returns:
        torch.Tensor: Chamfer distance between source and target.
    """

    min_dist_source_to_target = torch.tensor(0.0).to(source_pc.device)
    min_dist_target_to_source = torch.tensor(0.0).to(source_pc.device)

    # Add batch dimension as expected by torch.cdist
    source_pc = source_pc.view(1, -1, 3)
    target_pc = target_pc.view(1, -1, 3)

    def _chamfer_dist(source, target):
        dist = torch.cdist(source, target, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary").pow(2)
        dist = dist.view(source.shape[1], target.shape[1])
        min_dist_source_to_target, _ = torch.min(dist, dim=1)
        min_dist_target_to_source, _ = torch.min(dist, dim=0)
        return min_dist_source_to_target.sum(), min_dist_target_to_source.sum()

    if chunk_size is None:
        min_dist_source_to_target, min_dist_target_to_source = _chamfer_dist(source_pc, target_pc)
    else:
        for i in range(0, source_pc.shape[1], chunk_size):
            max_idx = min(i + chunk_size, source_pc.shape[1])
            min_dist_source_to_target_add, _ = _chamfer_dist(source_pc[:, i:max_idx], target_pc)
            min_dist_source_to_target_add = (
                min_dist_source_to_target_add / target_pc.shape[1]
                if normalize_with_target
                else min_dist_source_to_target_add
            )
            min_dist_source_to_target += min_dist_source_to_target_add
        for i in range(0, target_pc.shape[1], chunk_size):
            max_idx = min(i + chunk_size, target_pc.shape[1])
            min_dist_target_to_source_add, _ = _chamfer_dist(target_pc[:, i:max_idx], source_pc)
            min_dist_target_to_source_add = (
                min_dist_target_to_source_add / target_pc.shape[1]
                if normalize_with_target
                else min_dist_target_to_source_add
            )
            min_dist_target_to_source += min_dist_target_to_source_add
    return min_dist_source_to_target + min_dist_target_to_source
