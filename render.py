"""Render a coloured point cloud of a still along a camera move."""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.realpath(__file__))
if "camerathief_cammove" in sys.modules:
    _cammove = sys.modules["camerathief_cammove"]
else:
    _spec = importlib.util.spec_from_file_location(
        "camerathief_cammove",
        os.path.join(_DIR, "cammove.py"),
    )
    _cammove = importlib.util.module_from_spec(_spec)
    sys.modules["camerathief_cammove"] = _cammove
    _spec.loader.exec_module(_cammove)

as_fovs = _cammove.as_fovs
median_depth = _cammove.median_depth
resample = _cammove.resample
retime_series = _cammove.retime_series
snap_frames = _cammove.snap_frames


def snap_dim(value: int) -> int:
    value = max(32, int(value))
    return value - (value % 32)


def resize_map(array: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] == height and array.shape[1] == width:
        return array
    from PIL import Image

    if array.ndim == 2:
        image = Image.fromarray(array, mode="F")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)
    channels = np.clip(array[..., :3], 0.0, 1.0)
    image = Image.fromarray((channels * 255.0).astype(np.uint8))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def unproject(depth: np.ndarray, fov_deg: float, stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Z-depth image to camera-space points and the pixel indices they came from."""
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    stride = max(1, int(stride))
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[ys, xs]
    fx = (width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pixels = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=1)
    return points, pixels


def valid_points(depth: np.ndarray, sky: np.ndarray | None, stride: int) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 1e-4)
    if sky is not None:
        mask &= np.asarray(sky) < 0.5
    height, width = depth.shape
    stride = max(1, int(stride))
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    return mask[ys, xs].reshape(-1)


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return points @ rotation.T + translation


def project(points: np.ndarray, fov_deg: float, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points[:, 2]
    fx = (width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    u = points[:, 0] * fx / np.maximum(z, 1e-6) + cx
    v = points[:, 1] * fy / np.maximum(z, 1e-6) + cy
    return u, v, z


def splat(points: np.ndarray, colors: np.ndarray, pose: np.ndarray, fov_deg: float, height: int, width: int, point_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB, hole mask (1 = empty), and a min-max depth image."""
    camera = transform_points(points, pose)
    u, v, z = project(camera, fov_deg, height, width)
    keep = z > 1e-4
    u, v, z = u[keep], v[keep], z[keep]
    colors = colors[keep]
    radius = max(0, (max(1, int(point_size)) - 1) // 2)
    offsets = [(du, dv) for dv in range(-radius, radius + 1) for du in range(-radius, radius + 1)]
    image = np.zeros((height, width, 3), dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if u.size == 0:
        holes = np.ones((height, width), dtype=np.float32)
        return image, holes, np.zeros((height, width), dtype=np.float32)

    pieces_u = []
    pieces_v = []
    pieces_z = []
    pieces_c = []
    for du, dv in offsets:
        pieces_u.append(u + du)
        pieces_v.append(v + dv)
        pieces_z.append(z)
        pieces_c.append(colors)
    u = np.concatenate(pieces_u)
    v = np.concatenate(pieces_v)
    z = np.concatenate(pieces_z)
    colors = np.concatenate(pieces_c)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z, colors = ui[inside], vi[inside], z[inside], colors[inside]
    if ui.size == 0:
        holes = np.ones((height, width), dtype=np.float32)
        return image, holes, np.zeros((height, width), dtype=np.float32)

    pixels = vi.astype(np.int64) * width + ui.astype(np.int64)
    order = np.lexsort((z, pixels))
    pixels = pixels[order]
    _, first = np.unique(pixels, return_index=True)
    winners = order[first]
    image[vi[winners], ui[winners]] = colors[winners]
    depth[vi[winners], ui[winners]] = z[winners]
    holes = (~np.isfinite(depth)).astype(np.float32)
    seen = np.isfinite(depth)
    depth_image = np.zeros((height, width), dtype=np.float32)
    if np.any(seen):
        near = float(depth[seen].min())
        far = float(depth[seen].max())
        span = max(far - near, 1e-6)
        # Near is white, matching the depth control used in the feasibility test.
        depth_image[seen] = 1.0 - (depth[seen] - near) / span
    return image, holes, depth_image


def depth_units(move: dict, depth: np.ndarray) -> float:
    """Scale pose translation from the clip's depth units into the still's."""
    reference = move.get("depth_scale")
    if reference is None:
        return 1.0
    reference = float(reference)
    if not math.isfinite(reference) or reference <= 1e-6:
        return 1.0
    return median_depth(depth) / reference


def scale_translations(poses: list[np.ndarray], scale: float) -> list[np.ndarray]:
    if abs(scale - 1.0) <= 1e-8:
        return poses
    scaled = []
    for pose in poses:
        updated = np.array(pose, dtype=np.float64, copy=True)
        updated[:3, 3] *= scale
        scaled.append(updated)
    return scaled


def render_move(depth: np.ndarray, colors: np.ndarray, move: dict, frames: int, height: int, width: int, point_size: int = 4, stride: int = 2, sky: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = snap_frames(frames)
    height = snap_dim(height)
    width = snap_dim(width)
    depth = np.asarray(depth, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.shape[:2] != depth.shape:
        raise ValueError("the still and its depth map are different sizes")
    source_poses = list(np.asarray(move["poses"], dtype=np.float64))
    fovs = retime_series(as_fovs(move["fov_deg"], len(source_poses)), 1.0, frames)
    points, pixels = unproject(depth, fovs[0], stride)
    keep = valid_points(depth, sky, stride)
    points = points[keep]
    chosen = colors[pixels[keep, 0], pixels[keep, 1]]
    if chosen.ndim == 1:
        chosen = chosen.reshape(1, -1)
    if chosen.shape[-1] == 4:
        chosen = chosen[..., :3]
    poses = scale_translations(resample(source_poses, frames), depth_units(move, depth))
    images = []
    holes = []
    depths = []
    for pose, fov in zip(poses, fovs):
        image, hole, depth_image = splat(points, chosen, pose, float(fov), height, width, point_size)
        images.append(image)
        holes.append(hole)
        depths.append(np.repeat(depth_image[..., None], 3, axis=-1))
    return np.stack(images), np.stack(holes), np.stack(depths)
