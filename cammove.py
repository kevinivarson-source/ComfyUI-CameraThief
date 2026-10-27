"""Camera-move math. Poses are OpenCV world-to-camera 4x4 matrices, and pose 0 is the identity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FORMAT = "cammove"
VERSION = 1


def _as_poses(poses) -> list[np.ndarray]:
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (K, 4, 4), got {array.shape}")
    if array.shape[0] < 2:
        raise ValueError("a camera move needs at least two poses")
    if not np.isfinite(array).all():
        raise ValueError("camera poses contain non-finite values")
    return [pose.copy() for pose in array]


def pad_extrinsic(extrinsic) -> np.ndarray:
    """Accept a 3x4 or 4x4 world-to-camera matrix."""
    matrix = np.asarray(extrinsic, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape == (3, 4):
        full = np.eye(4, dtype=np.float64)
        full[:3, :4] = matrix
        return full
    raise ValueError(f"extrinsic must be 3x4 or 4x4, got {matrix.shape}")


def invert_se3(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def mat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        mixed = q0 + amount * (q1 - q0)
        return mixed / np.linalg.norm(mixed)
    theta = math.acos(min(1.0, max(-1.0, dot)))
    sine = math.sin(theta)
    return (math.sin((1.0 - amount) * theta) * q0 + math.sin(amount * theta) * q1) / sine


def interpolate_pose(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    rotation = quat_to_mat(slerp(mat_to_quat(start[:3, :3]), mat_to_quat(end[:3, :3]), amount))
    translation = (1.0 - amount) * start[:3, 3] + amount * end[:3, 3]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def sample_path(poses: list[np.ndarray], position: float) -> np.ndarray:
    if position <= 0.0:
        return poses[0].copy()
    if position >= 1.0:
        return poses[-1].copy()
    scaled = position * (len(poses) - 1)
    index = min(int(math.floor(scaled)), len(poses) - 2)
    return interpolate_pose(poses[index], poses[index + 1], scaled - index)


def normalize_frame0(poses: list[np.ndarray]) -> list[np.ndarray]:
    """Put the first camera at the origin, looking down its original axis."""
    origin = invert_se3(poses[0])
    return [pose @ origin for pose in poses]


def horizontal_fov_deg(intrinsic: np.ndarray, width: int | None = None) -> float:
    """Field of view from pixel intrinsics. The principal point supplies the width."""
    fx = float(intrinsic[0, 0])
    cx = float(intrinsic[0, 2])
    if fx < 2.0 and abs(cx) < 1.0:
        raise ValueError(
            "these views have no camera intrinsics. Use the Depth Anything 3 Base model in multiview mode."
        )
    implied = 2.0 * cx + 1.0
    if implied >= 2.0:
        width = implied
    if width is None or width < 2.0 or fx <= 1e-6:
        raise ValueError("camera intrinsics are not usable")
    return math.degrees(2.0 * math.atan(0.5 * float(width) / fx))


def median_depth(depth, sky=None) -> float:
    """Median z-depth of the first view. Pose translation uses these same units."""
    array = np.asarray(depth, dtype=np.float64)
    sky_array = None if sky is None else np.asarray(sky, dtype=np.float64)
    if array.ndim == 4:
        array = array[0]
        if sky_array is not None and sky_array.ndim == 4:
            sky_array = sky_array[0]
    if array.ndim == 3:
        array = array[0]
        if sky_array is not None and sky_array.ndim == 3:
            sky_array = sky_array[0]
    mask = np.isfinite(array) & (array > 1e-4)
    if sky_array is not None and getattr(sky_array, "shape", None) == array.shape:
        mask &= sky_array < 0.5
    if not np.any(mask):
        raise ValueError("the depth map has no usable values")
    return float(np.median(array[mask]))


def move_from_extrinsics(
    extrinsics,
    intrinsics,
    width: int | None = None,
    smoothing: float = 0.35,
    max_poses: int = 5,
    depth=None,
    sky=None,
) -> dict:
    """Build a normalized camera move from DA3 multiview extrinsics and intrinsics."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    count = int(extrinsics.shape[0])
    if count < 2:
        raise ValueError("need at least two camera views to steal a move")
    keep = min(count, max(2, int(max_poses)))
    indices = np.linspace(0, count - 1, keep)
    indices = [int(index) for index in np.round(indices).astype(int)]
    # Keep time order. np.unique would sort, which is harmless for a rising
    # line and wrong if a caller ever passed an unsorted selection.
    ordered = []
    for index in indices:
        if index not in ordered:
            ordered.append(index)
    poses = [pad_extrinsic(extrinsics[index]) for index in ordered]
    fovs = [horizontal_fov_deg(intrinsics[index], width) for index in ordered]
    move = {
        "fov_deg": fovs,
        "poses": normalize_frame0(poses),
    }
    if depth is not None:
        move["depth_scale"] = median_depth(depth, sky)
    move["poses"] = smooth_path(move["poses"], smoothing)
    move["fov_deg"] = smooth_series(fovs, smoothing)
    return canonical_move(move)


def smooth_path(poses: list[np.ndarray], amount: float) -> list[np.ndarray]:
    amount = float(amount)
    if amount <= 0.0 or len(poses) < 3:
        return [pose.copy() for pose in poses]
    iterations = max(1, int(round(amount * 6)))
    blend = 0.5 * min(1.0, amount)
    current = [pose.copy() for pose in poses]
    for _ in range(iterations):
        smoothed = [current[0]]
        for index in range(1, len(current) - 1):
            midpoint = interpolate_pose(current[index - 1], current[index + 1], 0.5)
            smoothed.append(interpolate_pose(current[index], midpoint, blend))
        smoothed.append(current[-1])
        current = smoothed
    return normalize_frame0(current)


def scale_rotation(rotation: np.ndarray, strength: float) -> np.ndarray:
    quat = mat_to_quat(rotation)
    if quat[0] < 0.0:
        quat = -quat
    half = math.acos(min(1.0, max(-1.0, float(quat[0]))))
    if half < 1e-8 or strength == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = quat[1:] / math.sin(half)
    new_half = half * strength
    scaled = np.concatenate([[math.cos(new_half)], math.sin(new_half) * axis])
    return quat_to_mat(scaled)


def apply_strength(poses: list[np.ndarray], strength: float) -> list[np.ndarray]:
    scaled = []
    for pose in poses:
        updated = np.eye(4, dtype=np.float64)
        updated[:3, :3] = scale_rotation(pose[:3, :3], strength)
        updated[:3, 3] = pose[:3, 3] * strength
        scaled.append(updated)
    return scaled


def mirror_pose(pose: np.ndarray) -> np.ndarray:
    flip = np.diag([-1.0, 1.0, 1.0])
    mirrored = np.eye(4, dtype=np.float64)
    mirrored[:3, :3] = flip @ pose[:3, :3] @ flip
    mirrored[:3, 3] = flip @ pose[:3, 3]
    return mirrored


def apply_speed(poses: list[np.ndarray], speed: float, count: int | None = None) -> list[np.ndarray]:
    count = len(poses) if count is None else int(count)
    if count < 2:
        raise ValueError("speed resampling needs at least two poses")
    speed = max(0.0, float(speed))
    retimed = []
    for index in range(count):
        time = 0.0 if count == 1 else index / (count - 1)
        retimed.append(sample_path(poses, min(1.0, time * speed)))
    return retimed


def resample(poses: list[np.ndarray], count: int) -> list[np.ndarray]:
    count = int(count)
    if count < 2:
        raise ValueError("a rendered move needs at least two frames")
    return [sample_path(poses, index / (count - 1)) for index in range(count)]


def as_fovs(fov, count: int) -> list[float]:
    if isinstance(fov, (int, float)) and not isinstance(fov, bool):
        values = [float(fov)] * count
    else:
        values = [float(value) for value in fov]
        if len(values) == 1:
            values = values * count
        if len(values) != count:
            raise ValueError("the field of view list must have one value per pose")
    if any(not math.isfinite(value) or value <= 1.0 or value >= 179.0 for value in values):
        raise ValueError("field of view must be between 1 and 179 degrees")
    return values


def sample_series(values: list[float], position: float) -> float:
    if position <= 0.0:
        return float(values[0])
    if position >= 1.0:
        return float(values[-1])
    scaled = position * (len(values) - 1)
    index = min(int(math.floor(scaled)), len(values) - 2)
    blend = scaled - index
    return (1.0 - blend) * float(values[index]) + blend * float(values[index + 1])


def retime_series(values: list[float], speed: float, count: int | None = None) -> list[float]:
    count = len(values) if count is None else int(count)
    speed = max(0.0, float(speed))
    retimed = []
    for index in range(count):
        time = 0.0 if count == 1 else index / (count - 1)
        retimed.append(sample_series(values, min(1.0, time * speed)))
    return retimed


def smooth_series(values: list[float], amount: float) -> list[float]:
    amount = float(amount)
    current = [float(value) for value in values]
    if amount <= 0.0 or len(current) < 3:
        return current
    iterations = max(1, int(round(amount * 6)))
    blend = 0.5 * min(1.0, amount)
    for _ in range(iterations):
        smoothed = [current[0]]
        for index in range(1, len(current) - 1):
            midpoint = 0.5 * (current[index - 1] + current[index + 1])
            smoothed.append((1.0 - blend) * current[index] + blend * midpoint)
        smoothed.append(current[-1])
        current = smoothed
    return current


def snap_frames(count: int) -> int:
    """LTX wants 8n+1 frames. 1 is allowed for a still."""
    count = max(1, int(count))
    if count == 1:
        return 1
    groups = max(0, int(round((count - 1) / 8.0)))
    return 8 * groups + 1


def retarget_move(move: dict, strength: float = 1.0, speed: float = 1.0, smoothing: float = 0.0, mirror: bool = False) -> dict:
    poses = _as_poses(move["poses"])
    fovs = as_fovs(move["fov_deg"], len(poses))
    if mirror:
        poses = [mirror_pose(pose) for pose in poses]
    poses = apply_strength(poses, float(strength))
    opening = fovs[0]
    fovs = [min(170.0, max(2.0, opening + (value - opening) * float(strength))) for value in fovs]
    poses = apply_speed(poses, float(speed))
    fovs = retime_series(fovs, float(speed))
    poses = smooth_path(poses, float(smoothing))
    fovs = smooth_series(fovs, float(smoothing))
    return canonical_move({"fov_deg": fovs, "poses": poses})


def canonical_move(move: dict) -> dict:
    poses = normalize_frame0(_as_poses(move["poses"]))
    canonical = {
        "format": FORMAT,
        "version": VERSION,
        "convention": "opencv-w2c",
        "frame0": "identity",
        "fov_deg": as_fovs(move["fov_deg"], len(poses)),
        "poses": poses,
    }
    if move.get("depth_scale") is not None:
        canonical["depth_scale"] = float(move["depth_scale"])
    return canonical


def move_to_json(move: dict) -> dict:
    canonical = canonical_move(move)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "convention": "opencv-w2c",
        "frame0": "identity",
        "fov_deg": [round(value, 6) for value in canonical["fov_deg"]],
        "poses": np.round(np.stack(canonical["poses"]), 6).tolist(),
    }
    if canonical.get("depth_scale") is not None:
        payload["depth_scale"] = round(float(canonical["depth_scale"]), 6)
    return payload


def move_from_json(payload: dict) -> dict:
    if payload.get("format") != FORMAT:
        raise ValueError("this file is not a .cammove camera move")
    if int(payload.get("version", 0)) != VERSION:
        raise ValueError("this .cammove file is from a newer Camera Thief")
    move = {"fov_deg": payload["fov_deg"], "poses": payload["poses"]}
    if payload.get("depth_scale") is not None:
        move["depth_scale"] = payload["depth_scale"]
    return canonical_move(move)


def save_move(move: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(move_to_json(move), indent=2), encoding="utf-8")
    return destination


def load_move(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return move_from_json(payload)
