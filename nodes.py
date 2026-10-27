"""ComfyUI nodes for stealing a camera move and applying it to a still."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

_DIR = os.path.dirname(os.path.realpath(__file__))


def _load(filename, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cammove = _load("cammove.py", "camerathief_cammove")
render = _load("render.py", "camerathief_render")

CATEGORY = "Camera Thief"


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _torch(array):
    array = np.ascontiguousarray(array)
    try:
        import torch
    except ImportError:
        return array
    return torch.from_numpy(array)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "").strip()).strip("-")[:80]
    if not cleaned:
        raise ValueError("name the camera move with letters, numbers, dashes, or underscores")
    return cleaned


def cammove_dir() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory())
    except Exception:
        root = Path.cwd() / "output"
    destination = root / "cammoves"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def steal_from_geometry(geometry: dict, smoothing: float, max_poses: int) -> dict:
    if geometry.get("mode") != "multiview" or "extrinsics" not in geometry or "intrinsics" not in geometry:
        raise ValueError(
            "Steal Camera Move needs a multi-view Depth Anything 3 result. "
            "Use the Base model and set the mode to multiview. The Metric model has no camera poses."
        )
    image = geometry.get("image")
    width = int(_numpy(image).shape[2]) if image is not None else None
    try:
        depth = geometry.get("depth")
        sky = geometry.get("sky")
        return cammove.move_from_extrinsics(
            _numpy(geometry["extrinsics"]),
            _numpy(geometry["intrinsics"]),
            width=width,
            smoothing=float(smoothing),
            max_poses=int(max_poses),
            depth=None if depth is None else _numpy(depth),
            sky=None if sky is None else _numpy(sky),
        )
    except ValueError as error:
        message = str(error)
        if "no camera intrinsics" in message:
            raise ValueError(
                "Steal Camera Move could not read a camera from this clip. "
                "Use the Depth Anything 3 Base model, set the mode to multiview, and use at least two frames."
            ) from error
        raise


def depth_from_geometry(geometry: dict) -> tuple[np.ndarray, np.ndarray | None]:
    if geometry.get("mode") == "multiview":
        raise ValueError(
            "Apply Move to Image needs the still's depth. "
            "Run Depth Anything 3 in mono mode on the still, with the Metric model."
        )
    if "depth" not in geometry:
        raise ValueError("this Depth Anything 3 result has no depth map")
    depth = _numpy(geometry["depth"]).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    sky = None
    if "sky" in geometry:
        sky = _numpy(geometry["sky"]).astype(np.float32)
        if sky.ndim == 3:
            sky = sky[0]
    return depth, sky


def still_frame(image) -> np.ndarray:
    frame = _numpy(image).astype(np.float32)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError("the still must be an image")
    frame = frame[..., :3]
    if float(np.nanmax(frame)) > 1.5:
        frame = frame / 255.0
    return np.clip(frame, 0.0, 1.0)


class StealCameraMove:
    CATEGORY = CATEGORY
    FUNCTION = "steal"
    RETURN_TYPES = ("CAMERA_MOVE", "FLOAT", "INT")
    RETURN_NAMES = ("camera_move", "fov_degrees", "poses")
    DESCRIPTION = "Read the camera path from a multi-view Depth Anything 3 Base result."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "da3_geometry": ("DA3_GEOMETRY",),
                "smoothing": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_poses": ("INT", {"default": 5, "min": 2, "max": 24}),
            }
        }

    def steal(self, da3_geometry, smoothing, max_poses):
        move = steal_from_geometry(da3_geometry, smoothing, max_poses)
        return (move, float(move["fov_deg"][0]), len(move["poses"]))


class SaveCameraMove:
    CATEGORY = CATEGORY
    FUNCTION = "save"
    RETURN_TYPES = ("CAMERA_MOVE", "STRING")
    RETURN_NAMES = ("camera_move", "path")
    OUTPUT_NODE = True
    DESCRIPTION = "Write the camera move to a .cammove file in the ComfyUI output folder."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_move": ("CAMERA_MOVE",),
                "name": ("STRING", {"default": "move"}),
            }
        }

    def save(self, camera_move, name):
        destination = cammove_dir() / f"{safe_name(name)}.cammove"
        cammove.save_move(camera_move, destination)
        return (camera_move, str(destination))


class LoadCameraMove:
    CATEGORY = CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = ("CAMERA_MOVE",)
    RETURN_NAMES = ("camera_move",)
    DESCRIPTION = "Load a .cammove file saved by Save Camera Move."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "move"}),
            }
        }

    def load(self, name):
        destination = cammove_dir() / f"{safe_name(name)}.cammove"
        if not destination.is_file():
            raise ValueError(f"no saved camera move named {safe_name(name)}")
        return (cammove.load_move(destination),)


class RetargetCameraMove:
    CATEGORY = CATEGORY
    FUNCTION = "retarget"
    RETURN_TYPES = ("CAMERA_MOVE",)
    RETURN_NAMES = ("camera_move",)
    DESCRIPTION = "Scale, speed up, smooth, or mirror a stolen camera move."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_move": ("CAMERA_MOVE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mirror": ("BOOLEAN", {"default": False}),
            }
        }

    def retarget(self, camera_move, strength, speed, smoothing, mirror):
        return (cammove.retarget_move(camera_move, strength, speed, smoothing, bool(mirror)),)


class ApplyMoveToImage:
    CATEGORY = CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT")
    RETURN_NAMES = ("guide", "depth", "hole_mask", "frames")
    DESCRIPTION = "Project the still along the camera move. White in the depth image is near; the mask marks holes."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "still": ("IMAGE",),
                "da3_geometry": ("DA3_GEOMETRY",),
                "camera_move": ("CAMERA_MOVE",),
                "frames": ("INT", {"default": 9, "min": 1, "max": 121}),
                "width": ("INT", {"default": 512, "min": 64, "max": 1280, "step": 32}),
                "height": ("INT", {"default": 320, "min": 64, "max": 1280, "step": 32}),
                "point_size": ("INT", {"default": 5, "min": 1, "max": 15}),
                "stride": ("INT", {"default": 2, "min": 1, "max": 8}),
            }
        }

    def apply(self, still, da3_geometry, camera_move, frames, width, height, point_size, stride):
        colors = still_frame(still)
        depth, sky = depth_from_geometry(da3_geometry)
        depth = render.resize_map(depth, colors.shape[0], colors.shape[1])
        if sky is not None:
            sky = render.resize_map(sky, colors.shape[0], colors.shape[1])
        images, holes, depths = render.render_move(
            depth,
            colors,
            camera_move,
            frames=int(frames),
            height=int(height),
            width=int(width),
            point_size=int(point_size),
            stride=int(stride),
            sky=sky,
        )
        return (_torch(images), _torch(depths), _torch(holes), int(images.shape[0]))


NODE_CLASS_MAPPINGS = {
    "StealCameraMove": StealCameraMove,
    "SaveCameraMove": SaveCameraMove,
    "LoadCameraMove": LoadCameraMove,
    "RetargetCameraMove": RetargetCameraMove,
    "ApplyMoveToImage": ApplyMoveToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StealCameraMove": "Steal Camera Move",
    "SaveCameraMove": "Save Camera Move",
    "LoadCameraMove": "Load Camera Move",
    "RetargetCameraMove": "Retarget Camera Move",
    "ApplyMoveToImage": "Apply Move to Image",
}
