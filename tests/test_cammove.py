import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cammove
import nodes
import render


def translation(x, y, z):
    pose = np.eye(4)
    pose[:3, 3] = (x, y, z)
    return pose


def rot_y(degrees):
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    return pose


def camera_center(pose):
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return -(rotation.T @ translation)


def pinhole(width, fov_deg):
    fx = (width * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    cx = (width - 1) * 0.5
    intrinsic = np.eye(3)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fx
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cx
    return intrinsic


class CameraMoveTests(unittest.TestCase):
    def test_frame0_becomes_identity_and_keeps_the_relative_move(self):
        extrinsics = np.stack([translation(5, 0, 0)[:3], translation(5.2, 0, 0)[:3]])
        intrinsics = np.stack([pinhole(100, 60), pinhole(100, 64)])
        move = cammove.move_from_extrinsics(extrinsics, intrinsics, smoothing=0)
        first = move["poses"][0]
        second = move["poses"][1]
        self.assertTrue(np.allclose(first, np.eye(4), atol=1e-6))
        self.assertAlmostEqual(second[0, 3], 0.2, places=5)
        self.assertAlmostEqual(move["fov_deg"][0], 60, places=4)
        self.assertAlmostEqual(move["fov_deg"][1], 64, places=4)

    def test_strength_speed_and_mirror(self):
        move = cammove.canonical_move({"fov_deg": [50, 60], "poses": [np.eye(4), translation(1, 0, 0)]})
        locked = cammove.retarget_move(move, strength=0)
        self.assertTrue(np.allclose(locked["poses"][1], np.eye(4), atol=1e-6))
        self.assertAlmostEqual(locked["fov_deg"][1], 50, places=5)
        doubled = cammove.retarget_move(move, strength=2)
        self.assertAlmostEqual(doubled["poses"][1][0, 3], 2, places=5)
        self.assertAlmostEqual(doubled["fov_deg"][1], 70, places=5)
        mirrored = cammove.retarget_move(move, mirror=True)
        self.assertAlmostEqual(mirrored["poses"][1][0, 3], -1, places=5)
        fast = cammove.retarget_move(
            cammove.canonical_move({"fov_deg": [40, 50, 60], "poses": [np.eye(4), translation(0, 0, 1), translation(0, 0, 2)]}),
            speed=2,
        )
        self.assertAlmostEqual(fast["poses"][1][2, 3], 2, places=5)
        self.assertAlmostEqual(fast["fov_deg"][1], 60, places=5)

    def test_rotation_strength_and_mirror(self):
        move = cammove.canonical_move({"fov_deg": 40, "poses": [np.eye(4), rot_y(40)]})
        half = cammove.retarget_move(move, strength=0.5)
        recovered = cammove.mat_to_quat(half["poses"][1][:3, :3])
        self.assertAlmostEqual(math.degrees(2 * math.acos(recovered[0])), 20, places=3)
        mirrored = cammove.retarget_move(move, mirror=True)
        self.assertAlmostEqual(mirrored["poses"][1][0, 2], -math.sin(math.radians(40)), places=5)

    def test_smoothing_pulls_down_a_spike(self):
        poses = [translation(0, 0, 0), translation(0, 0, 0), translation(0, 0, 5), translation(0, 0, 0), translation(0, 0, 0)]
        smoothed = cammove.smooth_path(poses, 1)
        self.assertLess(abs(smoothed[2][2, 3]), 5)

    def test_roundtrip_and_rejected_intrinsics(self):
        move = cammove.canonical_move({"fov_deg": [30, 33], "poses": [np.eye(4), translation(0, 0.25, 0)]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "move.cammove"
            cammove.save_move(move, path)
            loaded = cammove.load_move(path)
        self.assertTrue(np.allclose(loaded["poses"][1], move["poses"][1], atol=1e-5))
        self.assertAlmostEqual(loaded["fov_deg"][1], 33, places=4)
        with self.assertRaises(ValueError):
            cammove.move_from_extrinsics(np.zeros((2, 3, 4)), np.stack([np.eye(3), np.eye(3)]))

    def test_render_follows_a_sideways_move_and_snaps_frames(self):
        depth = np.zeros((32, 32), dtype=np.float32)
        colors = np.zeros((32, 32, 3), dtype=np.float32)
        depth[16, 16] = 2
        colors[16, 16] = (1, 0, 0)
        move = cammove.canonical_move({"fov_deg": 90, "poses": [np.eye(4), translation(-0.5, 0, 0)]})
        images, holes, depths = render.render_move(depth, colors, move, frames=10, height=32, width=32, point_size=1, stride=1)
        self.assertEqual(images.shape[0], 9)
        start = np.argmax(images[0, :, :, 0].sum(axis=0))
        end = np.argmax(images[-1, :, :, 0].sum(axis=0))
        self.assertLess(end, start)
        self.assertGreater(holes[0].mean(), 0.5)
        self.assertGreater(depths[0, 16, 16, 0], 0)

    def test_nodes_reject_the_wrong_geometry_and_roundtrip_a_name(self):
        with self.assertRaises(ValueError):
            nodes.steal_from_geometry({"mode": "mono", "depth": np.ones((1, 4, 4))}, 0, 5)
        with self.assertRaises(ValueError):
            nodes.depth_from_geometry({"mode": "multiview", "depth": np.ones((2, 4, 4))})
        self.assertEqual(nodes.safe_name("My Move"), "My-Move")
        extrinsics = np.stack([np.eye(4)[:3], translation(0.2, 0, 0)[:3]])
        intrinsics = np.stack([pinhole(80, 50), pinhole(80, 50)])
        stolen = nodes.StealCameraMove().steal(
            {"mode": "multiview", "extrinsics": extrinsics, "intrinsics": intrinsics, "image": np.zeros((2, 8, 80, 3))},
            smoothing=0,
            max_poses=5,
        )[0]
        self.assertAlmostEqual(stolen["poses"][1][0, 3], 0.2, places=4)

    def test_straight_dolly_start_and_end_differ(self):
        """A one-way dolly must stay one-way through smooth, save, resample, and mirror."""
        distances = (0.0, 1.0, 2.0, 3.0, 4.0)
        extrinsics = np.stack([translation(0, 0, -distance)[:3] for distance in distances])
        intrinsics = np.stack([pinhole(100, 50) for _ in distances])
        depth = np.full((8, 8), 1.0)
        move = cammove.move_from_extrinsics(
            extrinsics, intrinsics, smoothing=0.35, max_poses=5, depth=depth
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dolly.cammove"
            cammove.save_move(move, path)
            loaded = cammove.load_move(path)
        raw = [camera_center(cammove.pad_extrinsic(row))[2] for row in extrinsics]
        saved = [camera_center(pose)[2] for pose in loaded["poses"]]
        resampled = [camera_center(pose)[2] for pose in cammove.resample(loaded["poses"], 9)]
        mirrored = cammove.retarget_move(loaded, mirror=True, smoothing=1.0)
        mirrored_z = [camera_center(pose)[2] for pose in mirrored["poses"]]
        self.assertEqual(loaded["depth_scale"], 1.0)
        self.assertLess(abs(saved[0]), 1e-6)
        self.assertGreater(saved[-1], 3.5)
        self.assertTrue(all(later >= earlier - 1e-6 for earlier, later in zip(saved, saved[1:])))
        self.assertGreater(resampled[-1], resampled[0] + 3.0)
        self.assertTrue(all(later >= earlier - 1e-6 for earlier, later in zip(resampled, resampled[1:])))
        self.assertGreater(mirrored_z[-1], 3.5)
        self.assertGreater(saved[-1] * raw[-1], 0.0)

    def test_rendered_dolly_does_not_return_and_respects_depth_units(self):
        depth = np.full((32, 32), 8.0, dtype=np.float32)
        colors = np.zeros((32, 32, 3), dtype=np.float32)
        colors[16, 22] = (1, 0, 0)
        small = cammove.canonical_move({
            "fov_deg": 60,
            "depth_scale": 1.0,
            "poses": [np.eye(4), translation(0, 0, -0.5)],
        })
        matched = cammove.canonical_move({
            "fov_deg": 60,
            "poses": [np.eye(4), translation(0, 0, -4.0)],
        })
        small_images, _, _ = render.render_move(depth, colors, small, frames=9, height=32, width=32, point_size=1, stride=1)
        matched_images, _, _ = render.render_move(depth, colors, matched, frames=9, height=32, width=32, point_size=1, stride=1)

        def shift(image):
            xs = np.nonzero(image[:, :, 0] > 0.5)[1]
            self.assertGreater(xs.size, 0)
            return float(xs.mean())

        self.assertGreater(shift(small_images[-1]), shift(small_images[0]) + 1.0)
        self.assertAlmostEqual(shift(small_images[-1]), shift(matched_images[-1]), delta=0.6)
        self.assertFalse(np.allclose(small_images[0], small_images[-1]))


if __name__ == "__main__":
    unittest.main()
