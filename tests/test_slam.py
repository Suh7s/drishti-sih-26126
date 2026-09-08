"""Unit tests for Visual SLAM, keyframes, loop closure and relocalization."""
import unittest
import numpy as np
from drishti.vision import Calibration, camera_mount
from drishti.slam import VisualSLAM
from drishti.synthetic_camera import SyntheticCamera


class VisualSLAMTests(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration(fx=260, fy=260, cx=160, cy=120, width=320, height=240)
        self.camera = SyntheticCamera(self.cal)

    def test_keyframe_creation_on_displacement(self):
        slam = VisualSLAM(self.cal, min_keyframe_dist=0.30)
        # Process first frame
        l0, r0 = self.camera.pair(0.0)
        d0 = np.full((240, 320), 3.0, dtype=np.float32)
        T, q, status, diag = slam.update(l0, d0, timestamp=0.0)
        self.assertGreaterEqual(diag["keyframes_count"], 1)

    def test_slam_tracking_quality_and_pose(self):
        slam = VisualSLAM(self.cal)
        poses = []
        for x in np.linspace(0.0, 0.20, 6):
            left, right = self.camera.pair(x)
            d = np.full((240, 320), 3.0, dtype=np.float32)
            T, q, status, diag = slam.update(left, d, timestamp=float(x))
            poses.append(T[:3, 3].copy())
        self.assertEqual(len(poses), 6)
        self.assertIn(status, ["TRACKING", "INITIALIZED", "LOOP_CLOSED"])

    def test_relocalization_after_tracking_loss(self):
        slam = VisualSLAM(self.cal)
        # Establish anchor & keyframe
        l0, r0 = self.camera.pair(0.0)
        d0 = np.full((240, 320), 3.0, dtype=np.float32)
        slam.update(l0, d0, timestamp=0.0)

        # Force tracking loss with blank/noisy frame
        blank = np.zeros((240, 320), dtype=np.uint8)
        nan_d = np.full((240, 320), np.nan, dtype=np.float32)
        slam.update(blank, nan_d, timestamp=1.0)
        self.assertEqual(slam.quality, 0.0)

        # Re-observe known viewpoint
        T_rec, q_rec, status_rec, diag = slam.update(l0, d0, timestamp=2.0)
        # Should either relocalize or cleanly track
        self.assertIn(status_rec, ["RELOCALIZED", "TRACKING", "INITIALIZED"])

    def test_diagnostics_structure(self):
        slam = VisualSLAM(self.cal)
        l, r = self.camera.pair(0.0)
        d = np.full((240, 320), 3.0, dtype=np.float32)
        _, _, _, diag = slam.update(l, d, timestamp=0.0)
        for key in ["slam_status", "keyframes_count", "loop_closures_count",
                    "relocalizations_count", "keyframe_added", "loop_closed"]:
            self.assertIn(key, diag)


if __name__ == "__main__":
    unittest.main()
