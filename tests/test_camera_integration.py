import unittest
import numpy as np
from drishti.vision import Calibration,StereoDepth,VisualOdometry,camera_mount
from drishti.synthetic_camera import SyntheticCamera
from drishti.pipeline import CameraNavigation

class CameraIntegrationTests(unittest.TestCase):
    def test_rgb_to_metric_motion(self):
        c=Calibration(fx=260,fy=260,cx=160,cy=120,width=320,height=240)
        sensor=SyntheticCamera(c);stereo=StereoDepth(c)
        initial=np.eye(4);initial[2,3]=.5
        vo=VisualOdometry(c,initial@camera_mount(height=.15))
        for x in np.linspace(0,.18,7):
            left,right=sensor.pair(x)
            d=stereo.compute(left,right)
            T,q,status=vo.update(left,d)
        base=T@np.linalg.inv(camera_mount(height=.15))
        self.assertEqual(status,"TRACKING")
        self.assertGreater(q,.30)
        self.assertAlmostEqual(base[0,3],.18,delta=.045)
        self.assertAlmostEqual(base[1,3],0,delta=.025)

    def test_duplicate_camera_frame_stops(self):
        c = Calibration(fx=260, fy=260, cx=160, cy=120, width=320, height=240)
        sensor = SyntheticCamera(c)
        pipe = CameraNavigation(c)
        left, right = sensor.pair()
        pipe.process(left, right, 1.0)
        cmd, stats, _ = pipe.process(left, right, 1.0)
        self.assertTrue(np.all(cmd == 0))
        self.assertEqual(stats["state"], "HOLD")

    def test_end_to_end_slam_perception_pipeline(self):
        c = Calibration(fx=260, fy=260, cx=160, cy=120, width=320, height=240)
        sensor = SyntheticCamera(c)
        pipe = CameraNavigation(c, goal=(5.0, 0.0), use_slam=True, use_perception=True)

        for step, x in enumerate(np.linspace(0.0, 0.15, 4)):
            t = 1.0 + step * 0.10
            left, right = sensor.pair(x)
            cmd, row, depth = pipe.process(left, right, timestamp=t)
            self.assertIn("slam_status", row)
            self.assertIn("traversable_fraction", row)
            self.assertTrue(np.all(np.isfinite(cmd)))

        self.assertGreater(len(pipe.trajectory), 0)
        self.assertIsNotNone(pipe.latest_semantic_mask)


if __name__=="__main__":unittest.main()
