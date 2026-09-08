import unittest
import cv2
import numpy as np
from drishti.vision import Calibration,StereoDepth,VisualOdometry,GroundMapper,estimate_motion,camera_mount,camera_ros_mount
from drishti.navigation import GridMap

class VisionTests(unittest.TestCase):
    def test_stereo_known_disparity(self):
        rng=np.random.default_rng(4)
        left=rng.integers(0,256,(240,320),dtype=np.uint8)
        right=np.zeros_like(left);right[:,:-12]=left[:,12:]
        c=Calibration(fx=240,fy=240,cx=160,cy=120,width=320,height=240)
        d=StereoDepth(c).compute(left,right)
        crop=d[30:210,120:270]
        self.assertGreater(np.isfinite(crop).mean(),.75)
        self.assertAlmostEqual(float(np.nanmedian(crop)),2.4,delta=.08)

    def test_textureless_depth_stays_unknown(self):
        c=Calibration(width=320,height=240,cx=160,cy=120)
        d=StereoDepth(c).compute(np.zeros((240,320),np.uint8),np.zeros((240,320),np.uint8))
        self.assertLess(np.isfinite(d).mean(),.01)

    def test_pnp_metric_translation(self):
        rng=np.random.default_rng(6);c=Calibration()
        points=rng.uniform([-1,-.7,2],[1,.7,6],(180,3)).astype(np.float32)
        expected=np.array([-.12,.02,.01])
        uv=cv2.projectPoints(points,np.zeros(3),expected,c.K,None)[0].reshape(-1,2)
        uv+=rng.normal(0,.1,uv.shape)
        T,n,ratio=estimate_motion(points,uv,c.K)
        self.assertIsNotNone(T);self.assertGreater(n,160)
        np.testing.assert_allclose(T[:3,3],expected,atol=.006)

    def test_invalid_depth_does_not_create_free_ground(self):
        g=GridMap();c=Calibration()
        GroundMapper(g,c).update(np.full((480,640),np.nan),camera_mount(),0)
        self.assertFalse(g.observed.any())

    def test_optical_frame_points_forward_downward(self):
        T=camera_mount()
        self.assertGreater(T[0,2],.6);self.assertLess(T[2,2],-.6)

    def test_ros_render_mount_matches_optical_forward(self):
        optical=camera_mount()[:3,:3]
        ros=camera_ros_mount()[:3,:3]
        # OpenCV +Z and ROS +X are both the physical viewing ray.
        np.testing.assert_allclose(optical[:,2],ros[:,2],atol=1e-12)
        self.assertLess(ros[2,2],-.6)

    def test_blank_vo_never_claims_tracking(self):
        c=Calibration();vo=VisualOdometry(c)
        _,q,status=vo.update(np.zeros((480,640),np.uint8),np.full((480,640),np.nan))
        self.assertEqual(q,0);self.assertEqual(status,"NO_FEATURES")

if __name__=="__main__":unittest.main()
