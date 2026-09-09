"""Regression tests for defects found in the imported 2.0 implementation."""
import unittest
import numpy as np
import cv2
from drishti.navigation import GridMap, Config, Navigator, astar
from drishti.slam import VisualSLAM, Keyframe
from drishti.vision import Calibration, fit_ground_plane_ransac
from drishti.perception import PerceptionModel

class RevisionSafetyTests(unittest.TestCase):
    def test_expired_obstacle_becomes_unknown(self):
        g=GridMap(20,20,.25);g.observed[:]=True;g.last_seen[:]=0;g.occupied[10,10]=True
        g.decay(5,3)
        self.assertFalse(g.occupied[10,10]);self.assertFalse(g.observed[10,10])
        self.assertTrue(g.blocked(.5)[10,11])
        self.assertFalse(g.footprint_observed(g.xy((10,10)),.1,now=5,max_age=float('inf')))

    def test_steep_terrain_inflates_robot_footprint(self):
        g=GridMap(20,20,.25);g.slope[10,10]=.6
        blocked=g.blocked(.5,max_slope=.3)
        self.assertTrue(blocked[10,11]);self.assertTrue(blocked[9,10])

    def test_goal_inside_inflation_is_not_substituted(self):
        g=GridMap(40,40,.25,(0,0));g.observed[:]=True;g.occupied[20,20]=True
        self.assertEqual(astar(g,[2,2],g.xy((20,21)),Config(radius=.5,margin=.1)),[])

    def test_pose_correction_keeps_rigid_rotations_and_landmarks(self):
        cal=Calibration(260,260,160,120,width=320,height=240);slam=VisualSLAM(cal)
        for i in range(3):
            t=np.eye(4);t[0,3]=i
            slam.keyframes.append(Keyframe(i,float(i),t,None,None,[],None,np.array([[i,0,2.]])))
        slam.next_keyframe_id=3
        old=np.eye(4);old[0,3]=3
        corrected=old.copy();corrected[:3,:3]=cv2.Rodrigues(np.array([0.,0.,.1]))[0];corrected[1,3]=.1
        out=slam.correct_pose_chain(slam.keyframes[0],old,corrected)
        for k in slam.keyframes:
            r=k.T_world_camera[:3,:3]
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(r),1.)
            np.testing.assert_allclose(k.points_3d[0],k.T_world_camera[:3,3]+r@np.array([0.,0.,2.]),atol=1e-10)
        np.testing.assert_allclose(out,corrected);self.assertEqual(slam.map_revision,1)

    def test_ground_plane_follows_elevated_vehicle(self):
        rng=np.random.default_rng(55);xy=rng.uniform(-1,1,(400,2))
        p=np.c_[xy,.12*xy[:,0]+.6+rng.normal(0,.002,400)]
        fit=fit_ground_plane_ransac(p,cam_xy=[0,0],expected_ground_z=.6)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(-fit[3]/fit[2],.6,places=2)
        self.assertAlmostEqual(fit[5],np.arctan(.12),places=2)
        self.assertIsNone(fit_ground_plane_ransac(p,cam_xy=[0,0],expected_ground_z=0))

    def test_lookahead_cannot_cross_unobserved_ground(self):
        g=GridMap(40,40,.1,(0,0));g.observed[:]=True
        nav=Navigator(Config(radius=.05,margin=.01))
        nav.path=[[1.05,1.05],[1.15,1.05],[1.55,1.05]]
        g.observed[g.cell([1.35,1.05])]=False
        target=nav.select_target(g,np.array(nav.path[0]),g.blocked(.06))
        np.testing.assert_allclose(target,nav.path[1])

    def test_classifier_rejects_invalid_features(self):
        with self.assertRaises(ValueError):PerceptionModel().forward(np.full((1,40),np.nan))

if __name__=='__main__':unittest.main()
