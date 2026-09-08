"""Regressions found during real Webots integration."""
import unittest
import numpy as np
from drishti.navigation import Config, GridMap, Navigator
from drishti.vision import Calibration, GroundMapper


class WebotsSafetyTests(unittest.TestCase):
    def test_turn_does_not_require_untraversed_ground_ahead(self):
        grid=GridMap(30,30,.25,(0,0))
        cfg=Config(radius=.1,margin=.01)
        grid.observed[grid.cell([2.125,2.125])]=True
        nav=Navigator(cfg)
        command=nav.command(grid,[2.125,2.125,0],[2.125,5],1,0)
        self.assertEqual(command[0],0)
        self.assertGreater(command[2],0)

    def test_turn_stops_when_current_footprint_is_unverified(self):
        grid=GridMap(30,30,.25,(0,0))
        nav=Navigator(Config(radius=.1,margin=.01))
        command=nav.command(grid,[2.125,2.125,0],[2.125,5],1,0)
        np.testing.assert_array_equal(command,[0,0,0])

    def test_launch_diagnostic_does_not_filter_out_wrong_height(self):
        c=Calibration(80,80,80,60,.12,160,120)
        grid=GridMap(40,40,.25,(-5,-5))
        mapper=GroundMapper(grid,c)
        # Every reconstructed point is at z=2. A plane-agreement filtered
        # diagnostic must not silently report z=0 from a selected subset.
        stats=mapper.update(np.full((120,160),2,dtype=np.float32),np.eye(4),0)
        self.assertAlmostEqual(stats['ground_z_median'],2)
        self.assertFalse(grid.observed.any())

    def test_persistent_support_does_not_override_stale_images(self):
        grid=GridMap(30,30,.25,(0,0));grid.observed[:]=True
        nav=Navigator(Config(radius=.1,support_max_age=float('inf')))
        command=nav.command(grid,[2,2,0],[5,2],1,1,now=100)
        np.testing.assert_array_equal(command,[0,0,0])
        self.assertEqual(nav.reason,'Camera data stale')
