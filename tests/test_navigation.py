import unittest
import numpy as np
from drishti.navigation import Config,GridMap,Navigator,astar

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.g=GridMap(40,40,.25,(0,0));self.g.observed[:]=True;self.g.last_seen[:]=0
        self.cfg=Config(radius=.25,margin=.1)
        self.pose=[2,5,0];self.goal=[8,5]

    def test_route_around_barrier(self):
        self.g.occupied[8:30,20]=True
        path=astar(self.g,self.pose[:2],self.goal,self.cfg)
        self.assertTrue(path)
        blocked=self.g.blocked(self.cfg.radius+self.cfg.margin)
        self.assertTrue(all(not blocked[self.g.cell(p)] for p in path))

    def test_blocked_goal_has_no_route(self):
        self.g.occupied[self.g.cell(self.goal)]=True
        self.assertEqual(astar(self.g,self.pose[:2],self.goal,self.cfg),[])

    def test_no_corner_cutting(self):
        g=GridMap(7,7,1,(0,0));g.observed[:]=True
        g.occupied[2,3]=g.occupied[3,2]=True
        cfg=Config(radius=.01,margin=0)
        path=astar(g,[2.5,2.5],[3.5,3.5],cfg)
        if path:self.assertNotEqual(path[1],[3.5,3.5])

    def test_stale_camera_stops(self):
        nav=Navigator(self.cfg)
        cmd=nav.command(self.g,self.pose,self.goal,.9,1)
        self.assertEqual(nav.state,"HOLD");self.assertTrue(np.all(cmd==0))

    def test_tracking_loss_stops(self):
        nav=Navigator(self.cfg)
        self.assertTrue(np.all(nav.command(self.g,self.pose,self.goal,.1,0)==0))

    def test_unknown_support_stops(self):
        self.g.observed[19:23,8:13]=False
        nav=Navigator(self.cfg)
        self.assertTrue(np.all(nav.command(self.g,self.pose,self.goal,.9,0)==0))
        self.assertEqual(nav.state,"OBSERVE")

    def test_stale_ground_stops(self):
        nav=Navigator(self.cfg)
        self.assertTrue(np.all(nav.command(self.g,self.pose,self.goal,.9,0,now=10)==0))

    def test_nan_input_stops(self):
        nav=Navigator(self.cfg)
        self.assertTrue(np.all(nav.command(self.g,self.pose,self.goal,float('nan'),0)==0))

    def test_risk_route_changes(self):
        self.g.risk[15:26,13:27]=.9
        base=astar(self.g,self.pose[:2],self.goal,self.cfg,risk_aware=False)
        aware=astar(self.g,self.pose[:2],self.goal,self.cfg,risk_aware=True)
        score=lambda path:sum(self.g.risk[self.g.cell(p)] for p in path)
        self.assertLess(score(aware),score(base))

    def test_goal_stop(self):
        nav=Navigator(self.cfg)
        cmd=nav.command(self.g,[8,5,0],self.goal,.9,0)
        self.assertEqual(nav.state,"ARRIVED");self.assertTrue(np.all(cmd==0))

if __name__=="__main__":unittest.main()
