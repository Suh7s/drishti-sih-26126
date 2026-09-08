"""Camera-only navigation integration. RGB pair in, body velocity command out."""
import math
import numpy as np
from .navigation import Config,GridMap,Navigator
from .vision import StereoDepth,VisualOdometry,GroundMapper,camera_mount


class CameraNavigation:
    def __init__(self,calibration,goal=(10.0,0.0),initial_base_height=.50):
        self.calibration=calibration
        self.goal=np.asarray(goal,float)
        self.grid=GridMap(100,80,.25,(-3.0,-10.0))
        self.mount=camera_mount(height=.15)
        initial=np.eye(4);initial[2,3]=initial_base_height
        self.stereo=StereoDepth(calibration)
        self.vo=VisualOdometry(calibration,initial@self.mount)
        self.mapper=GroundMapper(self.grid,calibration)
        self.navigator=Navigator(Config(radius=.55,margin=.15,max_speed=.30))
        self.last_stamp=None
        self.bootstrap_done=False
        self.trajectory=[]

    def process(self,left,right,timestamp,sensor_age=0.0):
        if self.last_stamp is not None and timestamp<=self.last_stamp:
            return self.navigator.stop("HOLD","Duplicate or reversed camera timestamp"),{"state":"HOLD","reason":"Camera timestamp invalid"},None
        dt=.10 if self.last_stamp is None else timestamp-self.last_stamp
        self.last_stamp=timestamp
        depth=self.stereo.compute(left,right)
        T,q,status=self.vo.update(left,depth)
        base=T@np.linalg.inv(self.mount)
        pose=np.array([base[0,3],base[1,3],math.atan2(base[1,0],base[0,0])])
        if not self.bootstrap_done:
            # Declared setup prior: operator places robot on a verified flat launch pad.
            # This is not a scene-wide truth map. Initial metric origin is the launch pad.
            rr,cc=np.indices(self.grid.observed.shape)
            x=self.grid.origin[0]+(cc+.5)*self.grid.resolution
            y=self.grid.origin[1]+(rr+.5)*self.grid.resolution
            pad=x*x+y*y<=1.0**2
            self.grid.observed[pad]=True;self.grid.last_seen[pad]=timestamp
            self.bootstrap_done=True
        map_stats={}
        if q>=self.navigator.cfg.min_quality:
            map_stats=self.mapper.update(depth,T,timestamp)
        cmd=self.navigator.command(self.grid,pose,self.goal,q,sensor_age,dt,now=None)
        # Pose tilt is an estimate from VO, never simulator truth.
        tilt=math.acos(float(np.clip(base[2,2],-1,1)))
        if tilt>math.radians(18):
            cmd=self.navigator.stop("HOLD","Estimated attitude exceeds flat-ground limit")
        self.trajectory.append(pose.tolist())
        return cmd,{"t":timestamp,"pose":pose.tolist(),"quality":q,"vo_status":status,
                    "inliers":self.vo.last_inliers,"state":self.navigator.state,
                    "reason":self.navigator.reason,"command":cmd.tolist(),
                    "valid_depth_fraction":float(np.isfinite(depth).mean()),**map_stats},depth
