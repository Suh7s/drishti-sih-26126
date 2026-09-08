"""Small analytic RGB renderer for camera-pipeline tests, not Isaac/Spot evidence.

Truth geometry exists only in this test sensor generator. The tested vision module
receives RGB pairs and calibration, never the renderer's distances or camera pose.
"""
import numpy as np
import cv2
from .vision import camera_mount

class SyntheticCamera:
    def __init__(self,calibration,seed=26126):
        self.c=calibration
        rng=np.random.default_rng(seed)
        self.texture=rng.integers(40,245,(1024,1024),dtype=np.uint8)
        self.texture=cv2.GaussianBlur(self.texture,(3,3),.7)
        v,u=np.indices((self.c.height,self.c.width))
        self.rays=np.stack(((u-self.c.cx)/self.c.fx,(v-self.c.cy)/self.c.fy,np.ones_like(u)),axis=-1)
        self.boxes=[(np.array([3.6,-.4,0.0]),np.array([4.3,.4,.8])),
                    (np.array([6.4,1.2,0.0]),np.array([7.0,1.8,1.1]))]

    def render(self,T):
        origin=T[:3,3]
        ray=self.rays@T[:3,:3].T
        with np.errstate(divide='ignore',invalid='ignore'):
            ground=-origin[2]/ray[:,:,2]
        dist=np.where((ray[:,:,2]<-1e-6)&(ground>0),ground,np.inf)
        material=np.zeros(dist.shape,int)
        for i,(lo,hi) in enumerate(self.boxes,1):
            with np.errstate(divide='ignore',invalid='ignore'):
                a=(lo-origin)/ray;b=(hi-origin)/ray
            near=np.max(np.minimum(a,b),axis=2);far=np.min(np.maximum(a,b),axis=2)
            hit=(far>=near)&(near>0)&(near<dist)
            dist[hit]=near[hit];material[hit]=i
        finite=np.isfinite(dist)
        points=origin+ray*np.where(finite,dist,0)[:,:,None]
        u=points[:,:,0]*95+points[:,:,2]*71
        v=points[:,:,1]*95+points[:,:,2]*113
        # Bilinear texture lookup is consistent between viewpoints.
        tex=cv2.remap(self.texture,np.mod(u,1023).astype(np.float32),np.mod(v,1023).astype(np.float32),cv2.INTER_LINEAR).astype(float)
        colors=np.stack((tex*.86,tex*.93,tex*.69),axis=-1)
        rock=material>0
        colors[rock]=np.stack((tex[rock]*.98,tex[rock]*.91,tex[rock]*.80),axis=-1)
        colors[~finite]=[147,184,198]
        return np.clip(colors,0,255).astype(np.uint8)

    def pair(self,x=0.0,y=0.0,yaw=0.0):
        base=np.eye(4);base[:3,3]=[x,y,.5]
        c,s=np.cos(yaw),np.sin(yaw)
        base[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        return tuple(self.render(base@camera_mount(height=.15,left=side)) for side in (.06,-.06))
