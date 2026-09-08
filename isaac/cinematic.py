"""Presentation-only camera director. Simulator truth stays inside this module.

Never pass the camera director's subject position to a navigation component.
The movie records the actual simulation state, including stops and failures.
"""
import math
import cv2
import numpy as np
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import rot_matrix_to_quat


class CinematicRecorder:
    def __init__(self,output,subject_position,width=1280,height=720):
        self.position=subject_position
        self.width,self.height=width,height
        self.camera=Camera(prim_path="/World/CinematicCamera",name="cinematic",resolution=(width,height),frequency=25)
        self.camera.set_focal_length(28.0)
        self.camera.set_horizontal_aperture(36.0)
        self.camera.set_clipping_range(.1,250)
        self.camera.initialize()
        self.writer=cv2.VideoWriter(str(output/"cinematic.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),25,(width,height))
        if not self.writer.isOpened():raise RuntimeError("Cinematic encoder did not open")
        self.smoothed=None;self.last_stamp=-1.;self.frames=0

    def aim(self,t):
        subject=np.asarray(self.position(),float).reshape(-1)[:3]
        if not np.all(np.isfinite(subject)):return
        self.smoothed=subject if self.smoothed is None else .94*self.smoothed+.06*subject
        # Continuous establishing-to-following move without abrupt cuts or teleports.
        blend=min(1.,max(0.,(t-3)/12));blend=blend*blend*(3-2*blend)
        angle=-2.45+.10*math.sin(t*.10)
        radius=7.5*(1-blend)+4.2*blend
        height=4.8*(1-blend)+2.3*blend
        target=self.smoothed+np.array([.7,0,.08])
        eye=self.smoothed+np.array([radius*math.cos(angle),radius*math.sin(angle),height])
        forward=(target-eye);forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        R=np.column_stack((right,down,forward))
        self.camera.set_world_pose(position=eye,orientation=rot_matrix_to_quat(R),camera_axes="ros")

    def record(self,t,state="WARMUP"):
        stamp=self.camera.get_current_frame().get("rendering_time")
        if stamp is None or float(stamp)<=self.last_stamp:return
        self.last_stamp=float(stamp)
        rgb=self.camera.get_rgba()
        if rgb is None or rgb.size==0:return
        frame=cv2.cvtColor(np.asarray(rgb[:,:,:3],np.uint8),cv2.COLOR_RGB2BGR)
        # Letterboxing and restrained labels do not hide the robot or scene.
        cv2.rectangle(frame,(0,0),(self.width,42),(15,23,20),-1)
        cv2.rectangle(frame,(0,self.height-42),(self.width,self.height),(15,23,20),-1)
        cv2.putText(frame,"D R I S H T I",(28,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(220,233,194),1,cv2.LINE_AA)
        cv2.putText(frame,"ISAAC SIM 6.0.1 / ACTUAL SIMULATION",(self.width-415,28),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,196,180),1,cv2.LINE_AA)
        cv2.putText(frame,f"{state}     t = {t:.1f} s",(28,self.height-16),cv2.FONT_HERSHEY_SIMPLEX,.5,(221,229,211),1,cv2.LINE_AA)
        self.writer.write(frame);self.frames+=1

    def close(self):self.writer.release()
