"""Ground truth is recorded here, never supplied to the rover controller."""
import json
import numpy as np
import cv2
from pathlib import Path
from controller import Supervisor

robot=Supervisor()
step=int(robot.getBasicTimeStep())
rover=robot.getFromDef('ROVER')
shot=robot.getFromDef('SHOT')
out=Path(__file__).resolve().parents[3]/'results/webots_latest'
out.mkdir(parents=True,exist_ok=True)
log=(out/'ground_truth.jsonl').open('w',buffering=1)
frame=0
while robot.step(step)!=-1:
    if frame%3==0:
        p=np.array(rover.getPosition())
        eye=p+np.array([-3.0,-4.1,2.6])
        target=p+np.array([.6,0,.22])
        forward=target-eye;forward/=np.linalg.norm(forward)
        left=np.cross([0,0,1],forward);left/=np.linalg.norm(left)
        up=np.cross(forward,left)
        rv=cv2.Rodrigues(np.column_stack([forward,left,up]))[0].ravel()
        angle=float(np.linalg.norm(rv))
        shot.getField('position').setSFVec3f(eye.tolist())
        shot.getField('orientation').setSFRotation([*(rv/angle).tolist(),angle])
        log.write(json.dumps({'t':robot.getTime(),'position':rover.getPosition(),
                             'orientation':rover.getOrientation()})+'\n')
    if frame%150==0:
        robot.exportImage(str(out/'scene.jpg'),95)
    frame+=1
    if robot.getTime()>=120:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
