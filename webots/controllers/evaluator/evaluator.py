"""Ground truth is recorded here, never supplied to the rover controller."""
import json
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from controller import Supervisor

robot=Supervisor()
step=int(robot.getBasicTimeStep())
rover=robot.getFromDef('ROVER')
shot=robot.getFromDef('SHOT')
ROOT=Path(__file__).resolve().parents[3]
settings=json.loads(Path(os.environ['DRISHTI_CONFIG']).read_text()) if os.environ.get('DRISHTI_CONFIG') else {}
out=Path(settings.get('output',ROOT/'results/webots_latest'))
out.mkdir(parents=True,exist_ok=True)
log=(out/'ground_truth.jsonl').open('w',buffering=1)
start_position=np.array(rover.getPosition())
if np.linalg.norm(start_position[:2]) > .01:
    (out/'evaluation.json').write_text(json.dumps({'success':False,'error':'Unexpected start position; regenerate world'}))
    robot.simulationQuit(1)
    raise SystemExit(1)
frame=0
recording=False
finished_at=None
while robot.step(step)!=-1:
    if frame%3==0:
        p=np.array(rover.getPosition())
        a=.12*np.sin(robot.getTime()*.045)
        eye=p+np.array([-2.6*np.cos(a)+2.0*np.sin(a),-2.6*np.sin(a)-2.0*np.cos(a),1.7])
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
    if frame==15 and settings.get('record'):
        robot.movieStartRecording(str(out/'cinematic.mp4'),1280,720,0,90,1,False)
        recording=True
    frame+=1
    if (out/'complete.json').exists() and finished_at is None:
        finished_at=robot.getTime()
    done=(finished_at is not None and robot.getTime()>finished_at+2) or robot.getTime()>=settings.get('time_limit',120)
    if done:
        if recording:
            robot.movieStopRecording()
            while not robot.movieIsReady() and not robot.movieFailed():
                if robot.step(step)==-1: break
        robot.exportImage(str(out/'scene.jpg'),95)
        log.close()
        sys.path.insert(0,str(ROOT/'webots/tools'))
        from evaluate import evaluate
        if (out/'navigation.jsonl').exists():
            result=evaluate(out)
            print('EVALUATION '+json.dumps(result),flush=True)
        if settings.get('exit'):
            robot.simulationQuit(0)
        else:
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        break
while robot.step(step)!=-1: pass
