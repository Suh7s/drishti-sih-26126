"""Non-supervisor controller: only paired RGB frames enter navigation."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from controller import Robot

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from drishti.vision import Calibration, camera_mount
from drishti.pipeline import CameraNavigation
from drishti.navigation import Config

robot = Robot()
step = int(robot.getBasicTimeStep())
cams = [robot.getDevice(n) for n in ('left','right')]
for c in cams:
    c.enable(step*3)
motors = [robot.getDevice(n) for n in ('front_left','rear_left','front_right','rear_right')]
for m in motors:
    m.setPosition(float('inf')); m.setVelocity(0)
out = ROOT/'results/webots_latest'
out.mkdir(parents=True, exist_ok=True)
f = 640/(2*np.tan(1.4/2))
cal = Calibration(f,f,320,200,.16,640,400)
nav = CameraNavigation(cal, goal=(8,0), initial_base_height=.13,
    mount=camera_mount(height=.62,forward=.29,left=.08,pitch_degrees=45),
    config=Config(radius=.42,margin=.08,max_speed=.25,max_yaw_rate=.45,goal_tolerance=.35,
                  support_max_age=float('inf')))
# Ground support persists in this explicitly static, flat course. New occupied
# evidence still overrides it. This is not validation of dynamic obstacles.
log = (out/'navigation.jsonl').open('w', buffering=1)
frame = 0
try:
    while robot.step(step) != -1:
        if robot.getTime() < .96 or int(round(robot.getTime()*1000/step))%3:
            continue
        images=[]
        for c in cams:
            bgra=np.frombuffer(c.getImage(),np.uint8).reshape(400,640,4)
            images.append(cv2.cvtColor(bgra,cv2.COLOR_BGRA2RGB))
        start=time.perf_counter()
        command, row, depth=nav.process(*images,robot.getTime())
        row['compute_ms']=(time.perf_counter()-start)*1000
        log.write(json.dumps(row)+'\n')
        v,_,w=command
        wheel=[(v-w*.46/2)/.13]*2+[(v+w*.46/2)/.13]*2
        for m,value in zip(motors,wheel): m.setVelocity(float(value))
        if frame%10==0:
            print(f"DRISHTI {row['t']:.1f}s {row['state']} {row['pose'][:2]} q={row['quality']:.2f} {row['reason']}",flush=True)
            cv2.imwrite(str(out/'left.jpg'),cv2.cvtColor(images[0],cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/'right.jpg'),cv2.cvtColor(images[1],cv2.COLOR_RGB2BGR))
            np.savez_compressed(out/'diagnostics.npz',depth=depth,observed=nav.grid.observed,
                                occupied=nav.grid.occupied,pose=row['pose'])
            (out/'state.json').write_text(json.dumps(row))
        frame+=1
        if row['state']=='ARRIVED':
            print('DRISHTI estimated goal reached',flush=True)
            break
finally:
    for m in motors: m.setVelocity(0)
    log.close()
while robot.step(step)!=-1: pass
