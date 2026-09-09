"""Non-supervisor controller: only paired RGB frames enter navigation."""
import json
import os
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
settings=json.loads(Path(os.environ['DRISHTI_CONFIG']).read_text()) if os.environ.get('DRISHTI_CONFIG') else {}
out = Path(settings.get('output',ROOT/'results/webots_latest'))
out.mkdir(parents=True, exist_ok=True)
f = 640/(2*np.tan(1.4/2))
cal = Calibration(f,f,320,200,.16,640,400)
goal = tuple(settings.get('goal', [10.0, 0.0]))
nav = CameraNavigation(cal, goal=goal, initial_base_height=.13,
    mount=camera_mount(height=.62,forward=.29,left=.08,pitch_degrees=45),
    config=Config(radius=.44,margin=.24,max_speed=.28,max_yaw_rate=.55,goal_tolerance=.45,
                  support_max_age=float('inf'),
                  risk_weight=12.0, slope_weight=3.5, max_slope_rad=0.52))
log = (out/'navigation.jsonl').open('w', buffering=1)
video = cv2.VideoWriter(str(out/'perception.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1000/(step*3),(1280,400)) if settings.get('record') else None
frame = 0
try:
    while robot.step(step) != -1:
        if robot.getTime() >= settings.get('time_limit',140):
            break
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
        if video is not None:
            rgb=cv2.cvtColor(images[0],cv2.COLOR_RGB2BGR)
            if getattr(nav, 'latest_semantic_mask', None) is not None:
                sem_overlay = cv2.addWeighted(rgb, 0.68, nav.latest_semantic_mask, 0.32, 0)
            else:
                sem_overlay = rgb
            valid=np.isfinite(depth)
            d8=np.uint8(np.clip(np.nan_to_num(depth,nan=0)/6,0,1)*255)
            heat=cv2.applyColorMap(d8,cv2.COLORMAP_TURBO);heat[~valid]=[17,23,27]
            video.write(np.hstack([sem_overlay,heat]))
        if frame%10==0:
            print(f"DRISHTI {row['t']:.1f}s {row['state']} {row['pose'][:2]} q={row['quality']:.2f} {row['reason']}",flush=True)
            cv2.imwrite(str(out/'left.jpg'),cv2.cvtColor(images[0],cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/'right.jpg'),cv2.cvtColor(images[1],cv2.COLOR_RGB2BGR))
            np.savez_compressed(out/'diagnostics.npz',depth=depth,observed=nav.grid.observed,
                                occupied=nav.grid.occupied,unresolved=nav.grid.unresolved_hazards,pose=row['pose'])
            (out/'state.json').write_text(json.dumps(row))
        frame+=1
        if row['state']=='ARRIVED':
            (out/'complete.json').write_text(json.dumps(row))
            print('DRISHTI estimated goal reached',flush=True)
            break
finally:
    for m in motors: m.setVelocity(0)
    log.close()
    if video is not None: video.release()
while robot.step(step)!=-1: pass
