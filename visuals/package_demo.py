"""Create a synchronised presentation from actual Webots camera recordings.

No navigation or physics is generated here. All values come from saved logs.
"""
import argparse
import bisect
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,default=ROOT/'demo')
a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
rows=[json.loads(x) for x in (a.run/'navigation.jsonl').read_text().splitlines()]
score=json.loads((a.run/'evaluation.json').read_text())
truth=[json.loads(x) for x in (a.run/'ground_truth.jsonl').read_text().splitlines()]
start,end=rows[0]['t'],rows[-1]['t'];ts=[r['t'] for r in rows]
scene=cv2.VideoCapture(str(a.run/'cinematic.mp4'));sensors=cv2.VideoCapture(str(a.run/'perception.mp4'))
if not scene.isOpened() or not sensors.isOpened():raise RuntimeError('Both simulator recordings are required')
sfps=scene.get(cv2.CAP_PROP_FPS);pfps=sensors.get(cv2.CAP_PROP_FPS)
fonts=[Path('/System/Library/Fonts/Supplemental/Arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
font_path=next((x for x in fonts if x.exists()),None)
def font(size):return ImageFont.truetype(str(font_path),size) if font_path else ImageFont.load_default()
F={n:font(n) for n in [12,14,16,19,25,32,44]}
BG='#081311';INK='#edf4ed';MUTED='#819891';TEAL='#59d7b0'
W,H=1600,900
base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base)
d.text((32,23),'D R I S H T I',font=F[32],fill=INK)
d.text((34,67),'VISION-BASED AUTONOMOUS GROUND NAVIGATION',font=F[12],fill=MUTED)
d.text((1100,30),'GPS-DENIED  /  STEREO RGB',font=F[19],fill=TEAL)
d.text((1100,62),'SIH 26126   ·   MAC / WEBOTS R2025a',font=F[12],fill=MUTED)
d.line((32,101,1568,101),fill='#264037',width=1)
d.text((32,114),'01   ACTUAL PHYSICS RECORDING',font=F[14],fill=MUTED)
d.text((1100,114),'02   LEFT CAMERA / RGB',font=F[14],fill=MUTED)
d.text((1100,442),'03   STEREO DEPTH / 0–5 m',font=F[14],fill=MUTED)
d.text((1100,746),'Dark pixels: invalid depth. No simulator depth input.',font=F[12],fill=MUTED)
d.rounded_rectangle((32,784,1568,869),radius=10,fill='#11251e')
d.text((32,881),'Recorded closed-loop run · flat static course · known start pose · classical stereo + visual odometry',font=F[12],fill=MUTED)
writer=cv2.VideoWriter(str(a.output/'mission_uncompressed.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(W,H))
if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
last_s=last_p=None;si=pi=-1
samples=np.arange(start,end+.001,.05)
for idx,t in enumerate(samples):
    # Webots movie starts at t=0.512 (step 16); stereo starts at t=0.960.
    want_s=int(round((t-.512)*sfps));want_p=min(len(rows)-1,int(round((t-start)*pfps)))
    while si<want_s:
        ok,x=scene.read()
        if not ok:break
        last_s=x;si+=1
    while pi<want_p:
        ok,x=sensors.read()
        if not ok:break
        last_p=x;pi+=1
    if last_s is None or last_p is None:raise RuntimeError('Recording truncated before first frame')
    row=rows[max(0,bisect.bisect_right(ts,t)-1)]
    im=base.copy();im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(last_s,(1040,585)),cv2.COLOR_BGR2RGB)),(32,144))
    im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(last_p[:,:640],(468,292)),cv2.COLOR_BGR2RGB)),(1100,144))
    im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(last_p[:,640:],(468,292)),cv2.COLOR_BGR2RGB)),(1100,470))
    dr=ImageDraw.Draw(im)
    dr.rectangle((32,731,1072,734),fill='#20372f')
    dr.rectangle((32,731,32+1040*(t-start)/(end-start),734),fill=TEAL)
    dr.text((32,747),'Stereo RGB → Visual SLAM → Perception AI → Slope Map → Dynamic A* → Wheels',font=F[14],fill=MUTED)
    goal_xy = np.array([10.0, 0.0] if score.get('scenario') == 'disaster' else [8.0, 0.0])
    fields=[('STATE',row['state']),('SIMULATION',f'{t-start:05.1f} s'),('COMMAND',f"{row['command'][0]:.2f} m/s"),
            ('TRACKING',f"{row.get('inliers', 0)} inliers"),('SLAM KEYFRAMES',f"{row.get('keyframes_count', 0)} kf"),
            ('GOAL DISTANCE',f"{np.linalg.norm(np.array(row['pose'][:2])-goal_xy):.2f} m")]
    for j,(label,value) in enumerate(fields):
        x=54+j*254
        dr.text((x,798),label,font=F[12],fill=MUTED)
        dr.text((x,821),value,font=F[25],fill=TEAL if j==0 else INK)
    if idx==min(240,len(samples)-1):im.save(a.output/'mission-preview.jpg',quality=94)
    writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
    if idx%300==0:print(f'Composed {idx}/{len(samples)} frames',flush=True)
# Four-second measured result card, explicitly scoped to this run.
card=Image.new('RGB',(W,H),BG);dr=ImageDraw.Draw(card)
dr.text((80,90),'D R I S H T I  /  MISSION COMPLETE',font=F[19],fill=TEAL)
dr.text((80,160),'Autonomous GPS-Denied Disaster Navigation',font=F[44],fill=INK)
dr.text((80,239),f"Measured in Webots · {score.get('scenario', 'Nepal flood disaster course')}",font=F[25],fill=MUTED)
for j,(label,value) in enumerate([('SIMULATED TIME',f"{score['elapsed_sim_seconds']:.2f} s"),('POSITION RMSE',f"{score['position_rmse_m']*100:.2f} cm"),('MIN. CLEARANCE',f"{score['conservative_min_clearance_m']*100:.1f} cm")]):
    x=80+j*510;dr.text((x,399),label,font=F[16],fill=MUTED);dr.text((x,445),value,font=F[44],fill=TEAL)
dr.text((80,650),'RGB-only navigation with Perception AI, Visual SLAM, and Slope-Aware Mapping.',font=F[25],fill=INK)
dr.text((80,705),'SIH Problem 26126 · Bharat Electronics Limited · Autonomous Ground Vehicle.',font=F[19],fill=MUTED)
card.save(a.output/'result-card.jpg',quality=94)
for _ in range(80):writer.write(cv2.cvtColor(np.array(card),cv2.COLOR_RGB2BGR))
writer.release();scene.release();sensors.release()
(a.output/'run-data.json').write_text(json.dumps({'score':score,'navigation':rows,'truth':truth},separators=(',',':')))
print('Saved',a.output/'mission_uncompressed.mp4')
