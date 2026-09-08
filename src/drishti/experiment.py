"""Deterministic 2-D planning harness. NO camera, SLAM or Spot physics claims.

Sensor observations use an idealized 360-degree range/visibility oracle with
occlusion. Pose is exact. Risk and image quality are scripted experimental inputs.
The same navigator is used by the camera pipeline, but these results only test
planning/supervision logic. A source disclosure is embedded into every output.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from .navigation import Config,GridMap,Navigator

DISCLOSURE = "2-D synthetic-observation planning harness. Exact pose. No camera inference, SLAM, Spot physics or real-world validation."
SCENARIOS = ("rocky_path","uncertain_shortcut","camera_dropout","ditch")


def scenario(name,seed=7):
    rng=np.random.default_rng(seed)
    g=GridMap(76,48,0.25,(0,0))
    g.observed[:]=True
    g.last_seen[:]=0.0
    # Physical objects vary by seed; endpoint areas remain clear.
    centres=[(6.0,5.7),(9.8,8.0),(12.6,4.2),(14.0,8.0)]
    yy,xx=np.indices(g.occupied.shape)
    for x,y in centres:
        x,y=np.array([x,y])+rng.uniform(-0.35,0.35,2)
        g.occupied |= ((xx+0.5)*.25-x)**2+((yy+0.5)*.25-y)**2<.6**2
    if name=="uncertain_shortcut":
        g.occupied[:]=False
        g.risk[15:34,23:48]=0.90
    if name=="ditch":
        g.occupied[18:32,30:35]=True
        g.risk[16:34,28:37]=0.8
    if name not in SCENARIOS:
        raise ValueError(name)
    return g,np.array([1.8,6.0,0.0]),np.array([17.0,6.0])


def observe(truth,estimated,pose,t,reach=5.2):
    # Ray casting is an abstract sensor here, never supplied to the camera pipeline.
    for a in np.linspace(-math.pi,math.pi,241,endpoint=False):
        for d in np.arange(0,reach,.18):
            rc=truth.cell(pose[:2]+d*np.array([math.cos(a),math.sin(a)]))
            if not truth.inside(rc):
                break
            estimated.observed[rc]=True
            estimated.occupied[rc]=truth.occupied[rc]
            estimated.risk[rc]=truth.risk[rc]
            estimated.last_seen[rc]=t
            if truth.occupied[rc]:
                break


def run_episode(name,seed=7,risk_aware=True,record=False,max_steps=900):
    truth,pose,goal=scenario(name,seed)
    g=GridMap(truth.width,truth.height,truth.resolution,truth.origin)
    nav=Navigator(Config(radius=.40,margin=.15),risk_aware=risk_aware)
    dt=.15
    distance,risk_exposure,holds=0.0,0.0,0
    frames=[]
    result="TIMEOUT"
    sensor_stamp=0.0
    for step in range(max_steps):
        t=step*dt
        dropout=name=="camera_dropout" and 9<=t<12
        if not dropout:
            observe(truth,g,pose,t)
            sensor_stamp=t
        q=.50 if truth.risk[truth.cell(pose[:2])]>.5 else .95
        cmd=nav.command(g,pose,goal,q,t-sensor_stamp,dt,now=t)
        old=pose.copy()
        # Semi-implicit unicycle approximation, no legged dynamics.
        pose[2]+=cmd[2]*dt
        pose[:2]+=cmd[0]*dt*np.array([np.cos(pose[2]),np.sin(pose[2])])
        distance+=np.linalg.norm(pose[:2]-old[:2])
        risk_exposure+=truth.risk[truth.cell(pose[:2])]*dt
        holds+=int(nav.state in ("HOLD","OBSERVE","BLOCKED"))
        if truth.blocked(nav.cfg.radius)[truth.cell(pose[:2])]:
            result="COLLISION"
        if nav.state=="ARRIVED":
            result="ARRIVED"
        if record and step%2==0:
            frames.append({"t":round(t,2),"pose":np.round(pose,3).tolist(),
                "path":nav.path,"state":nav.state,"reason":nav.reason,
                "speed":round(float(cmd[0]),3),"quality":q,"age":round(t-sensor_stamp,2),
                "observed":np.flatnonzero(g.observed).tolist(),
                "distance":round(float(distance),2)})
        if result!="TIMEOUT":
            break
    metrics={"scenario":name,"seed":seed,"mode":"risk_aware" if risk_aware else "baseline",
             "result":result,"elapsed_s":round(t,2),"distance_m":round(float(distance),3),
             "scripted_risk_exposure_s":round(float(risk_exposure),3),"hold_steps":holds}
    episode={"disclosure":DISCLOSURE,"metrics":metrics,"goal":goal.tolist(),
             "width":truth.width,"height":truth.height,"resolution":truth.resolution,
             "occupied":np.flatnonzero(truth.occupied).tolist(),
             "risk":[[int(i),round(float(truth.risk.ravel()[i]),2)] for i in np.flatnonzero(truth.risk)],
             "frames":frames}
    return episode


def main():
    parser=argparse.ArgumentParser(description=DISCLOSURE)
    parser.add_argument("--seeds",type=int,default=3)
    parser.add_argument("--output",default="results")
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    episodes,rows=[],[]
    for name in SCENARIOS:
        for seed in range(7,7+args.seeds):
            for aware in (False,True):
                e=run_episode(name,seed,aware,record=seed==7)
                rows.append(e["metrics"])
                if seed==7:episodes.append(e)
                print(json.dumps(e["metrics"]),flush=True)
    (out/"replays.json").write_text(json.dumps(episodes,separators=(",",":")))
    (out/"provenance.json").write_text(json.dumps({"disclosure":DISCLOSURE,"episodes":len(rows),
        "seeds":list(range(7,7+args.seeds)),"implemented_controller":"A* and heading controller",
        "not_implemented_in_harness":["camera inference","visual SLAM","legged dynamics","learned uncertainty"]},indent=2))
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


if __name__=="__main__":main()
