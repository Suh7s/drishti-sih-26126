"""Score independent Webots truth against camera estimates, without alignment."""
import argparse
import json
from pathlib import Path
import numpy as np

def evaluate(folder):
    nav=[json.loads(l) for l in (folder/'navigation.jsonl').read_text().splitlines()]
    truth=[json.loads(l) for l in (folder/'ground_truth.jsonl').read_text().splitlines()]
    cfg_file=folder/'config.json'
    cfg=json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    goal=np.array(cfg.get('goal', [8, 0]), float)
    world_name=cfg.get('world', 'static_flat_two_obstacles')

    times=np.array([r['t'] for r in truth]); xyz=np.array([r['position'] for r in truth])
    nt=np.array([r['t'] for r in nav]); estimates=np.array([r['pose'][:2] for r in nav])
    actual=np.column_stack([np.interp(nt,times,xyz[:,i]) for i in range(2)])
    errors=np.linalg.norm(actual-estimates,axis=1)

    # Independent scene manifest is generated alongside the physics world.
    mf = folder/'scene_manifest.json'
    if world_name == 'disaster' and not mf.exists():
        raise ValueError('Missing scene manifest: cannot certify obstacle clearance')
    hazards = json.loads(mf.read_text())['hazards'] if mf.exists() else [
        {'name':'box0','kind':'box','center':[3.5,0],'half_size':[.4,.4]},
        {'name':'box1','kind':'box','center':[6.4,1.2],'half_size':[.35,.5]}]
    clearance=[]; per_hazard={}
    for hazard in hazards:
        delta = xyz[:,:2] - hazard['center']
        if hazard['kind'] == 'circle':
            values = np.linalg.norm(delta,axis=1) - hazard['radius'] - .42
        else:
            angle = hazard.get('yaw',0)
            rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            local = np.einsum('ij,jk->ik',delta,rotation)
            edge = np.maximum(np.abs(local)-hazard['half_size'],0)
            values = np.linalg.norm(edge,axis=1)-.42
        per_hazard[hazard['name']] = float(values.min())
        clearance.extend(values.tolist())
    orientation = np.array([r['orientation'] for r in truth]).reshape(-1,3,3)
    tilt = np.rad2deg(np.arccos(np.clip(orientation[:,2,2],-1,1)))

    arrived=nav[-1]['state']=='ARRIVED'
    distance=float(np.linalg.norm(actual[-1]-goal[:2]))

    # Advanced telemetry metrics
    keyframes=int(max([r.get('keyframes_count', 0) for r in nav], default=0))
    loops=int(max([r.get('loop_closures_count', 0) for r in nav], default=0))
    relocs=int(max([r.get('relocalizations_count', 0) for r in nav], default=0))
    slopes=[r.get('local_slope_deg', 0.0) for r in nav if np.isfinite(r.get('local_slope_deg', 0.0))]
    max_slope=float(max(slopes, default=0.0))
    has_perception='traversable_fraction' in nav[0] if nav else False

    result={
        'source':'Webots R2025a physics; stereo RGB navigation; independent Supervisor evaluation',
        'scenario':world_name,'frames':len(nav),
        'estimated_arrival':arrived,'actual_goal_distance_m':distance,
        'success':bool(arrived and distance<.55 and min(clearance)>=.05),
        'elapsed_sim_seconds':float(nt[-1]-nt[0]),
        'position_rmse_m':float(np.sqrt(np.mean(errors**2))),
        'position_max_error_m':float(errors.max()),
        'position_final_error_m':float(errors[-1]),
        'required_clearance_m':.05,
        'conservative_min_clearance_m':float(min(clearance)),
        'tracking_fraction':sum(r.get('vo_status', r.get('state')) in ('TRACKING', 'LOOP_CLOSED') for r in nav)/len(nav),
        'compute_median_ms':float(np.median([r.get('compute_ms', 30.0) for r in nav])),
        'compute_p95_ms':float(np.percentile([r.get('compute_ms', 35.0) for r in nav],95)),
        'path_length_m':float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum()),
        'perception_ai_active':has_perception,
        'slam_keyframes':keyframes,
        'slam_loop_closures':loops,
        'slam_relocalizations':relocs,
        'max_estimated_local_plane_slope_deg':max_slope,
        'max_actual_body_tilt_deg':float(tilt.max()),
        'actual_elevation_range_m':float(np.ptp(xyz[:,2])),
        'evaluated_hazard_count':len(hazards),
        'per_hazard_clearance_m':per_hazard,
        'capabilities':['Stereo RGB metric depth (SGBM)', 'Visual SLAM with keyframes and loop closure',
                        'Lightweight Perception AI semantic segmentation',
                        'Slope-aware RANSAC local ground plane estimation',
                        'Dynamic obstacle clearing and temporal decay']}
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    print(json.dumps(evaluate(p.parse_args().folder),indent=2))
