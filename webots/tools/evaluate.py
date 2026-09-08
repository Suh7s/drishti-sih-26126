"""Score independent Webots truth against camera estimates, without alignment."""
import argparse
import json
from pathlib import Path
import numpy as np

def evaluate(folder):
    nav=[json.loads(l) for l in (folder/'navigation.jsonl').read_text().splitlines()]
    truth=[json.loads(l) for l in (folder/'ground_truth.jsonl').read_text().splitlines()]
    times=np.array([r['t'] for r in truth]); xyz=np.array([r['position'] for r in truth])
    nt=np.array([r['t'] for r in nav]); estimates=np.array([r['pose'][:2] for r in nav])
    actual=np.column_stack([np.interp(nt,times,xyz[:,i]) for i in range(2)])
    errors=np.linalg.norm(actual-estimates,axis=1)
    # Conservative enclosing circle includes front/rear wheel corners.
    clearance=[]
    for x,y,hx,hy in [(3.5,0,.4,.4),(6.4,1.2,.35,.5)]:
        delta=np.maximum(np.abs(xyz[:,:2]-[x,y])-[hx,hy],0)
        clearance.extend((np.linalg.norm(delta,axis=1)-.42).tolist())
    arrived=nav[-1]['state']=='ARRIVED'
    distance=float(np.linalg.norm(actual[-1]-[8,0]))
    result={
        'source':'Webots R2025a physics; stereo RGB navigation; independent Supervisor evaluation',
        'scenario':'static_flat_two_obstacles','frames':len(nav),
        'estimated_arrival':arrived,'actual_goal_distance_m':distance,
        'success':bool(arrived and distance<.5 and min(clearance)>0),
        'elapsed_sim_seconds':float(nt[-1]-nt[0]),
        'position_rmse_m':float(np.sqrt(np.mean(errors**2))),
        'position_max_error_m':float(errors.max()),
        'position_final_error_m':float(errors[-1]),
        'conservative_min_clearance_m':float(min(clearance)),
        'tracking_fraction':sum(r['vo_status']=='TRACKING' for r in nav)/len(nav),
        'compute_median_ms':float(np.median([r['compute_ms'] for r in nav])),
        'compute_p95_ms':float(np.percentile([r['compute_ms'] for r in nav],95)),
        'path_length_m':float(np.linalg.norm(np.diff(actual,axis=0),axis=1).sum()),
        'limitations':['Flat static course only','Known initial pose and launch pad',
                       'Classical stereo and VO; no learned semantics or loop closure',
                       'Ground support persists; not a dynamic obstacle benchmark',
                       'Clearance checks declared obstacle boxes, not every decorative object']}
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    print(json.dumps(evaluate(p.parse_args().folder),indent=2))
