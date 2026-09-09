"""Configure an isolated run and launch Webots on macOS or Linux."""
import argparse
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--webots',help='Webots executable or macOS .app path')
p.add_argument('--prepare-only',action='store_true')
p.add_argument('--world',choices=['disaster','drishti','benchmark'],default='disaster')
p.add_argument('--record',action='store_true')
p.add_argument('--fast',action='store_true')
p.add_argument('--exit',action='store_true',help='Close simulator after evaluation')
p.add_argument('--output',type=Path)
p.add_argument('--time-limit',type=float,default=140)
a=p.parse_args()
import numpy,cv2
for c in ('rover','evaluator'):
    (HERE/'controllers'/c/'runtime.ini').write_text('[python]\nCOMMAND = '+sys.executable+'\n')
world=HERE/'worlds'/f'{a.world}.wbt'
out=(a.output or HERE.parent/'results'/datetime.now().strftime('webots_run_%Y%m%d_%H%M%S')).resolve()
if out.exists() and any(out.iterdir()):
    p.error('Output folder is nonempty; use a new folder to preserve previous evidence.')
out.mkdir(parents=True,exist_ok=True)
# Rebuild the canonical generated scene so GUI autosaves cannot move the start.
if a.world == 'disaster':
    shutil.copy2(world, out/'world_before_reset.wbt')
    subprocess.run([sys.executable, str(HERE/'tools/build_disaster_world.py')], check=True)
manifest=world.with_suffix('.manifest.json')
if manifest.exists(): shutil.copy2(manifest,out/'scene_manifest.json')
goal=[10,0] if a.world=='disaster' else [8,0]
settings={'output':str(out),'record':a.record,'exit':a.exit,'world':a.world,'time_limit':a.time_limit,
          'camera_hz':1000/96,'goal':goal,'seed':26126,'sensor_input':'rectified RGB pair',
          'perception_ai':'40-feature MLP semantic segmentation (Traversable/Obstacle/Mud/Vegetation)',
          'slam':'Visual SLAM with keyframes and loop closure',
          'ground_support':'slope-aware RANSAC ground plane and dynamic obstacle decay',
          'python':sys.version.split()[0]}
settings['source_sha256']={str(f.relative_to(HERE.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [*sorted((HERE.parent/'src/drishti').glob('*.py')),HERE/'controllers/rover/rover.py',
              HERE/'controllers/evaluator/evaluator.py',HERE/'tools/evaluate.py',world,
              HERE.parent/'src/drishti/models/perception_weights.npz']}
settings['navigation_config']={'radius_m':.44,'margin_m':.24,'max_speed_m_s':.28,
                               'semantic_patch_size_px':40,'ground_projection_radius_m':2.0}
config=out/'config.json';config.write_text(json.dumps(settings,indent=2)+'\n')
print('Run outputs:',out,flush=True)
if a.prepare_only:
    print('Prepared. Open:',world);sys.exit(0)
candidates=[a.webots,os.environ.get('WEBOTS_HOME'),'/Applications/Webots.app',
            str(HERE.parents[2]/'work/webots-install/Webots/Webots.app'),shutil.which('webots')]
app=next((Path(x).expanduser() for x in candidates if x and Path(x).expanduser().exists()),None)
if app is None:p.error('Install Webots R2025a or pass --webots PATH.')
if app.suffix=='.app':executable=app/'Contents/MacOS/webots'
elif app.is_dir():executable=app/'webots'
else:executable=app
env=os.environ.copy();env['DRISHTI_CONFIG']=str(config)
command=[str(executable),'--batch','--stdout','--stderr','--mode='+('fast' if a.fast else 'realtime'),str(world)]
raise SystemExit(subprocess.call(command,env=env))
