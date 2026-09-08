"""Configure Python and launch the project on macOS or Linux."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--webots',help='Webots executable or macOS .app path')
parser.add_argument('--prepare-only',action='store_true')
args=parser.parse_args()
import numpy,cv2  # Fail before launching if this interpreter lacks dependencies.
for controller in ('rover','evaluator'):
    (HERE/'controllers'/controller/'runtime.ini').write_text('[python]\nCOMMAND = '+sys.executable+'\n')
world=HERE/'worlds/drishti.wbt'
if args.prepare_only:
    print('Ready. Open:',world);sys.exit(0)
candidates=[args.webots,os.environ.get('WEBOTS_HOME'),'/Applications/Webots.app',
            str(HERE.parents[2]/'work/webots-install/Webots/Webots.app'),shutil.which('webots')]
app=next((Path(p) for p in candidates if p and Path(p).exists()),None)
if app is None:
    parser.error('Install Webots R2025a from cyberbotics.com, or pass --webots PATH.')
if sys.platform=='darwin' and app.suffix=='.app':
    subprocess.run(['open','-a',str(app),str(world)],check=True)
else:
    executable=app/'webots' if app.is_dir() else app
    subprocess.Popen([str(executable),'--mode=realtime',str(world)])
