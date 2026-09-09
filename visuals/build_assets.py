"""Repository graphics generated from the measured submission trajectory."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/drishti-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
r=Path(__file__).resolve().parents[1];out=r/'docs/assets';out.mkdir(parents=True,exist_ok=True)
nav=[json.loads(x) for x in (r/'results/submission_run/navigation.jsonl').read_text().splitlines()]
truth=[json.loads(x) for x in (r/'results/submission_run/ground_truth.jsonl').read_text().splitlines()]
n=np.array([x['pose'][:2] for x in nav]);g=np.array([x['position'][:2] for x in truth]);nt=np.array([x['t'] for x in nav]);tt=np.array([x['t'] for x in truth])
actual=np.column_stack([np.interp(nt,tt,g[:,i]) for i in range(2)]);err=np.linalg.norm(n-actual,axis=1)*100
points=' '.join(f'{900+x*85:.1f},{260-y*85:.1f}' for x,y in n)
(out/'banner.svg').write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="440" viewBox="0 0 1600 440"><rect width="1600" height="440" fill="#08120f"/><g fill="none" stroke="#233c30"><path d="M840 80H1540M840 170H1540M840 260H1540M840 350H1540M900 55V365M1070 55V365M1240 55V365M1410 55V365"/></g><text x="62" y="74" fill="#59d7b0" font-family="Arial" font-size="17" letter-spacing="4">STEREO VISION / GPS-DENIED NAVIGATION</text><text x="55" y="188" fill="#edf4e8" font-family="Arial" font-size="100" letter-spacing="9">DRISHTI</text><text x="62" y="252" fill="#b5e685" font-family="Georgia" font-size="39">Navigate beyond GPS.</text><text x="64" y="334" fill="#9aafa0" font-family="Arial" font-size="18">SIH 26126 · Webots · Camera-driven rover</text><rect x="1163" y="226" width="68" height="68" fill="#617464"/><rect x="1414" y="115" width="60" height="85" fill="#617464"/><polyline points="{points}" fill="none" stroke="#59d7b0" stroke-width="4"/><circle cx="900" cy="260" r="9" fill="#08120f" stroke="#b5e685" stroke-width="2"/><circle cx="1580" cy="260" r="9" fill="#b5e685"/><text x="885" y="310" fill="#9aafa0" font-family="Arial" font-size="13">START</text><text x="1090" y="401" fill="#9aafa0" font-family="Arial" font-size="14">MEASURED VISUAL ODOMETRY / SUBMISSION RUN</text></svg>''')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12,4),gridspec_kw={'width_ratios':[1.5,1]})
ax[0].plot(actual[:,0],actual[:,1],color='#53645a',lw=3,label='Simulator truth')
ax[0].plot(n[:,0],n[:,1],color='#0b967b',lw=1.5,label='Visual odometry')
for x,y,w,h in [(3.5,0,.8,.8),(6.4,1.2,.7,1)]:ax[0].add_patch(Rectangle((x-w/2,y-h/2),w,h,color='#b9bcb3'))
ax[0].scatter([0,8],[0,0],s=50,color='#182f20',zorder=5);ax[0].set(xlabel='x (m)',ylabel='y (m)',title='Camera odometry vs independent simulator truth');ax[0].axis('equal');ax[0].legend(loc='lower right',frameon=False);ax[0].grid(alpha=.15)
ax[1].plot(nt-nt[0],err,color='#0b967b',lw=1.5);ax[1].fill_between(nt-nt[0],err,color='#0b967b',alpha=.12);ax[1].set(xlabel='Simulation time (s)',ylabel='Position error (cm)',title='Unaligned position error');ax[1].grid(alpha=.15)
fig.suptitle('DRISHTI / measured flat, static two-obstacle run',x=.05,ha='left',fontsize=15,fontweight='bold');fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=180);plt.close(fig)
print('Repository graphics generated')
