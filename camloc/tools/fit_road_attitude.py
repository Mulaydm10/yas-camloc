"""TRAINING-TIME fit of the road-attitude model in camloc/road.py (uses GT roll/pitch).

    python -m camloc.tools.fit_road_attitude   # prints fits; writes cache/attitude_fit.json
"""
import numpy as np
from scipy.spatial import cKDTree
from camloc import config as C
from camloc.groundtruth import load_gt6
from camloc.tools.overlay_borders import bounds3d
from camloc.geometry import densify3
g=load_gt6(); b=bounds3d(C.DEFAULT_MCAP)
L=densify3(b['track_left'],0.25); R=densify3(b['track_right'],0.25); tL=cKDTree(L[:,:2]); tR=cKDTree(R[:,:2])
i=np.arange(0,len(g['t']),20); i=i[(g['t'][i]-g['t'][0])/1e9>90]; i=i[(g['t'][i]-g['t'][0])/1e9<1160]
x,y,yaw=g['x'][i],g['y'][i],g['yaw'][i]
def road(px,py):
    dl,il=tL.query([px,py]); dr,ir=tR.query([px,py])
    return L[il],R[ir],(L[il,2]*dr+R[ir,2]*dl)/(dl+dr)
pr,rr=[],[]
for k in range(len(i)):
    l,r,_=road(x[k],y[k]); w=np.hypot(*(l[:2]-r[:2]))
    rr.append(np.arctan2(l[2]-r[2], w))   # left higher -> positive roll? REP103 roll about x fwd: left up = negative roll
    f=np.array([np.cos(yaw[k]),np.sin(yaw[k])])*10
    zf=road(x[k]+f[0],y[k]+f[1])[2]; zb=road(x[k]-f[0],y[k]-f[1])[2]
    pr.append(-np.arctan2(zf-zb,20))      # nose up (road rising) = negative pitch in REP103
pr,rr=np.array(pr),np.array(rr); gp,gr=g['pitch'][i],g['roll'][i]
for n,a,b_ in (('pitch',pr,gp),('roll',rr,gr)):
    c=np.polyfit(a,b_,1); r=b_-np.polyval(c,a)
    print(f'{n}: GT std {np.degrees(b_.std()):.3f} deg | map residual (fit {c[0]:+.2f}x{np.degrees(c[1]):+.2f}deg) std {np.degrees(r.std()):.3f} | corr {np.corrcoef(a,b_)[0,1]:+.2f}')
import json
from camloc.tools.calibrate_odometry import load_state
from camloc.odometry import batch
t,F=load_state(); p=json.load(open('camloc/assets/odometry.json')); v,wz=batch(t,F,p)
ay=np.interp(g['t'][i],t,v*wz)
X=np.c_[rr,ay,np.ones_like(ay)]; c=np.linalg.lstsq(X,gr,rcond=None)[0]; r=gr-X@c
print('roll = %.3f*roll_map + %.5f*ay %+.4f  -> residual std %.3f deg'%(c[0],c[1],c[2],np.degrees(r.std())))
X=np.c_[pr,np.ones_like(pr)]; c2=np.linalg.lstsq(X,gp,rcond=None)[0]
print('pitch = %.3f*pitch_map %+.4f'%tuple(c2))
json.dump({'roll_map':c[0],'roll_ay':c[1],'roll_c':c[2],'pitch_map':c2[0],'pitch_c':c2[1],'roll_resid_deg':float(np.degrees(r.std()))},open('cache/attitude_fit.json','w'))
