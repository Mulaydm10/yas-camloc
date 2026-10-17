"""TRAINING DIAGNOSTIC (EXP-0006): near-border lateral sensitivity (px per m) per camera with the GT pose.

    python -m camloc.tools.lateral_sensitivity
"""
import json, numpy as np
from camloc import config as C
from camloc.data import load_static
from camloc.geometry import T_map_base, project
from camloc.groundtruth import load_gt6
from camloc.measure import near_border_points
from camloc.road import Road3D
from camloc.align import rot_adjust
from camloc.tools.overlay_borders import bounds3d
cams,_=load_static(); road=Road3D(bounds3d(C.DEFAULT_MCAP)); border=np.vstack([road.L,road.R]); g=load_gt6()
adj=json.load(open('camloc/assets/camera_adjust.json')); rep=json.load(open('cache/calib_report.json'))
idx=np.searchsorted((g['t']-g['t'][0])/1e9, np.arange(100,1150,15))
out={}
for c in ('cl','cr','rl','rr','fr','fl'):
    s=[]
    for i in idx:
        x,y,yaw=g['x'][i],g['y'][i],g['yaw'][i]; T=T_map_base(x,y,g['z'][i],g['roll'][i],g['pitch'][i],yaw)
        pts=near_border_points(T,border,c)
        if len(pts)<5: continue
        A=rot_adjust(*adj[c])
        T2=T.copy(); T2[:3,3]+=0.5*T[:3,1]
        uv0,i0,_=project(cams[c],T,pts,A,80); uv1,i1,_=project(cams[c],T2,pts,A,80)
        common=np.intersect1d(i0,i1)
        if len(common)<5: continue
        a=uv0[np.searchsorted(i0,common),1]; b=uv1[np.searchsorted(i1,common),1]
        s.append(np.median(np.abs(b-a))*0.5/0.5)   # half-res px per m
    sens=float(np.median(s)); noise=rep[c]['val_after']['frame_median_rms']
    out[c]=(sens,noise,noise/sens)
    print(f'{c}: sensitivity {sens:.1f} px/m (half-res), per-frame residual rms {noise:.2f} px -> ~{noise/sens:.2f} m lateral-equivalent per camera-frame')
