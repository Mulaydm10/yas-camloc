"""TRAINING DIAGNOSTIC (EXP-0007): frame-to-frame visual rotation (LK flow + essential matrix) per camera
vs the GT yaw change, 600-720 s.

    python -m camloc.tools.vo_rotation_check
"""
import numpy as np, cv2, json, time
from camloc import config as C
from camloc.data import Replay, FrameSet, load_static
from camloc.groundtruth import load_gt6
from camloc.align import rot_adjust
from camloc.tools.make_masks import load_masks
cams,_=load_static(); g=load_gt6(); adj=json.load(open('camloc/assets/camera_adjust.json'))
masks={c:(m[::2,::2][:364,:753]).astype(np.uint8)*255 for c,m in load_masks().items()}
CAMS=('fl','fr','r','cl','cr','rl','rr')
yu=np.unwrap(g['yaw'])
def Rbc(c):
    T=cams[c].T_base_cam@ (rot_adjust(*adj[c]) if c in adj else np.eye(4)); return T[:3,:3]
def K2(c):
    K=cams[c].K.copy(); K[:2]*=0.5; return K
prev=None; res={c:[] for c in CAMS}; tt=[]; t0=None; k=0; tim=[]
for ev in Replay(state=None,start_s=600,duration_s=120):
    if not isinstance(ev,FrameSet): continue
    gray={c:cv2.imdecode(np.frombuffer(ev.images[c],np.uint8),cv2.IMREAD_REDUCED_GRAYSCALE_2) for c in CAMS}
    if prev is not None:
        dyaw_gt=np.interp(ev.stamp_ns,g['t'],yu)-np.interp(prev[0],g['t'],yu)
        row={}
        for c in CAMS:
            t1=time.perf_counter()
            p0=cv2.goodFeaturesToTrack(prev[1][c],400,0.01,8,mask=masks[c])
            if p0 is None or len(p0)<20: continue
            p1,st,_=cv2.calcOpticalFlowPyrLK(prev[1][c],gray[c],p0,None,winSize=(21,21),maxLevel=3)
            ok=st.ravel()==1
            if ok.sum()<20: continue
            K=K2(c); D=cams[c].D
            n0=cv2.undistortPoints(p0[ok],K,D); n1=cv2.undistortPoints(p1[ok],K,D)
            E,inl=cv2.findEssentialMat(n0,n1,np.eye(3),cv2.RANSAC,0.999,1.5/K[0,0])
            if E is None or E.shape!=(3,3): continue
            _,R,tv,_=cv2.recoverPose(E,n0,n1,np.eye(3),mask=inl)
            # camera motion: x1 = R x0 + t (points), so camera rotation = R^T; to base: Rb = Rbc R^T Rbc^T
            Rb=Rbc(c)@R.T@Rbc(c).T
            dyaw=np.arctan2(Rb[1,0],Rb[0,0])
            tim.append(time.perf_counter()-t1)
            row[c]=(dyaw,dyaw_gt,int(inl.sum()))
        for c,v in row.items(): res[c].append(v)
    prev=(ev.stamp_ns,gray)
for c in CAMS:
    a=np.array(res[c]); 
    if not len(a): continue
    e=np.degrees(a[:,0]-a[:,1]); 
    print(f'{c}: n {len(a)}  dyaw err per frame: median {np.median(e):+.3f} deg, MAD {np.median(np.abs(e-np.median(e))):.3f}, rms {np.sqrt(np.mean(e**2)):.3f}; sign check corr {np.corrcoef(a[:,0],a[:,1])[0,1]:+.3f}; inliers {np.median(a[:,2]):.0f}')
print('ms per camera-pair: %.1f'%(np.mean(tim)*1e3))
# compare: odometry yaw rate error per 0.1 s for reference
from camloc.tools.calibrate_odometry import load_state
from camloc.odometry import batch
t,F=load_state(); p=json.load(open('camloc/assets/odometry.json')); _,wz=batch(t,F,p)
m=(t>g['t'][0]+600e9)&(t<g['t'][0]+720e9); wzg=np.gradient(np.interp(t,g['t'],yu),t/1e9)
print('odometry wz error rms -> per 0.1 s yaw error %.3f deg'%(np.degrees(np.sqrt(np.mean((wz[m]-wzg[m])**2)))*0.1))
