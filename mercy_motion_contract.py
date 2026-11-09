"""Validate pose interface for MERCY downstream exporters: delta_g0 (T,P,4,4)."""
import argparse
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--poses',required=True)
    a=p.parse_args()
    with np.load(a.binding,allow_pickle=False) as x:labels=x['labels'].astype(int);dyn=x['dynamic_indices'].astype(int)
    with np.load(a.poses,allow_pickle=False) as x:pose=x['delta_g0']
    P=int(labels.max())+1
    assert pose.ndim==4 and pose.shape[1:]==(P,4,4),(pose.shape,P)
    assert np.isfinite(pose).all();assert np.allclose(pose[:,:,3,:],[0,0,0,1],atol=1e-4)
    assert np.allclose(pose[0],np.eye(4),atol=1e-3),'First frame must be identity'
    assert np.all((0<=dyn)&(dyn<len(labels)))
    print('PASS: frames',len(pose),'parts',P,'Gaussian dynamic',len(dyn))
if __name__=='__main__':main()
