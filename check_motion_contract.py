import argparse
import numpy as np

def check(path):
    with np.load(path,allow_pickle=False) as d:
        needed=['delta_g0','valid','reliable','unknown','static_assumed','rgb_optimized','frame_indices']
        missing=[x for x in needed if x not in d]
        if missing:raise ValueError('Missing '+repr(missing))
        x=d['delta_g0'];t,p=x.shape[:2]
        if x.shape!=(t,p,4,4):raise ValueError('Wrong delta_g0 shape')
        if not np.isfinite(x).all():raise ValueError('Non-finite matrix')
        if not np.allclose(x[...,3,:],[0,0,0,1],atol=1e-5):raise ValueError('Invalid homogeneous row')
        if not np.allclose(x[0],np.eye(4),atol=2e-4):raise ValueError('First-frame not identity')
        if not np.allclose(np.linalg.det(x[...,:3,:3]),1,atol=5e-3):raise ValueError('Not SE(3)')
        for key in ['valid','reliable','unknown','static_assumed','rgb_optimized']:
            if d[key].shape!=(t,p):raise ValueError('Wrong mask: '+key)
        if d['frame_indices'].shape!=(t,):raise ValueError('Wrong frame indices')
        return t,p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True)
    a=ap.parse_args();print('PASS, frames/parts:',check(a.input))
