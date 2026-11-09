"""Produce a NEW dynamic binding: confirmed Part IDs intersected with flow-first Gaussian evidence.
Original labels and canonical G0 are never changed.
"""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    a=argparse.ArgumentParser(description=__doc__)
    for x in ('binding','flow-candidates','confirmation','output'):a.add_argument('--'+x,required=True)
    args=a.parse_args()
    with np.load(args.binding,allow_pickle=False) as z: original={k:z[k] for k in z.files}
    labels=original['labels'].astype(np.int32)
    with np.load(args.flow_candidates,allow_pickle=False) as z: candidates=z['candidate_dynamic_indices'].astype(np.int64)
    if np.any(candidates<0) or np.any(candidates>=len(labels)):raise ValueError('Candidate Gaussian indices out of bounds')
    confirmed=json.loads(Path(args.confirmation).read_text())['moving_parts']
    # Strict intersection: no automatic promotion of unseen/uncorroborated Gaussians.
    keep=np.isin(labels[candidates],confirmed)
    dynamic=np.unique(candidates[keep]).astype(np.int32)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    original['dynamic_indices']=dynamic
    original['motion_source']=np.asarray('flow_first_rgb_event_se3')
    np.savez_compressed(output,**original)
    print('Confirmed moving Parts:',confirmed,'Dynamic Gaussian indices:',len(dynamic),'Wrote',output)
    if not len(dynamic):raise SystemExit('WARNING: zero confirmed moving Gaussian; inspect verification results')
if __name__=='__main__':main()
