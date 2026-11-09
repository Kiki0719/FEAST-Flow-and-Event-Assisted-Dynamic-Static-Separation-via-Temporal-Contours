"""Part-level motion evidence from CoTracker point correspondences.

CoTracker provides 2D measurements, not a motion label. We test the fraction
of visible independent query points that move coherently relative to a window
anchor, while PnP supplies an independent temporal SE(3) estimate.
"""
import numpy as np
from scipy.spatial.transform import Rotation

class PartRigidityEvidence:
    def __init__(self, parts, *, min_points=12, min_displacement_px=2.0,
                 visible_threshold=0.5):
        self.parts = int(parts)
        self.min_points = int(min_points)
        self.min_displacement_px = float(min_displacement_px)
        self.visible_threshold = float(visible_threshold)
        self.windows = []

    def ingest(self, start, part_ids, query_xy, tracks_xy, visibility, validation=None):
        part_ids = np.asarray(part_ids, dtype=np.int32)
        q = np.asarray(query_xy, dtype=np.float64)
        xy = np.asarray(tracks_xy, dtype=np.float64)
        vis = np.asarray(visibility)
        if xy.shape != (len(vis), len(q), 2):
            raise ValueError('Track/visibility/query dimensions mismatch')
        if validation is not None:
            validation = np.asarray(validation)
            if validation.shape != vis.shape:
                raise ValueError('Point validation shape mismatch')
        record = {'start': int(start), 'parts': {}}
        for p in range(self.parts):
            sel = np.flatnonzero(part_ids == p)
            if len(sel) < self.min_points:
                continue
            per_frame = []
            for t in range(1, len(xy)):
                mask = (vis[t, sel] >= self.visible_threshold)
                good = np.isfinite(xy[t, sel]).all(axis=1) & mask
                if validation is not None:
                    # 1: geometrically not rejected, 5: accepted PnP inlier.
                    # 2/3: occluded/cross-Part drift, 4: RANSAC outlier.
                    good &= np.isin(validation[t, sel], (1, 5))
                if int(good.sum()) < self.min_points:
                    continue
                displacement = xy[t, sel][good] - q[sel][good]
                mag = np.linalg.norm(displacement, axis=1)
                moved_ratio = float(np.mean(mag > self.min_displacement_px))
                # Robust 2D coherent motion component; rotation can have a
                # near-zero median flow, so moved_ratio is independent.
                vector = np.median(displacement, axis=0)
                coherence = float(np.mean(np.linalg.norm(displacement-vector,axis=1) <
                                          max(3.0, np.percentile(mag,75))))
                per_frame.append({'frame': int(start+t),
                                  'visible_points': int(good.sum()),
                                  'moving_ratio': moved_ratio,
                                  'median_displacement_px': float(np.median(mag)),
                                  'coherence': coherence})
            if per_frame: record['parts'][p] = per_frame
        self.windows.append(record)

    def classify(self, poses, valid, scene_extent=1.0):
        poses = np.asarray(poses, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        T,P=valid.shape
        classifications=[]; static=[]; uncertain=[]
        for p in range(P):
            records=[a for w in self.windows for a in w['parts'].get(p,[])]
            strong=[a for a in records if a['moving_ratio'] >= .65 and
                    a['median_displacement_px'] >= self.min_displacement_px and
                    a['coherence'] >= .5]
            still=[a for a in records if a['moving_ratio'] < .2 and
                   a['median_displacement_px'] < self.min_displacement_px]
            ids=np.flatnonzero(valid[1:,p])+1
            sane=[]
            for t in ids:
                mat=poses[t,p]; rot=mat[:3,:3]
                if np.isfinite(mat).all() and abs(np.linalg.det(rot)-1)<.02:
                    sane.append(t)
            angle=0.;translation=0.
            if sane:
                angle=float(np.percentile(np.degrees(Rotation.from_matrix(poses[sane,p,:3,:3]).magnitude()),75))
                translation=float(np.percentile(np.linalg.norm(poses[sane,p,:3,3],axis=1),75))
            has_motion=len(strong)>=3 and (angle>1.0 or translation>.01*scene_extent)
            has_still=len(records)>=6 and len(still)/len(records)>=.80 and len(sane)>=max(5,int(.35*(T-1))) and angle<1.6 and translation<.015*scene_extent
            if has_motion:
                state='moving'
            elif has_still:
                state='static';static.append(p)
            else:
                state='unverified_motion_conflict'; uncertain.append(p)
            classifications.append({'part':p,'status':state,
              'tracked_frames':int(valid[:,p].sum()),
              '2d_evidence_frames':len(records),
              '2d_coherent_moving_frames':len(strong),
              '2d_static_frames':len(still),
              'rotation_p75_deg':angle,'translation_p75':translation,
              'note':'CoTracker measurements + part-wide consensus; not ground-truth motion'})
        return static, uncertain, classifications
