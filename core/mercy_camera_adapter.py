"""Camera adapter for MERCY / SC-GS. pose_gl is OpenGL C2W."""
import math
import numpy as np
import torch
from types import SimpleNamespace
from FEAST.utils.graphics_utils import getProjectionMatrix

def make_camera(pose_gl, width, height, fovy):
    pose_gl=np.asarray(pose_gl,dtype=np.float64)
    if pose_gl.shape!=(4,4): raise ValueError("Expected 4x4 OpenGL C2W")
    if not torch.cuda.is_available(): raise RuntimeError("MERCY renderer requires CUDA")
    c2w_cv=pose_gl @ np.diag([1.,-1.,-1.,1.])
    w2c_cv=np.linalg.inv(c2w_cv)
    fx=float(height)/(2*math.tan(float(fovy)/2))
    fovx=2*math.atan(float(width)/(2*fx))
    device='cuda'
    view=torch.as_tensor(w2c_cv.T.copy(),dtype=torch.float32,device=device)
    projection=getProjectionMatrix(znear=.01,zfar=100.,fovX=fovx,fovY=float(fovy)).T.to(device)
    return SimpleNamespace(FoVx=fovx,FoVy=float(fovy),image_width=int(width),image_height=int(height),
       world_view_transform=view,projection_matrix=projection,full_proj_transform=view@projection,
       camera_center=torch.linalg.inv(view)[3,:3],znear=.01,zfar=100.)
