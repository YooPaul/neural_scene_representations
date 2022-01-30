import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, (nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # in-place operation


d_0 = 0.05
max_iter = 10
matmul = lambda A, X : X @ A.transpose(0, 1)

def to_world_coord(K, R, t, uv, d):
    '''
    Pinhole camera model
    [d*u, d*v, d].T = K(Rx + t)
    So the inverse is
    x = R.inv * (K.inv * [du, dv, d].T - t)
    '''
    uvd = torch.concat((uv*d,d), dim=1) # image_size^2 x 3
    return matmul(R.transpose(0,1), matmul(K.inverse(), uvd)  - t ) # image_size^2 x 3

def find_intersection(srn, rmlstm, K, R, t, uv, device):
    '''
    uv -> image_size^2 x 2
    K, R -> 3 x 3
    t -> 3
    '''

    N = uv.shape[0]
    hx, cx  = torch.zeros(N, 16).to(device), torch.zeros(N, 16).to(device)
    d = torch.ones(N, 1).to(device) * d_0

    cam_origin = (- R.transpose(0,1) @ t).expand(uv.shape[0], 3)
    world = to_world_coord(K, R, t, uv, d)
    ray_directions = F.normalize(world - cam_origin, dim=1)
    for i in range(max_iter):
        v = srn(world) # image_size^2 x 256
        distances, (hx, cx) = rmlstm(v, hx, cx) # out -> image_size^2 x 1
        world = cam_origin + ray_directions  * distances

    cam_coords = matmul(R, world) + t
    z = cam_coords[:,-1]
    return world, z
