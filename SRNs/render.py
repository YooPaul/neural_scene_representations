import torch
from torchvision import transforms
import os
from math import cos, sin

from models import *
from utils import find_intersection

def camera_lookat(from_pos, to_pos):
    '''
    Args
    from_pos: location of the camera
    to_pos: the point camera is looking at

    Returns
    R: rotation matrix of the camera
    t: translation vector of the camera
    '''
    forward = from_pos - to_pos
    forward = -forward # negate forward because we want the camera to face the positive z axis
    forward = forward / torch.linalg.norm(forward) # normalize

    #up = torch.tensor([0,1,0]).float()
    up = torch.tensor([0,0,1]).float() # in the shapenet cars dataset used in the paper, z axis points up in world space
    right = torch.cross(forward, up)
    right = right / torch.linalg.norm(right)

    up = torch.cross(forward, right)
    up = up / torch.linalg.norm(up)

    R = torch.zeros(3,3)
    R[:,0] = right
    R[:,1] = up
    R[:,2] = forward
    R = R.transpose(0,1) # world to camera transformation
    t = -R @ from_pos

    return R, t

def main():
    # render 360 view around the object
    fps = 30
    period = 10  # time it takes to circle around once in seconds
    image_size = 64

    to_pos = torch.tensor([0,0,0]).float() # let camera look at the origin

    srn = SRN()
    rmlstm = RMLSTM()
    pixel_generator = PixelGenerator()

    checkpoint = torch.load('Models/SRN_model.pt', map_location=torch.device('gpu'))
    rmlstm.load_state_dict(checkpoint['r_model_state_dict'])
    pixel_generator.load_state_dict(checkpoint['p_model_state_dict'])
    srn.load_state_dict(checkpoint['s_model_state_dict'])

    srn.eval()
    rmlstm.eval()
    pixel_generator.eval()

    grid_x, grid_y = torch.meshgrid(torch.linspace(0,64, image_size), torch.linspace(0,64, image_size), indexing='xy')
    uv = torch.cat(tuple(torch.dstack([grid_x, grid_y])))

    ToPILImage = transforms.ToPILImage()
    normalize = lambda  x : (x - x.min()) / (x.max() - x.min())

    save_dir = 'outputs/' # directory to save output images
    r = 1 # circle radius

    # we want t to start from 0 and end at 2pi
    full_circle = torch.linspace(0, 2* torch.pi, fps * period)
    f = image_size / 512 *  525 
    K = torch.tensor([[f, 0, image_size // 2], 
                    [0, f, image_size // 2], 
                    [0, 0, 1]])
    for idx, t in enumerate(full_circle):
        x = r * cos(t)
        y = r  * sin(t)
        from_pos = torch.tensor([x,y,0.5])
        R, T = camera_lookat(from_pos, to_pos)

        intersection_coords, ds = find_intersection(srn, rmlstm, K, R, T, uv, device=None)
        v = srn(intersection_coords) # image_size^2 x 256
        output = pixel_generator(v) # image_size^2 x 3

        image = output.transpose(0,1).reshape(3, image_size, image_size)
        image = ToPILImage((normalize(image) * 255).to(torch.uint8))
        image.save(os.path.join(save_dir, '%04d.png' % idx))

if __name__ == '__main__':
    main()
