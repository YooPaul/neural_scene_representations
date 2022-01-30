import torch
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import os

class ShapeNetCars(torch.utils.data.Dataset):
    '''
    Multiple images of the same car object from different poses
    '''
    def __init__(self, root, image_size, transform=None):
        self.root = root
        self.transform = transform
        self.image_size = image_size

        # save filenames without extention
        self.files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.root, 'rgb'))]

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        img_path = os.path.join(self.root, 'rgb', self.files[index] + '.png')
        intrinsics_path = os.path.join(self.root, 'intrinsics', self.files[index] + '.txt')
        pose_path = os.path.join(self.root, 'pose', self.files[index] + '.txt')

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        with open(pose_path) as f:
            tokens = f.read()[:-1].split(' ')
            assert len(tokens) == 16
            R_inv = torch.tensor([float(tokens[i]) for i in range(12) if (i + 1) % 4 !=  0]).reshape(3,3) # 3x3
            R_inv_neg_t = torch.tensor([float(tokens[i]) for i in range(12) if (i + 1) % 4 ==  0]) # 3
            R = R_inv.transpose(0,1)
            t = -R @ R_inv_neg_t

        with open(intrinsics_path) as f:
            tokens = f.read()[:-1].split(' ')
            K = torch.tensor([float(token) for token in tokens]).reshape(3,3) # 3x3
            height = K[1][2] * 2
            focal = K[0][0]
            new_focal = self.image_size / height * focal
            K[0][0] = new_focal
            K[1][1] = new_focal
            if self.image_size is not None:
                # correct c_x and c_y based on resized image size
                K[0][2] = self.image_size // 2
                K[1][2] = self.image_size // 2


        return img, K, R, t

# If using pose estimates from COLMAP
class COLMAPData(torch.utils.data.Dataset):
    '''
    Multiple images of the same object from different poses.
    Using output from COLMAP.
    '''
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.cameras, self.images = get_camera_info(root)
        self.image_names = list(self.images.keys())
                                                 

    def __getitem__(self, index):

        image = self.images[self.image_names[index]]
        fullpath = os.path.join(self.root, image[:-1]) # get rid of \n character

        img = Image.open(fullpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        extrinsics = self.images[image]
        intrinsics = self.cameras[extrinsics['c_id']]

        R = extrinsics['R']
        t = extrinsics['t']
        #camera_pos = - R.T @ t

        f = intrinsics['f']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        K = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        return img, K, torch.from_numpy(R), torch.from_numpy(t).flatten()

    def __len__(self):
        return len(self.image_names)

def get_camera_info(base_path):
    cameras = {}
    images = {}

    with open(os.path.join(base_path,'cameras.txt')) as f:
        lines = f.readlines()
        for i in range(3, len(lines)):
            vals = lines[i].split(' ')

            camera_id = int(vals[0])
            intrinsics = {}

            intrinsics['W'] = int(vals[2])
            intrinsics['H'] = int(vals[3])

            intrinsics['f'] = float(vals[4])
            intrinsics['cx'] = int(vals[5])
            intrinsics['cy'] = int(vals[6])

            intrinsics['k'] = float(vals[7])

            cameras[camera_id] = intrinsics

    with open(os.path.join(base_path,'/images.txt')) as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):
            vals = lines[i].split(' ')
            image_name = vals[-1]
            extrinsics = {}

            qw = float(vals[1])
            qx = float(vals[2])
            qy = float(vals[3])
            qz = float(vals[4])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            
            tx  = float(vals[5])
            ty = float(vals[6])
            tz = float(vals[7])
            t = np.array([tx, ty, tz]).reshape((3,1))

            extrinsics['R'] = R
            extrinsics['t'] = t
            extrinsics['c_id'] = int(vals[8])
            
            images[image_name] = extrinsics

    return cameras, images
