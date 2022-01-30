import torch
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.utils.tensorboard as tensorboard
import os

from models import *
from utils import *
from data import *


EPOCHS = 90
BATCH_SIZE = 1
MODEL_DATA = "Models/SRN_model.pt"
DATASET_ROOT = '/content/drive/MyDrive/cars_train/1641efa5c92514d86c4f4dbcda5f2fc0'

lr = 5e-5
z_dim = 256
image_channels = 3
image_size = 64
lambda_image = 200
lambda_depth = 10
lambda_latent = 1
single_scene = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)


# Set up tensorboard writers
summary_writer = tensorboard.SummaryWriter('Tensorboard/SRN/logs/')

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(), #  normalizes image pixel values to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # shifts the image pixels values to [-1, 1]
])


train_dataset =  ShapeNetCars(DATASET_ROOT, preprocess, image_size)
train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

fixed_view = train_dataset[0]

rmlstm = RMLSTM().to(device)
pixel_generator = PixelGenerator().to(device)
rmlstm.train()
pixel_generator.train()

parameters = list(rmlstm.parameters()) + list(pixel_generator.parameters())
if single_scene:
    srn=SRN().to(device)
    srn.train()
    parameters += list(srn.parameters())
else:
    hypernet = HyperNetwork().to(device)
    z = torch.rand((z_dim), requires_grad=True, device=device)
    parameters += list(hypernet.parameters()) + [z]


# optimizer updates parameters of the hypernetwork, ray-marching lstm, pixel generator, and latent code z
optim = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))
step = 0

if os.path.isfile(MODEL_DATA):
    checkpoint = torch.load(MODEL_DATA)
    rmlstm.load_state_dict(checkpoint['r_model_state_dict'])
    pixel_generator.load_state_dict(checkpoint['p_model_state_dict'])
    if not single_scene:
        hypernet.load_state_dict(checkpoint['h_model_state_dict'])
        z.data = checkpoint['latent_code']
        hypernet.train()

    else:
        srn.load_state_dict(checkpoint['s_model_state_dict'])
        srn.train()


    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    print('Previous step:', step)
    print('Previous generator loss', loss)

    rmlstm.train()
    pixel_generator.train()


else:
    # Initialize model weights
    pixel_generator.apply(init_weights)
    hypernet.apply(init_weights)

grid_x, grid_y = torch.meshgrid(torch.linspace(0,64, image_size), torch.linspace(0,64, image_size), indexing='xy')
uv = torch.cat(tuple(torch.dstack([grid_x, grid_y]))).to(device)

for epoch in range(EPOCHS):
    for idx, (image, K, R, t) in enumerate(train):
        # one image at a time
        image = image.squeeze(0).to(device) # 3 x image_size x image_size
        K = K.squeeze(0).to(device)
        R = R.squeeze(0).to(device)
        t = t.squeeze(0).to(device)
        pixel_values = image[:,uv[:,1], uv[:,0]].transpose(0, 1) # image_size^2 x 3

        if not single_scene:
            # generate SRN from the hypernetwork
            srn = hypernet(z)
        optim.zero_grad()

        # image_size^2 x 3, image_size^2 x 1
        intersection_coords, Zs = find_intersection(srn, rmlstm, K, R, t, uv, device)
        v = srn(intersection_coords) # image_size^2 x 256
        predicted_pixel_values = pixel_generator(v) # image_size^2 x 3

        image_loss = lambda_image * F.mse_loss(predicted_pixel_values, pixel_values, reduction='mean')
        distance_loss = lambda_depth  * torch.square(torch.min(Zs, torch.zeros_like(Zs))).mean() # a regularization term to constrain the depths to be positive

        loss = image_loss + distance_loss
        if not single_scene:
            loss += lambda_latent * torch.square(z).mean() # a regularization term: squared 2-norm of z
        loss.backward()

        optim.step()

        step += 1
        # Print loss and save model
        if (step - 1) % 5 == 0:
            summary_writer.add_scalar('Loss', loss.item(), global_step=step)

        if (step - 1) % 100 == 0:
            print('Epoch: %d/%d\tBatch: %03d/%d\tImage Loss: %f\tDistance Loss:  %f\tMin,Max Distance: %f,%f' %
                  (epoch, EPOCHS, idx, len(train), image_loss.item(), distance_loss.item(), Zs.min(), Zs.max()))
            state_dict = {'step': step,
                          'r_model_state_dict': rmlstm.state_dict(),
                          'p_model_state_dict': pixel_generator.state_dict(),
                          'optimizer_state_dict': optim.state_dict(),
                          'loss': loss.item(),
                         }
            if single_scene:
                state_dict['s_model_state_dict'] = srn.state_dict()
            else:
                state_dict['h_model_state_dict'] = hypernet.state_dict()
                state_dict['latent_code'] = z.data

            torch.save(state_dict, MODEL_DATA)


        if (step - 1) % 50 == 0:
            with torch.no_grad():
                image = image.to(device).unsqueeze(0) # 1x 3 x image_size x image_size
                predicted_image = predicted_pixel_values.transpose(0,1).reshape(3, image_size, image_size).unsqueeze(0)
                grid =  utils.make_grid(torch.cat((predicted_image, image), dim=0), scale_each=False, normalize=True)

                summary_writer.add_image('Model Rendering vs Ground Truth', grid, global_step=step)



