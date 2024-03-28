import os
from time import time
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.Ours as Ours
import models.modules.TMNet as TMNet

from pdb import set_trace as bp
from data.util import imresize_np

parser = argparse.ArgumentParser()
parser.add_argument('--space_scale', type=int, default=4, help="upsampling space scale")
parser.add_argument('--time_scale', type=int, default=6, help="upsampling time scale")
parser.add_argument('--data_path', type=str, required=True, help="data path for testing")
# parser.add_argument('--out_path_lr', type=str, default="./output/LR/", help="output path (Low res image)")
# parser.add_argument('--out_path_bicubic', type=str, default="./output/Bicubic/", help="output path (bicubic upsampling)")
parser.add_argument('--out_path_ours', type=str, default="./output/bmx/VideoINR/", help="output path (VideoINR)")
#parser.add_argument('--model_path', type=str, default="../latest_G.pth", help="model parameter path")
parser.add_argument('--model_path', type=str, default="best.pth", help="model parameter path")
# parser.add_argument('--model_path', type=str, default="/home/abcd233746pc/tmnet_multiple_frames.pth", help="model parameter path")
opt = parser.parse_known_args()[0]

device = 'cuda'
#model = Sakuya_arch.LunaTokis(64, 6, 8, 5, 40)
model = Ours.LunaTokis() 
# model = TMNet.TMNet(64, 6, 8, 5, 40) 
model.load_state_dict(torch.load(opt.model_path), strict=True)

model.eval()
model = model.to(device)

def single_forward(model, imgs_in, space_scale, time_scale):
    with torch.no_grad():
        '''b, n, c, h, w = imgs_in.size()
        h_n = int(4 * np.ceil(h / 4))
        w_n = int(4 * np.ceil(w / 4))
        imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
        imgs_temp[:, :, :, 0:h, 0:w] = imgs_in'''

        time_Tensors = [torch.tensor([i / time_scale])[None].to(device) for i in range(time_scale+1)]
        #model_output = model(imgs_in, time_Tensors, space_scale, test=False)
        
        '''model_output = []
        for i in range(time_scale):
            output = model(imgs_in,None, [torch.tensor([i / time_scale])[None].to(device)], space_scale, use_GT=False)[0][0]
            model_output.append(output)
        model_output = torch.stack(model_output, 0)
        print(model_output.shape)'''
        
        model_output = model(imgs_in, None, time_Tensors, space_scale, use_GT=False)[0]
        return model_output

os.makedirs(opt.out_path_ours, exist_ok=True)

subdirs = os.listdir(opt.data_path)

for subdir in tqdm(sorted(subdirs)):
    data_path = os.path.join(opt.data_path, subdir)

    path_list = [os.path.join(data_path, name) for name in sorted(os.listdir(data_path))]
    index = 0
    for ind in tqdm(range(0, len(path_list)-2, 2)):
        os.makedirs(os.path.join(opt.out_path_ours, subdir), exist_ok=True)

        imgpath1 = os.path.join(path_list[ind])
        imgpath2 = os.path.join(path_list[ind + 2])

        img1 = cv2.imread(imgpath1, cv2.IMREAD_UNCHANGED)[:,:,:3]
        img2 = cv2.imread(imgpath2, cv2.IMREAD_UNCHANGED)[:,:,:3]

        '''
        We apply down-sampling on the original video
        in order to avoid CUDA out of memory.
        You may skip this step if your input video
        is already of relatively low resolution.
        '''
        img1 = img1.astype(np.float32) / 255.
        img2 = img2.astype(np.float32) / 255.

        # Image.fromarray((np.clip(img1[:, :, [2, 1, 0]], 0, 1) * 255).astype(np.uint8)).save(
        #     os.path.join(opt.out_path_lr, path_list[ind].split('/')[-1].replace('.exr', '.png')))

        imgs = np.stack([img1, img2], axis=0)[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].to(device)

        # print(imgs.shape)
        output = single_forward(model, imgs, opt.space_scale, opt.time_scale)
        # print(len(output), output[0].shape)

        '''
        Save results of VideoINR and bicubic up-sampling.
        '''
        for out_ind in range(len(output)):
            img = output[out_ind][0]
            img = Image.fromarray((img.clamp(0., 1.).detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8))
            img.save(os.path.join(opt.out_path_ours, subdir, '{:08d}.png'.format(index+1)))
            index += 1

        index -= 1
