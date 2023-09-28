'''
Vimeo7 dataset
support reading images from lmdb, image folder and memcached
'''
import os
import sys
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass
from pdb import set_trace as bp

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass

logger = logging.getLogger('base')


class AdobeDataset(data.Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(AdobeDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.half_N_frames = opt['N_frames'] // 2
        self.LR_N_frames = 1 + self.half_N_frames
        #assert self.LR_N_frames > 1, 'Error: Not enough LR frames to interpolate'
        #### determine the LQ frame list
        '''
        N | frames
        1 | error
        3 | 0,2
        5 | 0,2,4
        7 | 0,2,4,6
        '''

        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        if opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            cache_keys = opt['cache_keys']
        else:
            cache_keys = 'Vimeo7_train_keys.pkl'
        logger.info('Using cache keys - {}.'.format(cache_keys))
        self.paths_GT = pickle.load(open('/home/abcd233746pc/VideoINR-Continuous-Space-Time-Super-Resolution/{}'.format(cache_keys), 'rb'))
     
        #assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))
        
        with open('/work/abcd233746pc/adobe240fps_folder_test.txt') as t:
            video_list = t.readlines()
        #video_list = ['walk', 'foliage', 'city', 'calendar', ]
        video_list = ['calendar']
            
        self.file_list = []
        self.gt_list = []
        interval_num = opt['ref_num'] - 1
        for video in video_list:
            if video[-1] == '\n':
                video = video[:-1]
            interval = 1
            index = 0
            frames = (os.listdir(os.path.join(self.GT_root , video)))
            frames = sorted([int(frame[:-4]) for frame in frames]) 
            
            frames += [frames[-2], frames[-3], frames[-4]]
            print(frames)
            
            #frames = [str(frame) + '.png' for frame in frames]
            frames = ['{:03d}'.format(frame) + '.png' for frame in frames]
            while index + (interval + 1) * interval_num < len(frames) - 0:
                videoInputs = [frames[i] for i in range(index, index + (1 + interval) * interval_num + 1, (1 + interval))]
                video_all_gt = [frames[i] for i in range(index, index + (1 + interval) * (interval_num) + 1 )]
                videoInputs = [os.path.join(video, f) for f in videoInputs]
                videoGts = [os.path.join(video, f) for f in video_all_gt]
                #print(videoInputs)
                #print(videoGts)
                self.file_list.append(videoInputs)
                self.gt_list.append(videoGts)
                index += (1 + interval) * (interval_num)
            
            print(self.file_list, self.gt_list)
                
                
    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        
        scale = self.opt['scale']
        # print(scale)
        N_frames = self.opt['N_frames']
        GT_size = self.opt['GT_size']

        #### get the GT image (as the center frame)
        img_GT_l = []
        img_LQop_l = [osp.join(self.LQ_root, fp) for fp in self.file_list[index]]
        img_GTop_l = np.array([osp.join(self.GT_root, fp) for fp in self.gt_list[index]])
        
        #gt_sampled_idx = [0] + sorted(random.sample(range(len(img_GTop_l)), self.opt['sample_num'])) + [len(img_GTop_l)-1]
        #print(gt_sampled_idx)
        gt_sampled_idx = [0,0,1,2,3,4,5,6,6]
        #gt_sampled_idx = [0,0,4,8,8]
        # print(gt_sampled_idx)
        # gt_sampled_idx = [0, 4, 8]
        img_GTop_l = img_GTop_l[gt_sampled_idx]
        
        times = []
        for i in gt_sampled_idx[1:-1]:
            times.append(torch.tensor([i / 6]))
        img_LQ_l = [cv2.imread(fp) for fp in img_LQop_l]
        img_GT_l = [cv2.imread(fp) for fp in img_GTop_l]
        #print("LQ: ", img_LQo_l, "GT: ", img_GTo_l)
        #print(img_LQ_l, img_GT_l)
        img_LQ_l = [img_.astype(np.float32) / 255. for img_ in img_LQ_l]
        img_GT_l = [img_.astype(np.float32) / 255. for img_ in img_GT_l]
            
        if img_LQ_l[0].ndim == 2:
            img_LQ_l = [np.expand_dims(img_, axis=2) for img_ in img_LQ_l]
            img_GT_l = [np.expand_dims(img_, axis=2) for img_ in img_GT_l]
            
        if img_LQ_l[0].shape[2] > 3:
            img_LQ_l = [img_[:, :, :3] for img_ in img_LQ_l]
            img_GT_l = [img_[:, :, :3] for img_ in img_GT_l]
                
   
            
        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        #for img in img_GT_l:
        #    print(img.shape)
        img_GTs = np.stack(img_GT_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]

        img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GTs, 'key': times, 'time': times}

    def __len__(self):
        return len(self.file_list)
