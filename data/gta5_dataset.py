"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform
import numpy as np
import util.util as util
from PIL import Image
import torch

class GTA5Dataset(Pix2pixDataset):
    
    def __init__(self):
        super().__init__()
        self.objectid = [0,2,10,19,1]
        self.objectnum = 5
        self.K = 400
        self.L = 10000
        self.rarity_bin=[]
        for i in range(self.objectnum):
            self.rarity_bin.append(np.load("./datasets/gta5/rarity/kdecolor_rarity_bin_%d.npy"%self.objectid[i]))
        self.rarity_mask=np.load("./datasets/gta5/rarity/GTA_weightedrarity_mask.npy",mmap_mode='r')
        self.phase = None
        
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=20)
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(batchSize=1)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(z_dim_local=20)
        opt, _ = parser.parse_known_args()
        
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        self.phase = phase

        label_dir = os.path.join(root, 'LabelgrayFull20', phase)
        self.original_label_paths = make_dataset(label_dir, recursive=True)
        
        image_dir = os.path.join(root, 'RGB256Full', phase)
        self.original_image_paths = make_dataset(image_dir, recursive=True)
        
        instance_paths = []
        
        util.natural_sort(self.original_label_paths)
        util.natural_sort(self.original_image_paths)
        
        if phase == 'test':
            self.L = len(self.original_image_paths)
        
        return self.original_label_paths[:], self.original_image_paths[:], instance_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])
            
    def sample_image_based_on_rarity_bin(self):
        
        label_paths = []
        image_paths = []
        loss_masks = []
        for i in range(self.K):
            idx = np.searchsorted(self.rarity_bin[(i)%self.objectnum],np.random.rand())
            label_paths.append(self.original_label_paths[idx])
            image_paths.append(self.original_image_paths[idx])
            loss_masks.append(self.rarity_mask[idx])

        self.label_paths = label_paths
        self.image_paths = image_paths 
        self.loss_masks = loss_masks

    def __getitem__(self, index):
        if self.phase == 'train':
            index=np.random.randint(self.K)
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
        
        loss_mask = torch.from_numpy(self.loss_masks[index].copy()).permute(2,0,1) if self.phase == 'train' else 0

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'loss_mask': loss_mask
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
    
    def __len__(self):
        return self.L