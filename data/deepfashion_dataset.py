"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import util.util as util
from PIL import Image
import numpy as np

class DeepFashionDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=16)
        parser.set_defaults(aspect_ratio=1.)
        parser.set_defaults(batchSize=1)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(ngf=160)
        parser.set_defaults(z_dim_local=16)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
                
        label_dir = os.path.join(root)
        label_paths_all = make_dataset(label_dir+'/'+phase, recursive=True)
        label_paths = [p for p in label_paths_all if 'png' in p]

        image_dir = os.path.join(root)
        image_paths_all = make_dataset(image_dir+'/'+phase, recursive=True)
        image_paths = [p for p in image_paths_all if 'jpg' in p]
        
        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        
        instance_paths = []
        
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return name1 == os.path.splitext(name2)[0]+'_label.png'
    
    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0

        # input image (real images)
        image_path = self.image_paths[index]
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

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict