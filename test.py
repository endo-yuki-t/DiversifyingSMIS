"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
import cv2
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    
    if opt.style_id>-1:
        label_tensor, image_tensor, style_path = dataloader.dataset.get_label_and_image_from_idx(opt.style_id)
        data_i['label_style'] = label_tensor.unsqueeze(0)
        data_i['image_style'] = image_tensor.unsqueeze(0)
        print('style:', style_path)

    img_path = data_i['path']    
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])      
        if opt.style_id>-1:
            img_path[b:b + 1] = [os.path.splitext(img_path[b])[0]+'_'+os.path.basename(style_path)]
            
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
