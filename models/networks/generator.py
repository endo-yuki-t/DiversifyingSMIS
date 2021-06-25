"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
                
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        
        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.embedding = nn.Embedding(opt.semantic_nc, opt.z_dim_local)
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
        
        if opt.use_vae:
            tmp_semantic_nc = opt.semantic_nc
            opt.semantic_nc = opt.z_dim_local
        
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        
        if opt.use_vae:
            opt.semantic_nc = tmp_semantic_nc
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
            
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input.clone()
        
        if self.opt.use_vae:
            label_map = seg.argmax(dim=1)
            label_maps = [F.interpolate(label_map.float().unsqueeze(1), size=(int(label_map.shape[1]/(2**i)), int(label_map.shape[2]/(2**i))), mode='nearest')[:,0] for i in range(1,6)]
            embed_seg_list = [self.embedding(label_maps[layer_id].long()).permute(0,3,1,2) for layer_id in reversed(range(len(label_maps)))]
            
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = {}
                unique_labels = label_map.unique().cpu().numpy()
                for label in unique_labels:
                    _z = torch.randn(input.size(0), self.opt.z_dim,
                                    dtype=torch.float32, device=input.get_device())
                    z[label] = _z
                _z = torch.randn(input.size(0), self.opt.z_dim,
                                    dtype=torch.float32, device=input.get_device())
                z['global'] = _z
            
            z_maps_list = []
            for layer_id in reversed(range(len(label_maps))):
                z_maps = []
                for b in range(label_map.shape[0]):
                    z_map = torch.zeros(label_maps[layer_id].shape[1:]+(self.opt.z_dim_local,), dtype=torch.float32)
                    z_map = z_map.cuda() if len(self.opt.gpu_ids) > 0 else z_map
                    unique_labels = label_maps[layer_id][b].unique().cpu().numpy()
                    for label in unique_labels:
                        label = int(label)
                        z_map[label_maps[layer_id][b]==label] = z[label][b,layer_id]
                    z_maps.append(z_map.permute(2,0,1).unsqueeze(0))
                z_maps = torch.cat(z_maps,0)
                z_maps_list.append(z_maps)

            x = self.fc(z['global'])
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)

        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
            
        seg = embed_seg_list[0]+z_maps_list[0] if self.opt.use_vae else seg
        x = self.head_0(x, seg)
        x = self.up(x)
        seg = embed_seg_list[1]+z_maps_list[1] if self.opt.use_vae else seg
        x = self.G_middle_0(x, seg)
        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        seg = embed_seg_list[2]+z_maps_list[2] if self.opt.use_vae else seg
        x = self.up_0(x, seg)
        x = self.up(x)
        seg = embed_seg_list[3]+z_maps_list[3] if self.opt.use_vae else seg
        x = self.up_1(x, seg)
        x = self.up(x)
        seg = embed_seg_list[4]+z_maps_list[4] if self.opt.use_vae else seg
        x = self.up_2(x, seg)
        x = self.up(x)
        seg = input.clone() if self.opt.use_vae else seg
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
