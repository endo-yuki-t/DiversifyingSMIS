"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, opt.z_dim)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, opt.z_dim)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt
        
        self.nc = opt.z_dim_local
        self.fc_mu1 = nn.Linear(ndf, self.nc)
        self.fc_var1 = nn.Linear(ndf, self.nc)
        self.fc_mu2 = nn.Linear((ndf*2), self.nc)
        self.fc_var2 = nn.Linear((ndf*2), self.nc)
        self.fc_mu3 = nn.Linear((ndf*4), self.nc)
        self.fc_var3 = nn.Linear((ndf*4), self.nc)
        self.fc_mu4 = nn.Linear((ndf*8), self.nc)
        self.fc_var4 = nn.Linear((ndf*8), self.nc)
        self.fc_mu5 = nn.Linear((ndf*8), self.nc)
        self.fc_var5 = nn.Linear((ndf*8), self.nc)

    def forward(self, x, seg=None):

        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
            seg = F.interpolate(seg, size=(256, 256), mode='nearest')
        
        x_list = []
        x = self.layer1(x)
        x_list.append(x)
        x = self.layer2(self.actvn(x))
        x_list.append(x)
        x = self.layer3(self.actvn(x))
        x_list.append(x)
        x = self.layer4(self.actvn(x))
        x_list.append(x)
        x = self.layer5(self.actvn(x))
        x_list.append(x)
        
        label_map = seg.argmax(dim=1).float()
        label_maps = [F.interpolate(label_map.unsqueeze(1), size=(int(seg.shape[2]/(2**i)), int(seg.shape[3]/(2**i))), mode='nearest')[:,0] for i in range(1,6)]

        label_mu = {}
        label_logvar = {}
        is_kld_computed = {}
        unique_labels = label_map.unique().cpu().numpy()

        for label in unique_labels:
            label = int(label)
            label_mu[label], label_logvar[label], is_kld_computed[label] = [],[],[]
            for b in range(label_map.shape[0]):
                mu_list = []
                logvar_list = []
                kld_bool_list = []
                for _xid, _x in enumerate(x_list):
                    masked_features = _x[b,:,label_maps[_xid][b]==label] 
                    if masked_features.shape[1] != 0:
                        masked_features_max = masked_features.max(dim=1)[0]
                        mun = eval('self.fc_mu%d'%(_xid+1))(self.actvn(masked_features_max).unsqueeze(0))
                        logvarn = eval('self.fc_var%d'%(_xid+1))(self.actvn(masked_features_max).unsqueeze(0))
                        kld_bool_list.append(True)
                    else:
                        mun =  torch.zeros(1,self.nc).cuda() if len(self.opt.gpu_ids)>0 else torch.zeros(1,self.nc)
                        logvarn = torch.zeros(1,self.nc).cuda() if len(self.opt.gpu_ids)>0 else torch.zeros(1,self.nc)
                        kld_bool_list.append(False)

                    mu_list.append(mun)
                    logvar_list.append(logvarn)
                mu_list = torch.cat(mu_list,dim=0)
                logvar_list = torch.cat(logvar_list,dim=0)
                label_mu[label].append(mu_list.unsqueeze(0))
                label_logvar[label].append(logvar_list.unsqueeze(0))
                is_kld_computed[label].append(kld_bool_list)
            label_mu[label] = torch.cat(label_mu[label],dim=0)
            label_logvar[label] = torch.cat(label_logvar[label],dim=0)
        
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        
        label_mu['global'] = mu
        label_logvar['global'] = logvar
        
        return label_mu,label_logvar,is_kld_computed
