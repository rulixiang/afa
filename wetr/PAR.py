###
#local pixel refinement
###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_kernel():
    
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight

class PAR(nn.Module):

    def __init__(self, dilations, num_iter,):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d]*4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b*c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
 
        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)
        
        for d in self.dilations:
            pos_xy.append(ker*d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1,1,_imgs.shape[self.dim],1,1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1)**2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1)**2
        #pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks
