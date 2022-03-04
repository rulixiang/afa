import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
from . import mix_transformer


class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        #self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1, bias=False)


    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;
        
        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups


    def forward(self, x, cam_only=False, seg_detach=True,):

        _x, _attns = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x

        seg = self.decoder(_x)

        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, _attns

        cls_x4 = self.pooling(_x4,(1,1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)

        return cls_x4, seg, _attns
    

if __name__=="__main__":

    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)