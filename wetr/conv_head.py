import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation, bias=False)

class LargeFOV(nn.Module):
    def __init__(self, in_planes, out_planes,):
        super(LargeFOV, self).__init__()

        self.conv6 = conv3x3(in_planes=in_planes, out_planes=in_planes, padding=12, dilation=12)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = conv3x3(in_planes=in_planes, out_planes=in_planes, padding=12, dilation=12)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = conv1x1(in_planes=in_planes, out_planes=out_planes, padding=0)

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        return None

    def forward(self, x):
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        out = self.conv8(x)

        return out