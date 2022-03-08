### YOUR CODE HERE

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from configure import DEFAULT_CFG

"""This script defines the network.
"""
'''
class MyNetwork(object):

    def __init__(self, configs):
        self.configs = configs

    def __call__(self, inputs, training):

    	Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.

        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        return inputs
'''

class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_ch=3, stem_width=64,
                 down_kernel_size=1, actn=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 seblock=True, reduction_ratio=0.25, dropout_ratio=0.,
                 stochastic_depth_ratio=0., zero_init_last_bn=True):

        super().__init__()

        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.actn = actn
        self.dropout_ratio = float(dropout_ratio)
        self.stochastic_depth_ratio = stochastic_depth_ratio
        self.zero_init_last_bn = zero_init_last_bn
        self.conv1 = StemBlock(in_ch, stem_width, norm_layer, actn)
        channels = [64, 128, 256, 512]
        self.make_layers(block, layers, channels, stem_width*2,
                         down_kernel_size, seblock, reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.expansion=4
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)

    def make_layers(self, block, nlayers, channels, inplanes, kernel_size=1,
                    seblock=True, reduction_ratio=0.25):
        tot_nlayers = sum(nlayers)
        layer_idx = 0
        self.expansion=4
        for idx, (nlayer, channel) in enumerate(zip(nlayers, channels)):
            name = "layer" + str(idx+1)
            stride = 1 if idx == 0 else 2
            downsample = None
            if stride != 1 or inplanes != channel * self.expansion:
                downsample = Downsample(inplanes, channel * self.expansion,
                                        kernel_size=kernel_size, stride=stride,
                                        norm_layer=self.norm_layer)

            blocks = []
            for layer_idx in range(nlayer):
                downsample = downsample if layer_idx == 0 else None
                stride = stride if layer_idx == 0 else 1
                drop_ratio = (self.stochastic_depth_ratio*layer_idx
                              / (tot_nlayers-1))
                #print(type(block))
                blocks.append(block(inplanes, channel, stride, self.norm_layer,self.actn, downsample, seblock,reduction_ratio, drop_ratio,self.zero_init_last_bn))

                inplanes = channel * self.expansion
                layer_idx += 1

            self.add_module(*(name, nn.Sequential(*blocks)))

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1, -1)
        if self.dropout_ratio > 0.:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.fc(x)
        return x


class ResnetRS():
    def __init__(self):
        super().__init__()

    def create_model(self, block, layers, num_classes=10, in_ch=3,
                     stem_width=128, down_kernel_size=1,
                     actn=partial(nn.ReLU, inplace=True),
                     norm_layer=nn.BatchNorm2d, seblock=True,
                     reduction_ratio=0.25, dropout_ratio=0.25,
                     stochastic_depth_rate=0.0,
                     zero_init_last_bn=True):

        return Resnet(block, layers, num_classes=num_classes, in_ch=in_ch,
                      stem_width=stem_width, down_kernel_size=down_kernel_size,
                      actn=actn, norm_layer=norm_layer, seblock=seblock,
                      reduction_ratio=reduction_ratio,
                      dropout_ratio=dropout_ratio,
                      stochastic_depth_ratio=stochastic_depth_rate,
                      zero_init_last_bn=zero_init_last_bn)


    def _get_default_cfg(self):
        cfg = DEFAULT_CFG
        cfg['block'] = Bottleneck
        cfg['layers'] = [3, 4, 6, 3]
        return cfg



class StemBlock(nn.Module):
    def __init__(self, in_ch=3, stem_width=128, norm_layer=nn.BatchNorm2d,
                 actn=nn.ReLU):
        super().__init__()
        inplanes = 2 * stem_width
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(in_ch, stem_width, kernel_size=3, stride=1, padding=1,bias=False),
            norm_layer(stem_width),
            actn(inplace=True),
            nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,padding=1, bias=False),
            norm_layer(stem_width),
            actn(inplace=True),
            nn.Conv2d(stem_width, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        ])
        self.bn1 = norm_layer(inplanes)
        self.actn1 = actn(inplace=True)
        self.maxpool = nn.Sequential(*[
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1,
                      bias=False),
            norm_layer(inplanes),
            actn(inplace=True)
        ])
        self.init_weights()

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x.cuda())
        x = self.bn1(x)
        x = self.actn1(x)
        #out = self.maxpool(x)
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm_layer=nn.BatchNorm2d,
                 actn=nn.ReLU, downsample=None, seblock=True,
                 reduction_ratio=0.25, stochastic_depth_ratio=0.0,
                 zero_init_last_bn=True):
        super().__init__()
        outplanes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.actn1 = actn(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.actn2 = actn(inplace=True)

        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.seblock = seblock
        if seblock:
            self.se = SEBlock(outplanes, reduction_ratio)
        self.actn3 = actn(inplace=True)
        self.down = False
        if downsample is not None:
            self.downsample = downsample
            self.down = True
        self.drop_path = DropPath(stochastic_depth_ratio)
        self.init_weights(zero_init_last_bn)

    def init_weights(self, zero_init_last_bn=True):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if zero_init_last_bn:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = self.downsample(x.cuda()) if self.down else x.cuda()

        x = self.conv1(x.cuda())
        x = self.bn1(x)
        x = self.actn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.seblock:
            x = self.se(x)
        if self.drop_path.drop_prob:
            x = self.drop_path(x)
        x += shortcut
        x = self.actn3(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        if stride == 1:
            avgpool = nn.Identity()
        else:
            avgpool = nn.AvgPool2d(2, stride=stride, ceil_mode=True,
                                   count_include_pad=False)
        self.downsample = nn.Sequential(*[
            avgpool,
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                      padding=0, bias=False),
            norm_layer(out_ch)
        ])
        self.init_weights()

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.downsample(x.cuda())


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super().__init__()
        reduced_channels = int(channels * reduction_ratio)
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.actn = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=x.cuda()
        orig = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv1(x)
        x = self.actn(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return orig * x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        x=x.cuda()
        if self.drop_prob is None or self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],)+(1,)*(x.ndim-1)
        rand_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                             device=x.device)
        rand_tensor = rand_tensor.floor_()
        out = x.div(keep_prob) * rand_tensor
        return out

### END CODE HERE
