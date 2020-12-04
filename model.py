import math
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
from torchvision.ops import misc as misc_nn_ops
import timm
from mmcv.cnn import ConvModule, xavier_init, caffe2_xavier_init
from mmcv.cnn.bricks import NonLocal2d


class HRFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pooling_type='AVG',
                 conv_cfg=None,
                 stride=1):
        super(HRFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = len(in_channels)
        self.conv_cfg = conv_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None))

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        outs = [inputs[0]]
        for i in range(1, self.num_ins):
            outs.append(
                F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)

        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []

        for i in range(self.num_outs):
            tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)

        outputs.append(F.max_pool2d(outputs[-1], 1, 2, 0))

        return outputs


class BFP(nn.Module):
    def __init__(self,
                 in_channels_list,
                 out_channels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type

        self.num_levels = len(in_channels_list)
        assert 0 <= self.refine_level < self.num_levels

        self.convs = nn.ModuleList()
        for in_channels in self.in_channels_list:
            self.convs.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1))

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.out_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        inputs = [self.convs[i](x) for i, x in enumerate(inputs)]

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])

        outs.append(F.max_pool2d(outs[-1], 1, 2, 0))

        return outs


class ResNetWithBFP(nn.ModuleDict):
    def __init__(self):
        super(ResNetWithBFP, self).__init__()

        backbone = torchvision.models.resnet.__dict__['resnet50'](pretrained=True,
                                                                  norm_layer=misc_nn_ops.FrozenBatchNorm2d)

        trainable_layers = 3
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        returned_layers = [1, 2, 3, 4]
        return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        self.out_channels = 256

        self.backbone = torchvision.models._utils.IntermediateLayerGetter(backbone,
                                                                          return_layers=return_layers)
        self.neck = BFP(in_channels_list, self.out_channels)

    def forward(self, x):
        x = self.backbone(x)
        x = list(x.values())

        x = self.neck(x)

        out = {'0': x[0], '1': x[1], '2': x[2], '3': x[3], 'pool': x[4]}

        return out


class ResNetWithHRFPN(nn.ModuleDict):
    def __init__(self):
        super(ResNetWithHRFPN, self).__init__()

        backbone = torchvision.models.resnet.__dict__['resnet50'](pretrained=True,
                                                                  norm_layer=misc_nn_ops.FrozenBatchNorm2d)

        trainable_layers = 3
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        returned_layers = [1, 2, 3, 4]
        return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        self.out_channels = 256

        self.backbone = torchvision.models._utils.IntermediateLayerGetter(backbone,
                                                                          return_layers=return_layers)
        self.neck = BFP(in_channels_list, self.out_channels, refine_level=2, refine_type='non_local')

    def forward(self, x):
        x = self.backbone(x)
        x = list(x.values())

        x = self.neck(x)

        out = {'0': x[0], '1': x[1], '2': x[2], '3': x[3], 'pool': x[4]}

        return out


class ResNestWithFPN(nn.Module):
    def __init__(self):
        super(ResNestWithFPN, self).__init__()

        extra_blocks = LastLevelMaxPool()

        backbone = timm.models.resnest50d(pretrained=True)
        trainable_layers = 3
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
            elif name.find('bn') >= 0:
                parameter.requires_grad_(False)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=512,
            extra_blocks=extra_blocks,
        )
        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        feats = {'0': x1, '1': x2, '2': x3, '3': x4}

        feats = self.fpn(feats)
        return feats


class ResnestWithEFM(nn.Module):
    def __init__(self):
        super(ResnestWithEFM, self).__init__()

        extra_blocks = LastLevelMaxPool()

        backbone = timm.models.resnest50d(pretrained=True)
        trainable_layers = 3
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
            elif name.find('bn') >= 0:
                parameter.requires_grad_(False)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn = ExactFusionModel(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=512,
            transition=256,
            extra_blocks=extra_blocks,
        )
        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        feats = {'0': x1, '1': x2, '2': x3, '3': x4}
        feats = self.fpn(feats)

        return feats
