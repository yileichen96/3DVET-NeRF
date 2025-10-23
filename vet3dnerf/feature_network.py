import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        padding_mode="reflect",
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode="reflect"
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

        
class Easy_Conv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        outplanes,
        stride=1,
        groups=1,
        dilation=1,
    ):
        super(Easy_Conv2d, self).__init__()

        self.in_layer = nn.Sequential(conv3x3(inplanes, planes, stride, groups, dilation),
                                    nn.BatchNorm2d(planes, track_running_stats=False, affine=True),
                                    nn.ReLU(inplace=True))
        self.layer = BasicBlock(inplanes=planes,
                                 planes=planes,
                                 stride=stride,
                                 groups=groups,
                                 dilation=dilation,
                                 norm_layer=nn.BatchNorm2d)

        self.out_layer = nn.Sequential(conv3x3(planes, outplanes, stride, groups, dilation),
                                    nn.BatchNorm2d(planes, track_running_stats=False, affine=True),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.in_layer(x)
        x2 = self.layer(x1)
        out = self.out_layer(x2)

        return out
