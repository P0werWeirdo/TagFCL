import torch.nn as nn
import models.layers as nl
import pdb
import copy as cp
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nl.SharableConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nl.SharableConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Note that here we rename `fc` layer from official torchvision model as `classifier` layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(int(inplanes), width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = norm_layer(int(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, dataset_history, dataset2num_classes, network_width_multiplier, shared_layer_info, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.network_width_multiplier = network_width_multiplier
        self.shared_layer_info = shared_layer_info
        self.inplanes = int(64 * network_width_multiplier)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nl.SharableConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        network_width_multiplier = int(network_width_multiplier)

        self.layer1 = self._make_layer(block, network_width_multiplier * 64, layers[0])
        self.layer2 = self._make_layer(block, network_width_multiplier * 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, network_width_multiplier * 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, network_width_multiplier * 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes
        # we delete default self.classifier for imagenet, because we manually add it in packnet_imagenet_main.py

        if self.datasets:
            self._reconstruct_classifiers()
        
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                #nn.init.constant_(m.weight, 0)
                nn.init.normal_(m.weight, 0, 0.001)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(512, num_classes))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        result_planes = int(planes * block.expansion)
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != result_planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, result_planes, stride),
                norm_layer(result_planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = result_planes 
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def add_dataset(self, dataset, num_classes):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(512, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)

    def set_dataset(self, dataset):
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def copy_dataset(self, dataset, model):
        assert dataset in self.datasets
        self.classifiers[self.datasets.index(dataset)] = cp.deepcopy(model.classifier)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        extract_feature = x
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, extract_feature


def resnet18(dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], dataset_history, dataset2num_classes, **kwargs)

def resnet34(dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], dataset_history, dataset2num_classes, **kwargs)

def resnet50(dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], dataset_history, dataset2num_classes, **kwargs)
    
def resnet101(dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], dataset_history, dataset2num_classes, **kwargs)

def resnet152(dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], dataset_history, dataset2num_classes, **kwargs)

def resnext50_32x4d(dataset_history=[], dataset2num_classes={}, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], dataset_history, dataset2num_classes, groups=4, width_per_group=32, **kwargs)


def resnext101_32x8d(dataset_history=[], dataset2num_classes={}, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], dataset_history, dataset2num_classes, groups=8, width_per_group=32, **kwargs)
