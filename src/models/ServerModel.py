import torch.nn as nn
import models.layers as nl
import torch.nn.functional as F
import torch
import models.ServerDataset
from torch.utils.data import DataLoader

class ServerModel(nn.Module):

    def __init__(self, dataset_history, dataset2num_classes, args):
        super(ServerModel, self).__init__()
        self.device = args.device
        self.numclass = args.numclass

        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes  
        self.args = args
        self.criterion_KL = KL_Loss(self.args.temperature)

        if self.datasets:
            self._reconstruct_classifiers()

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            b4 = nn.Sequential(*resnet_block(128, 256, 2))
            b5 = nn.Sequential(*resnet_block(256, 512, 2))
            net = nn.Sequential(b4, b5,
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(), nn.Linear(512, 10))
            self.classifiers.append(net)

    def add_dataset(self, dataset, num_classes):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            b4 = nn.Sequential(*resnet_block(128, 256, 2))
            b5 = nn.Sequential(*resnet_block(256, 512, 2))
            net = nn.Sequential(b4, b5,
                                nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(), nn.Linear(512, 10))
            self.classifiers.append(net)

    def set_dataset(self, dataset):
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.classifier(x)
        return x

    def compute_loss(self, output, target):
        output, target = output.cuda(self.device), target.cuda(self.device)
        loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
        return loss_cur

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        return F.kl_div(F.log_softmax(output_batch/self.T, dim=1), F.softmax(teacher_outputs/self.T, dim=1)) * (self.T * self.T)
        