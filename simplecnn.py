from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch
import os
import numpy as np



class Model(torch.nn.Module):
    def __init__(self):
        pass


class SimpleCNN_CollectFM(Model):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels = self.in_channels,out_channels =  8, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.fc2 = torch.nn.Linear(128, 10)


    def forward(self, x):
        index = str(int(len(os.listdir())/5))

        f0 = x.cpu().data.numpy()

        #Computes the activation of the first convolution
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        f1 = x.cpu().data.numpy()


        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        f2 = x.cpu().data.numpy()


        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        f3 = x.cpu().data.numpy()

        # Reshape data to input to the input layer of the neural net
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 32 * 3 * 3)
        f4 = x.cpu().data.numpy()

        path = index
        np.save(path+'f0',f0)
        np.save(path+'f1',f1)
        np.save(path+'f2',f2)
        np.save(path+'f3',f3)
        np.save(path+'f4',f4)

        #Computes the activation of the first fully connected layer
        x = F.relu(self.fc1(x))

        #Computes the second fully connected layer (activation applied later)
        x = self.fc2(x)

        return(x)


class SimpleCNN(Model):

    def __init__(self):
        super(Model, self).__init__()
        in_channels = 1

        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(256 * 3 * 3, 2048)
        self.fc2 = torch.nn.Linear(2048, 10 )


    def forward(self, x):

        #Computes the activation of the first convolution
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        #Reshape data to input to the input layer of the neural net
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 256 * 3 * 3)

        #Computes the activation of the first fully connected layer
        x = F.relu(self.fc1(x))

        #Computes the second fully connected layer (activation applied later)
        x = self.fc2(x)

        return(x)




import math
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels):

        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  layers[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, layers[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# def ResNet18():
#
#     return ResNet(ResidualBlock, layers = [2,2,2,2], num_classes = 10, input_channels = 3 )























# import math
# import torch.nn as nn
#
#
#
# class ResNet(torch.nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, input_channels=3):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=2)
#
#         self.dropout = nn.Dropout2d(p=0.5,inplace=True)
#
#         #print "block.expansion=",block.expansion
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             # refer to the paper: https://arxiv.org/abs/1706.02677
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         print(x.shape)
#         x = self.conv1(x)
#         print(x.shape)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         print(x.shape)
#
#         x = self.layer1(x)
#         print(x.shape)
#         x = self.layer2(x)
#         print(x.shape)
#         x = self.layer3(x)
#         print(x.shape)
#         x = self.layer4(x)
#         print(x.shape)
#
#
#         x = self.avgpool(x)
#         x = self.dropout(x)
#         #print "avepool: ",x.data.shape
#         x = x.view(x.size(0), -1)
#         #print "view: ",x.data.shape
#         x = self.fc(x)
#         return x
#
#
#
#
#
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
