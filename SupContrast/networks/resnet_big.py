"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import torchvision.models as models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)  # Apply Max Pool here
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

# Bottleneck is a more complex type of residual block used in 
# larger ResNet architectures like ResNet-50, ResNet-101, and ResNet-152.
def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

# Custom models for CIFAR data:
# the torchvision ResNet uses a kernel size of 7 for the 
# first conv layer. It has drastically lowered the dimension 
# (i.e., H and W) of your feature maps. For CIFAR-10 or 100, 
# such large kernel is not suitable as the input image is just 32x32.
# source: https://github.com/HobbitLong/SupContrast/issues/132
def seresnext50timmCIFAR() -> ResNet:
    # Load the model
    model = timm.create_model('seresnext50_32x4d', pretrained=False, num_classes=0)
    # Change the first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the MaxPool2d layer
    model.maxpool = nn.Identity()

    return model

def resnet50timmCIFAR() -> ResNet:
    # Load the original ResNet50 model
    model = timm.create_model('resnet50', pretrained=False, num_classes=0)
    # Change the first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return model

def resnet50timmCIFARnoMaxpool() -> ResNet:
    # Load the original ResNet50 model
    model = timm.create_model('resnet50', pretrained=False, num_classes=0)
    # Change the first convolutional layer
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the MaxPool2d layer
    model.maxpool = nn.Identity()
    
    return model

def seresnext50timm() -> ResNet:
    model = timm.create_model('seresnext50_32x4d', pretrained=False, num_classes=0)
    return model

def resnet50timm() -> ResNet:
    # Load the original ResNet50 model
    model = timm.create_model('resnet50', pretrained=False, num_classes=0)
    return model

def resnet50pytorchCIFAR() -> ResNet:
    # Load the original ResNet50 model
    model = models.resnet50(weights=None)
    # Change the first layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # Remove the last layer
    modules = list(model.children())[:-1] # remove the last layer
    modules.append(nn.Flatten())  # Add the flattening layer to avoid the [256, 2048, 1, 1] output
    model = torch.nn.Sequential(*modules)

    return model

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'seresnext50timmCIFAR': [seresnext50timmCIFAR, 2048],
    'resnet50timmCIFAR': [resnet50timmCIFAR, 2048],
    'resnet50timmCIFARnoMaxpool': [resnet50timmCIFARnoMaxpool, 2048],
    'seresnext50timm': [seresnext50timm, 2048],
    'resnet50timm': [resnet50timm, 2048],
    'resnet50pytorchCIFAR': [resnet50pytorchCIFAR, 2048],
}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
# The model `seresnext50_32x4d.gluon_in1k` is a variant of the SE-ResNeXt-B image 
# classification model with Squeeze-and-Excitation channel attention⁶. This model 
# was trained on the ImageNet-1K dataset using the Gluon framework⁶. 

# Here are some features of this model⁶:
# - ReLU activations
# - Single layer 7x7 convolution with pooling
# - 1x1 convolution shortcut downsample
# - Grouped 3x3 bottleneck convolutions
# - Squeeze-and-Excitation channel attention

# The Squeeze-and-Excitation (SE) mechanism enables the network to perform dynamic 
# channel-wise feature recalibration¹⁶. The '32x4d' in the name refers to the 
# cardinality and the number of channels in each group convolution¹.

# The specific pre-trained weights for the `seresnext50_32x4d.gluon_in1k` model 
# can be downloaded from the provided URL⁶. As always, the performance of the model 
# can depend on the specific task and dataset you are working with. It's often a 
#     good idea to experiment with different pre-trained models to see which one 
#     works best for your specific use case.

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class FeatureExtractor(nn.Module):
    """backbone only"""
    def __init__(self, name='resnet50'):
        super(FeatureExtractor, self).__init__()
        model_fun, _ = model_dict[name]
        self.encoder = model_fun()

    def forward(self, x):
        return self.encoder(x)