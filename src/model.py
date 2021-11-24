import torch
from torch import nn
from torchvision.models import (
    resnet18, resnet50, resnet101,
    vgg11, vgg16, vgg19,
    densenet121, densenet169, densenet201
)
from efficientnet_pytorch import EfficientNet

"""
Here every model to be used for pretraining/training is defined.
Plain_OOOO : Models w/o pretrained weights
Pretrained_OOOO : Models w/ pretrained weights
"""

# ResNet-18
class PlainResnet18(nn.Module):
    def __init__(self):
        super(PlainResnet18, self).__init__()
        
        base_model = resnet18()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 128),
            nn.Linear(128, 6)
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out
    
class PretrainedResnet18(nn.Module):
    def __init__(self):
        super(PretrainedResnet18, self).__init__()
        
        base_model = resnet18(pretrained=True)
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 128),
            nn.Linear(128, 6)
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

# ResNet-50    
class PlainResnet50(nn.Module):
    def __init__(self):
        super(PlainResnet50, self).__init__()
        
        base_model = resnet50()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 128),
            nn.Linear(128, 6)
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

class PretrainedResnet50(nn.Module):
    def __init__(self):
        super(PretrainedResnet50, self).__init__()
        
        base_model = resnet50(pretrained=True)
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 128),
            nn.Linear(128, 6)
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

# ResNet-101    
class PlainResnet101(nn.Module):
    def __init__(self):
        super(PlainResnet101, self).__init__()
        
        base_model = resnet101()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 128),
            nn.Linear(128, 6)
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

class PretrainedResnet101(nn.Module):
    def __init__(self):
        super(PretrainedResnet101, self).__init__()
        
        base_model = resnet101(pretrained=True)
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 128),
            nn.Linear(128, 6)
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


# EfficientNet B4
class PlainEfficientnetB4(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB4, self).__init__()
        
        base_model = EfficientNet.from_name('efficientnet-b4', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

class PretrainedEfficientnetB4(nn.Module):
    def __init__(self):
        super(PretrainedEfficientnetB4, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

# EfficientNet B5
class PlainEfficientnetB5(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB5, self).__init__()
        
        base_model = EfficientNet.from_name('efficientnet-b5', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

class PretrainedEfficientnetB5(nn.Module):
    def __init__(self):
        super(PretrainedEfficientnetB5, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

# EfficientNet B7
class PlainEfficientnetB7(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB7, self).__init__()
        
        base_model = EfficientNet.from_name('efficientnet-b7', num_classes=1)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

class PretrainedEfficientnetB7(nn.Module):
    def __init__(self):
        super(PretrainedEfficientnetB7, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

# VggNet11
class PlainVgg11(nn.Module):
    def __init__(self):
        super(PlainVgg11, self).__init__()

        base_model = vgg11()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out


# VggNet16
class PlainVgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        base_model = vgg16()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)
      
    def forward(self, x):
        out = self.block(x)
        return out

# VggNet19
class PlainVgg19(nn.Module):
    def __init__(self):
        super(PlainVgg19, self).__init__()

        base_model = vgg19()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )
      
        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out

class PretrainedVgg19(nn.Module):
    def __init__(self):
        super(PretrainedVgg19, self).__init__()

        base_model = vgg19(pretrained=True)
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out


# DenseNet121
class PlainDensenet121(nn.Module):
    def __init__(self):
        super(PlainDensenet121, self).__init__()

        base_model = densenet121()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out

class PretainedDensenet121(nn.Module):
    def __init__(self):
        super(PretrainedDensenet121, self).__init__()

        base_model = densenet121()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out


# DenseNet169
class PlainDensenet169(nn.Module):
    def __init__(self):
        super(PlainDensenet169, self).__init__()

        base_model = densenet169()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out

# DenseNet201
class PlainDensenet201(nn.Module):
    def __init__(self):
        super(PlainDensenet201, self).__init__()

        base_model = densenet201()
        self.block = nn.Sequential(
           base_model,
           nn.Linear(1000, 128),
           nn.Linear(128, 6),
        )

        nn.init.xavier_normal_(self.block[1].weight)

    def forward(self, x):
        out = self.block(x)
        return out