import torch
from torch import nn

from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet101

from efficientnet_pytorch import EfficientNet


"""
Here every model to be used for pretraining/training is defined.
"""

# Resnet

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

# EfficientNet

class RawEfficientnetB4(nn.Module):
    def __init__(self):
        super(RawEfficientnetB4, self).__init__()
        
        base_model = EfficientNet.from_name('efficientnet-b4', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out

class PlainEfficientnetB4(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB4, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB5(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB5, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB7(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB7, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=6)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out
