import os
import torch
from torch import nn
import torchvision
from src.model import PlainResnet18, PlainResnet50, PlainEfficientnetB4

from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet
import IPython


class ModelWrapper(nn.Module):
    def __init__(self, base_model):
        super(ModelWrapper, self).__init__()
        
        self.block = nn.Sequential(
            base_model
        )
        
    def forward(self, x):
        out = self.block(x)
        return out

    
class CustomModel(nn.Module):
    """
    To add custom layers in base model, e.g. sigmoid layer.
    """
    def __init__(self, base_model, pretrained_model):
        super(CustomModel, self).__init__()
        self.block = nn.Sequential(
            base_model,
            #nn.Sigmoid(),
        )
        
        if pretrained_model == 'efficientnet':
            nn.init.xavier_normal_(self.block[0]._fc.weight)
        elif pretrained_model == 'resnet50':
            #IPython.embed(); exit(1)
            nn.init.xavier_normal_(self.block[0].block[0].fc[0].weight)
            nn.init.xavier_normal_(self.block[0].block[0].fc[2].weight)
    
    def forward(self, x):
        out = self.block(x)
        return out


class CallPretrainedModel():
    """
    model_type: [resnet50, efficientnet]
    """
    def __init__(self, train=True, model_index = None, model_type=None, path='./pretrained_model'):

        self.model_index = model_index
        self.model_type = model_type
        
        if model_type == 'resnet50':
        
            weight_path = os.path.join(path, 'pretrained_resnet.pth')
            base_model = resnet50()
            base_model.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 6),
            )
            base_model = ModelWrapper(base_model)
            model = CallPretrainedModel._load_weights(base_model, weight_path)
            
        elif model_type == 'efficientnet':
            
            weight_path = os.path.join(path, 'pretrained_efficientnet.pth')
            base_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=6)
            model = CallPretrainedModel._load_weights(base_model, weight_path)
        
        else:
            raise Exception(f"No such pretrained model: {model_type}")
        
        self.return_model = model
        
        
    def customize(self):
        return_model = CustomModel(self.return_model, self.model_type)
        return return_model
    
    
    @staticmethod
    def _load_weights(model, path):
        model.load_state_dict(torch.load(path))
        return model
