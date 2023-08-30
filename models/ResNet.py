import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch

class ResNet50_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        b, _, _, _ = x.size()
        return self.backbone(x).view(-1, 2048)