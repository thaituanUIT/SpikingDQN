import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F

class Backbone(nn.Module):
    """Base class for all backbones"""
    def __init__(self):
        super(Backbone, self).__init__()
        self.normalize = None
        self.output_dim = None

    def forward(self, x):
        if self.normalize:
            x = self.normalize(x)
        return self._extract(x)

    def _extract(self, x):
        raise NotImplementedError

class VGG16Backbone(Backbone):
    def __init__(self, pretrained=True, freeze=True):
        super(VGG16Backbone, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.output_dim = 25088 # 7*7*512
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.features(x)
        return x.reshape(x.size(0), -1)

class ResNetBackbone(Backbone):
    def __init__(self, model_name='resnet18', pretrained=True, freeze=True):
        super(ResNetBackbone, self).__init__()
        resnet = getattr(models, model_name)(pretrained=pretrained)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            self.output_dim = out.reshape(1, -1).size(1)
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.features(x)
        return x.reshape(x.size(0), -1)

class EfficientNetBackbone(Backbone):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, freeze=True):
        super(EfficientNetBackbone, self).__init__()
        effnet = getattr(models, model_name)(pretrained=pretrained)
        self.features = effnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
                
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.pool(self.features(dummy))
            self.output_dim = out.reshape(1, -1).size(1)
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.reshape(x.size(0), -1)

class MobileNetBackbone(Backbone):
    def __init__(self, model_name='mobilenet_v3_small', pretrained=True, freeze=True):
        super(MobileNetBackbone, self).__init__()
        mobilenet = getattr(models, model_name)(pretrained=pretrained)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
                
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.pool(self.features(dummy))
            self.output_dim = out.reshape(1, -1).size(1)
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.reshape(x.size(0), -1)

class ViTBackbone(Backbone):
    def __init__(self, model_name='vit_b_16', pretrained=True, freeze=True):
        super(ViTBackbone, self).__init__()
        vit = getattr(models, model_name)(pretrained=pretrained)
        vit.heads = nn.Identity()
        self.features = vit
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
                
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            self.output_dim = out.reshape(1, -1).size(1)
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.features(x)
        return x.reshape(x.size(0), -1)

def get_backbone(name, pretrained=True, freeze=True):
    if name == 'vgg16':
        return VGG16Backbone(pretrained, freeze)
    elif name == 'resnet18':
        return ResNetBackbone('resnet18', pretrained, freeze)
    elif name == 'efficientnet':
        return EfficientNetBackbone('efficientnet_b0', pretrained, freeze)
    elif name == 'mobilenet':
        return MobileNetBackbone('mobilenet_v3_small', pretrained, freeze)
    elif name == 'vit':
        return ViTBackbone('vit_b_16', pretrained, freeze)
    else:
        raise ValueError(f"Unknown backbone: {name}")

class DQN(nn.Module):
    def __init__(self, input_dim, outputs=9):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024),            
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=outputs)
        )
    def forward(self, x):
        return self.classifier(x)