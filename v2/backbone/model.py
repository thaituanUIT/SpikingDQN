import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """Base class for all backbones"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.normalize = None
        self.output_dim = None

    def forward(self, x):
        if self.normalize:
            x = self.normalize(x)
        return self._extract(x)

    def get_layers(self):
        raise NotImplementedError

    def get_normalize(self):
        return self.normalize

    def get_output_dim(self):
        return self.output_dim

class VGG16Backbone(FeatureExtractor):
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

    def get_layers(self):
        return self.features

class ResNetBackbone(FeatureExtractor):
    def __init__(self, model_name='resnet18', pretrained=True, freeze=True):
        super(ResNetBackbone, self).__init__()
        resnet = getattr(models, model_name)(pretrained=pretrained)
        # Remove the fully connected layer and the global average pooling
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Determine output dim by a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            self.output_dim = out.reshape(1, -1).size(1)
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.features(x)
        return x.reshape(x.size(0), -1)

    def get_layers(self):
        return self.features

class SimpleConvBackbone(FeatureExtractor):
    def __init__(self, input_channels=3):
        super(SimpleConvBackbone, self).__init__()
        
        # [CONV -> RELU -> CONV -> RELU -> POOL]*3
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 224, 224)
            conv_out = self.conv(dummy)
            self.output_dim = conv_out.reshape(1, -1).size(1)

    def _extract(self, x):
        x = self.conv(x)
        return x.reshape(x.size(0), -1)

    def get_layers(self):
        return self.conv

class FusionBackbone(FeatureExtractor):
    def __init__(self, model_name='resnet18', pretrained=True, freeze=True):
        super(FusionBackbone, self).__init__()
        resnet = getattr(models, model_name)(pretrained=pretrained)
        
        # Split resnet into layers
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2 # Shallow features
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 # Deep features
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
                
        # Determine output dim by a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.stem(dummy)
            x1 = self.layer1(x)
            shallow = self.layer2(x1)
            x3 = self.layer3(shallow)
            deep = self.layer4(x3)
            
            deep_up = F.interpolate(deep, size=shallow.shape[2:], mode='bilinear', align_corners=False)
            fused = torch.cat([shallow, deep_up], dim=1)
            fused_pooled = self.pool(fused)
            self.output_dim = fused_pooled.reshape(1, -1).size(1)
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _extract(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        shallow = self.layer2(x1)
        x3 = self.layer3(shallow)
        deep = self.layer4(x3)
        
        deep_up = F.interpolate(deep, size=shallow.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([shallow, deep_up], dim=1)
        fused_pooled = self.pool(fused)
        return fused_pooled.reshape(fused_pooled.size(0), -1)

    def get_layers(self):
        # This backbone has multiple parallel branches conceptually, 
        # but returning a sequential list of the main path for now.
        return nn.Sequential(self.stem, self.layer1, self.layer2, self.layer3, self.layer4)