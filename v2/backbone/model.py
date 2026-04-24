import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

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
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 224, 224)
            out = self.conv(dummy)
            self.output_dim = out.reshape(1, -1).size(1)

    def _extract(self, x):
        x = self.conv(x)
        return x.reshape(x.size(0), -1)

    def get_layers(self):
        return self.conv