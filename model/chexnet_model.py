import torch
import torch.nn as nn
import torchvision.models as models
import os

class CheXNet(nn.Module):
    def __init__(self, num_classes=14, use_pretrained=True):
        super(CheXNet, self).__init__()
        
        try:
            # New weights system (torchvision >= 0.13)
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.densenet121 = models.densenet121(weights=weights)
        except:
            # Legacy system
            self.densenet121 = models.densenet121(pretrained=use_pretrained)
        
        # Modify the classifier for CheXNet
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.densenet121(x)
    
    def load_pretrained_chexnet(self, weights_path=None):
        """Load weights if available, otherwise use ImageNet pre-trained"""
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'])
                else:
                    self.load_state_dict(checkpoint)
                print("CheXNet weights loaded successfully!")
            except Exception as e:
                print(f"Error loading CheXNet weights: {e}")
                print("Using ImageNet pre-trained weights as fallback")
        else:
            print("Using ImageNet pre-trained DenseNet-121 weights")

def create_chexnet_model(weights_path=None, num_classes=14):
    """Create CheXNet model with optional weights"""
    model = CheXNet(num_classes=num_classes)
    
    if weights_path:
        model.load_pretrained_chexnet(weights_path)
    
    return model
