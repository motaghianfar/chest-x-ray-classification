# setup_weights.py (Updated)
import torch
import torchvision.models as models
import os

def setup_model():
    print("Setting up CheXNet model...")
    
    # Create weights directory
    os.makedirs('weights', exist_ok=True)
    
    try:
        # New way (torchvision >= 0.13)
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        print("✓ Loaded DenseNet-121 with IMAGENET1K_V1 weights")
    except:
        # Fallback for older versions
        model = models.densenet121(pretrained=True)
        print("✓ Loaded DenseNet-121 with pretrained weights (legacy)")
    
    # Modify for CheXNet
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 14)
    
    # Save modified model
    torch.save(model.state_dict(), 'weights/chexnet_weights.pth')
    print("✅ Model weights saved to: weights/chexnet_weights.pth")
    print("✅ You can now run: streamlit run app.py")

if __name__ == "__main__":
    setup_model()
