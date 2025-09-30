import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def preprocess_image(image):
    """Preprocess image for CheXNet"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def get_chest_xray_labels():
    """Return CheXNet class labels"""
    return [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
        'Pleural Thickening', 'Hernia'
    ]
