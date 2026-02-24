import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def load_pneumonia_model():
    """
    Loads a pre-trained ResNet-18 model modified for binary classification.
    Normally, we would load state_dict (weights) from a trained checkpoint here.
    Since we are building the pipeline without a specific trained .pth file,
    it behaves as a placeholder that will return pseudo-predictions based on
    ImageNet weights, but the structural pipeline is fully functional and ready
    for real fine-tuned medical weights.
    """
    # Load a pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer for binary classification 
    # (0: Normal, 1: Pneumonia)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Here we would do:
    # checkpoint_path = "models/pneumonia_resnet18.pth"
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

def get_transforms():
    """
    Standard ImageNet transforms, suitable for a pre-trained ResNet.
    X-ray images should be resized to 224x224 and normalized.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # Ensure 3 channels for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_pneumonia(image_file, model):
    """
    Takes an uploaded image file from Streamlit, preprocesses it, and runs inference.
    """
    # Load image
    image = Image.open(image_file)
    
    # Preprocess
    transform = get_transforms()
    input_tensor = transform(image)
    
    # Add batch dimension (B, C, H, W)
    input_batch = input_tensor.unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Apply Softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # 0 = Normal, 1 = Pneumonia
    pred_idx = torch.argmax(probabilities).item()
    confidence = probabilities[pred_idx].item()
    
    labels = ["Normal", "Pneumonia"]
    return labels[pred_idx], confidence

# Pre-load the model once when the module is imported
# This saves time during Streamlit reruns
pneumonia_model = load_pneumonia_model()
