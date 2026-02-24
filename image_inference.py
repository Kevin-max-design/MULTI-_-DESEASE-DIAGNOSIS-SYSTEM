import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


# ═══════════════════════════════════════════════════════════════════
#  GRAD-CAM ENGINE
# ═══════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the last convolutional layer of a CNN to produce a heatmap
    showing which spatial regions most influenced the predicted class.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook to capture activations
        self.target_layer.register_forward_hook(self._forward_hook)
        # Register backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Run a forward + backward pass and compute the Grad-CAM heatmap.
        Returns a numpy array (H, W) normalized to [0, 1].
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Global average pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)  # ReLU to keep only positive influence

        # Resize to input image size (224×224)
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode='bilinear', align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output


def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """
    Overlay a Grad-CAM heatmap on the original image.
    Returns a PIL Image with the heatmap blended on top.
    """
    # Resize original image to 224x224 to match heatmap
    original = original_image.convert('RGB').resize((224, 224))
    original_np = np.array(original).astype(np.float32) / 255.0

    # Apply a colormap (jet) to the heatmap
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap)[:, :, :3]  # Drop alpha channel

    # Blend: overlay = alpha * heatmap + (1 - alpha) * original
    blended = alpha * heatmap_colored + (1 - alpha) * original_np
    blended = np.clip(blended, 0, 1)

    # Convert back to PIL
    blended_uint8 = (blended * 255).astype(np.uint8)
    return Image.fromarray(blended_uint8)


def create_analysis_figure(original_image, heatmap):
    """
    Creates a side-by-side matplotlib figure:
      Left  – Original CT scan
      Right – Grad-CAM heatmap overlay with colorbar
    Returns a matplotlib Figure object.
    """
    original = original_image.convert('RGB').resize((224, 224))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(original)
    axes[0].set_title('Original CT Scan', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Raw heatmap
    im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Activation Heatmap', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    overlay = overlay_heatmap(original_image, heatmap, alpha=0.45)
    axes[2].imshow(overlay)
    axes[2].set_title('Detected Pattern Overlay', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle('Grad-CAM Analysis — Region of Interest Detection',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
#  PNEUMONIA (ResNet-18)
# ═══════════════════════════════════════════════════════════════════

def load_pneumonia_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_pneumonia(image_file, model):
    image = Image.open(image_file)
    transform = get_transforms()
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    pred_idx = torch.argmax(probabilities).item()
    confidence = probabilities[pred_idx].item()

    labels = ["Normal", "Pneumonia"]
    return labels[pred_idx], confidence


# ═══════════════════════════════════════════════════════════════════
#  PANCREATITIS (DenseNet-121 + Grad-CAM)
# ═══════════════════════════════════════════════════════════════════

def load_pancreatitis_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    model.eval()
    return model

def get_ct_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_pancreatitis(image_file, model):
    """
    Runs inference AND Grad-CAM on the uploaded CT scan.
    Returns: (label, confidence, heatmap_figure)
    """
    image = Image.open(image_file)

    transform = get_ct_transforms()
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch.requires_grad_(True)

    # ── Grad-CAM on the last DenseBlock (features.denseblock4) ──
    # DenseNet-121 architecture: features → {... denseblock4, norm5} → classifier
    # We hook into the last dense block for the richest spatial features.
    target_layer = model.features.denseblock4
    grad_cam = GradCAM(model, target_layer)

    heatmap, output = grad_cam.generate(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    pred_idx = torch.argmax(probabilities).item()
    confidence = probabilities[pred_idx].item()

    labels = ["Normal", "Pancreatitis"]

    # Generate the analysis figure
    fig = create_analysis_figure(image, heatmap)

    return labels[pred_idx], confidence, fig


# ═══════════════════════════════════════════════════════════════════
#  PRE-LOAD MODELS at import time
# ═══════════════════════════════════════════════════════════════════
pneumonia_model = load_pneumonia_model()
pancreatitis_model = load_pancreatitis_model()
