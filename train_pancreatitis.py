"""
train_pancreatitis.py — Fine-tune DenseNet-121 on custom CT scan images.

Usage:
    1. Place your CT scan images into:
         data/normal/        ← Healthy pancreas CT scans
         data/pancreatitis/  ← Pancreatitis CT scans

    2. Run:  python train_pancreatitis.py

    The script will:
      • Apply heavy data augmentation (rotation, flip, color jitter, etc.)
      • Split data 80/20 into train/val sets
      • Fine-tune a pre-trained DenseNet-121
      • Save the best model to models/pancreatitis_densenet121.pth
      • Print training curves and final metrics
"""
import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "pancreatitis_densenet121.pth")

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7       # Early stopping patience (epochs with no improvement)
MIN_IMAGES = 10    # Minimum images required per class to start training

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                       else "cuda" if torch.cuda.is_available() 
                       else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════

def get_train_transforms():
    """
    Heavy augmentation for training — helps the model generalize 
    even with a small dataset (100–500 images).
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

def get_val_transforms():
    """Minimal transforms for validation — no augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════

def build_model(num_classes=2, freeze_backbone=True):
    """
    Loads DenseNet-121 pre-trained on ImageNet and modifies the 
    classifier head for binary classification.
    
    Strategy:
      Phase 1 (freeze_backbone=True):  Only train the classifier head.
      Phase 2 (freeze_backbone=False): Unfreeze and fine-tune all layers
                                        with a lower learning rate.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # Freeze backbone layers if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        for param in model.features.parameters():
            param.requires_grad = True
    
    # Replace classifier head
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, patience):
    """
    Full training loop with early stopping and best-model checkpointing.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n{'Epoch':<8} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    print("─" * 55)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            marker = " ✓ saved"
        else:
            epochs_no_improve += 1
        
        print(f"{epoch+1:<8} {train_loss:>11.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>9.4f}{marker}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    model.load_state_dict(best_model_wts)
    return model, best_val_acc, history


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # ── Check data directory ─────────────────────────────────────
    if not os.path.exists(DATA_DIR):
        os.makedirs(os.path.join(DATA_DIR, "normal"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "pancreatitis"), exist_ok=True)
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DATA DIRECTORY CREATED                                      ║
║                                                              ║
║  Please add your CT scan images to:                          ║
║    {DATA_DIR}/normal/         ← Healthy CT scans             ║
║    {DATA_DIR}/pancreatitis/   ← Pancreatitis CT scans        ║
║                                                              ║
║  Supported formats: .jpg, .jpeg, .png, .bmp, .tiff           ║
║  Minimum recommended: 50 images per class                    ║
║  Best results: 200+ images per class                         ║
╚══════════════════════════════════════════════════════════════╝
""")
        sys.exit(0)
    
    # ── Count images ─────────────────────────────────────────────
    class_dirs = [d for d in os.listdir(DATA_DIR) 
                  if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]
    
    if len(class_dirs) < 2:
        print("ERROR: Need at least 2 class folders (e.g., 'normal/' and 'pancreatitis/').")
        sys.exit(1)
    
    print("═" * 60)
    print("  PANCREATITIS CT SCAN TRAINING PIPELINE")
    print("═" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"\nClasses found:")
    
    total_images = 0
    for cls in sorted(class_dirs):
        cls_path = os.path.join(DATA_DIR, cls)
        count = len([f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        total_images += count
        print(f"  • {cls}: {count} images")
    
    if total_images < MIN_IMAGES * 2:
        print(f"\n⚠ WARNING: Only {total_images} images found. Minimum {MIN_IMAGES * 2} recommended.")
        print("  Training may overfit. Add more images for better results.")
    
    # ── Load dataset ─────────────────────────────────────────────
    # Use ImageFolder — expects data/class_name/image.jpg structure
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=get_train_transforms())
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nTotal images: {len(full_dataset)}")
    print(f"Classes: {class_names}")
    
    # 80/20 split
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Override val transforms (no augmentation)
    val_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {train_size} images | Val: {val_size} images")
    
    # ── Handle class imbalance ───────────────────────────────────
    class_counts = [0] * num_classes
    for _, label in full_dataset:
        class_counts[label] += 1
    
    weights = [1.0 / c for c in class_counts]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Class weights: {dict(zip(class_names, [f'{w:.3f}' for w in class_weights.cpu().numpy()]))}")
    
    # ═════════════════════════════════════════════════════════════
    #  PHASE 1: Train classifier head only (backbone frozen)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 1: Training classifier head (backbone frozen)")
    print("═" * 60)
    
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    
    # Only optimize the classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE * 10, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    model, phase1_acc, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=10, patience=5
    )
    print(f"\nPhase 1 best val accuracy: {phase1_acc:.4f}")
    
    # ═════════════════════════════════════════════════════════════
    #  PHASE 2: Fine-tune entire network (backbone unfrozen)
    # ═════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 2: Fine-tuning full network (all layers)")
    print("═" * 60)
    
    # Unfreeze backbone
    for param in model.features.parameters():
        param.requires_grad = True
    
    # Lower learning rate for backbone, higher for classifier
    optimizer = optim.Adam([
        {"params": model.features.parameters(), "lr": LEARNING_RATE * 0.1},
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    model, phase2_acc, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, patience=PATIENCE
    )
    print(f"\nPhase 2 best val accuracy: {phase2_acc:.4f}")
    
    # ── Save the best model ──────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # ── Final evaluation ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL EVALUATION")
    print("═" * 60)
    
    val_loss, val_acc, preds, labels = validate(model, val_loader, criterion)
    
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    print(f"Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"   Classes: {class_names}")
    print(f"   The Streamlit app will auto-load this model on next restart.")


if __name__ == "__main__":
    main()
