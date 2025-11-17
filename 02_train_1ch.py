import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from PIL import Image
import time
import copy
import os

# ---------------------------------------
# CONFIG
# ---------------------------------------
train_dir = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/image_patches/train"
num_classes = 2
batch_size = 32
lr = 1e-4
num_epochs = 100
patience = 10
val_ratio = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------
# DATA TRANSFORMS
# ---------------------------------------
class GrayscaleAndThreshold:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, img):
        img = img.convert('L')
        img_np = np.array(img)
        img_np = (img_np > self.threshold).astype(np.float32)
        img = Image.fromarray((img_np * 255).astype(np.uint8))  # multiply by 255 to bring it to range [0, 255]
        
        return img

grayscale_and_threshold = GrayscaleAndThreshold(threshold=0)

right_angle_rotation = transforms.RandomChoice([
    transforms.RandomRotation((0, 0)),
    transforms.RandomRotation((90, 90)),
    transforms.RandomRotation((180, 180)),
    transforms.RandomRotation((270, 270)),
])

train_transforms = transforms.Compose([
    grayscale_and_threshold,
    right_angle_rotation,
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    grayscale_and_threshold,
    right_angle_rotation,
    transforms.ToTensor()
])

# ---------------------------------------
# LOAD FULL DATASET AND SPLIT
# ---------------------------------------
full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

train_size = int((1 - val_ratio) * len(full_dataset))
val_size   = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Override transform for validation subset
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

dataloaders = {"train": train_loader, "val": val_loader}

# ---------------------------------------
# LOAD RESNET-101 (NO PRETRAINED WEIGHTS)
# ---------------------------------------
model = models.resnet101(weights=None)

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Replace classifier head
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------------------------------------
# EARLY STOPPING VARIABLES
# ---------------------------------------
best_loss = np.inf
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

# ---------------------------------------
# TRAINING LOOP WITH EARLY STOPPING
# ---------------------------------------
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ----- EARLY STOPPING CHECK -----
        if phase == "val":
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print("→ Validation loss improved, saving model.")
            else:
                epochs_no_improve += 1
                print(f"→ No improvement for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        print("Early stopping triggered!")
        break

print("Training complete.")
model.load_state_dict(best_model_wts)

# ---------------------------------------
# SAVE THE BEST MODEL
# ---------------------------------------
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/resnet101_best_1ch.pth")
print("Best model saved as resnet101_best_1ch.pth")
