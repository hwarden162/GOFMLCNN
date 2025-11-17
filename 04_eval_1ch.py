import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image

# ---------------------------------------
# CONFIG
# ---------------------------------------
test_dir = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/image_patches/test"
model_path = "saved_models/resnet101_best_1ch.pth"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------
# DATA TRANSFORMATIONS (RGB Model)
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

train_transforms = transforms.Compose([
    grayscale_and_threshold,
    transforms.ToTensor()
])

# ---------------------------------------
# LOAD THE TESTING DATASET
# ---------------------------------------
test_dataset = datasets.ImageFolder(test_dir, transform=train_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------------------------
# LOAD THE MODEL
# ---------------------------------------
# Initialize the model architecture (same as in training)
model = models.resnet101(weights=None)  # No pretrained weights

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Replace the fully connected layer for the correct number of output classes
in_features = model.fc.in_features
num_classes = 2  # Replace with the actual number of classes
model.fc = nn.Linear(in_features, num_classes)

# Load the trained model weights
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# ---------------------------------------
# EVALUATE ACCURACY
# ---------------------------------------
model.eval()  # Set the model to evaluation mode

correct = 0
total = 0

# Disable gradient computation during evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Update total and correct counts
        total += labels.size(0)
        correct += torch.sum(preds == labels.data)

# Calculate accuracy
accuracy = correct.double() / total
print(f"Testing Accuracy: {accuracy:.4f}")
