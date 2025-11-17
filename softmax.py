import torch
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn
import os
import numpy as np

# ---------------------------------------
# CONFIG
# ---------------------------------------
test_dir = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/image_patches/test"  # Path to your test data directory
model_path = "saved_models/resnet101_best.pth"  # Path to your saved model
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------
# DATA TRANSFORMATIONS (same as during training)
# ---------------------------------------
test_transforms = transforms.Compose([
    transforms.ToTensor()
])

# ---------------------------------------
# LOAD THE TEST DATASET
# ---------------------------------------
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------------------------
# LOAD THE MODEL
# ---------------------------------------
# Initialize the model architecture (same as in training)
model = models.resnet101(weights=None)  # No pretrained weights

# Replace the fully connected layer for the correct number of output classes
in_features = model.fc.in_features
num_classes = 2  # Replace with the actual number of classes
model.fc = nn.Linear(in_features, num_classes)

# Load the trained model weights
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# ---------------------------------------
# SOFTMAX FUNCTION
# ---------------------------------------
softmax = nn.Softmax(dim=1)

# ---------------------------------------
# EVALUATE SOFTMAX PROBABILITIES ON TEST DATA
# ---------------------------------------
model.eval()  # Set the model to evaluation mode

all_probs = []
all_labels = []

# Disable gradient computation during evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Apply softmax to get class probabilities
        probs = softmax(outputs)

        # Append the probabilities and true labels to the list
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenate all the probabilities and labels
all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Create a pandas DataFrame
df = pd.DataFrame(all_probs, columns=[f'Class_{i}' for i in range(num_classes)])
df['True_Label'] = all_labels

# Optionally, you can add the image paths to the DataFrame if you need
image_paths = [os.path.join(test_dir, test_dataset.imgs[i][0]) for i in range(len(test_dataset))]
df['Image_Path'] = image_paths

# Save to a CSV file
df.to_csv('softmax_predictions.csv', index=False)

print("Softmax probabilities and labels saved to softmax_predictions.csv")
