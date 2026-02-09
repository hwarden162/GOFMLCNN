import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os

num_classes = 2
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_transforms(mode):
    right_angle_rotation = transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
    ])
    if mode in ["large", "small", "masked"]:
        test_transforms = transforms.Compose([
            right_angle_rotation,
            transforms.ToTensor()
        ])
    if mode == "mask":
        class GrayscaleAndThreshold:
            def __init__(self, threshold=0):
                self.threshold = threshold
            def __call__(self, img):
                img = img.convert('L')
                img_np = np.array(img)
                img_np = (img_np > self.threshold).astype(np.float32)
                img = Image.fromarray((img_np * 255).astype(np.uint8))
                return img
        grayscale_and_threshold = GrayscaleAndThreshold(threshold=0)
        test_transforms = transforms.Compose([
            grayscale_and_threshold,
            right_angle_rotation,
            transforms.ToTensor()
        ])
    return test_transforms

def get_data_loaders(mode, test_transforms):
    cwd = os.getcwd()
    if mode == "large":
        calib_dir = os.path.join(cwd, "image_patches", "large", "calib")
        test_dir = os.path.join(cwd, "image_patches", "large", "test")
    if mode == "small":
        calib_dir = os.path.join(cwd, "image_patches", "small", "calib")
        test_dir = os.path.join(cwd, "image_patches", "small", "test")
    if mode in ["masked", "mask"]:
        calib_dir = os.path.join(cwd, "image_patches", "masked", "calib")
        test_dir = os.path.join(cwd, "image_patches", "masked", "test")
    calib_dataset = datasets.ImageFolder(calib_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {"calib": calib_loader, "test": test_loader}
    return dataloaders

def get_model_dir(mode):
    return f"saved_models/resnet101_best_{mode}.pth"

modes = ["large", "small", "masked", "mask"]
for mode in modes:
    print (f"Evaluating {mode.title()}")
    test_transforms = get_data_transforms(mode)
    dataloaders = get_data_loaders(mode, test_transforms)
    model = models.resnet101(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if mode == "mask":
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.load_state_dict(torch.load(get_model_dir(mode)))
    model = model.to(device)
    model.eval()
    calib_outs = []
    test_outs = []
    calib_labels = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in dataloaders["calib"]:
            calib_labels.append(labels.numpy())
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            calib_outs.append(torch.softmax(outputs, dim=1).numpy())
        for inputs, labels in dataloaders["test"]:
            test_labels.append(labels.numpy())
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            test_outs.append(torch.softmax(outputs, dim=1).numpy())
    os.makedirs("./softmax_outputs", exist_ok = True)
    os.makedirs(f"./softmax_outputs/{mode}", exist_ok = True)
    calib_df = pd.DataFrame(np.concat(calib_outs), columns=["bcat", "kras"])
    test_df = pd.DataFrame(np.concat(test_outs), columns=["bcat", "kras"])
    calib_df["label"] = np.concat(calib_labels)
    test_df["label"] = np.concat(test_labels)
    calib_df.to_csv(f"./softmax_outputs/{mode}/calib_softmax_{mode}.csv", index=False)
    test_df.to_csv(f"./softmax_outputs/{mode}/test_softmax_{mode}.csv", index=False)
