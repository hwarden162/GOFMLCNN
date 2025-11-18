import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, AutoModel
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --------------------------------------------
# CONFIG
# --------------------------------------------
TRAIN_DIR = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLCNN/image_patches/large/train"
CAL_DIR   = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLCNN/image_patches/large/calib"
TEST_DIR  = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLCNN/image_patches/large/test"

TRAIN_CSV = "foundation_embeddings/embeddings_train.csv"
CAL_CSV   = "foundation_embeddings/embeddings_calibration.csv"
TEST_CSV  = "foundation_embeddings/embeddings_test.csv"

BATCH_SIZE = 32
NUM_WORKERS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------
# LOAD MODEL + PROCESSOR (PHIKON-V2)
# --------------------------------------------
print("Loading Phikon-v2 model...")
# Get the current working directory
model_dir = os.path.join(os.getcwd(), "foundation_models")
model_name = "owkin/phikon-v2"
processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=model_dir)

# Verify that the model has been downloaded
print("Model and processor saved in:", model_dir)


# --------------------------------------------
# TRANSFORM
# --------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------------------------
# FUNCTION: LOAD DATASET
# --------------------------------------------
def get_loader(path):
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return dataset, loader, dataset.classes

# --------------------------------------------
# FUNCTION: EXTRACT EMBEDDINGS
# --------------------------------------------
def extract_embeddings(dataset, loader, classes, output_csv):
    all_embeds = []
    all_labels = []
    all_paths = []

    print(f"\nExtracting embeddings for: {output_csv}")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader)):
            # convert tensors back to PIL images (processor requirement)
            pil_images = [transforms.ToPILImage()(img) for img in images]

            # process using HuggingFace processor
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # mean-pooled transformer embeddings (1024 dims)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().numpy()

            all_embeds.append(embeddings)
            all_labels.extend(targets.numpy())

            # store original file paths
            start = batch_idx * loader.batch_size
            for i in range(len(images)):
                all_paths.append(dataset.imgs[start + i][0])

    # stack embeddings
    all_embeds = np.vstack(all_embeds)

    # create dataframe
    df = pd.DataFrame(all_embeds)
    df["label"] = [classes[i] for i in all_labels]
    df["path"] = all_paths

    # save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


# --------------------------------------------
# RUN
# --------------------------------------------

os.mkdir("foundation_embeddings")

train_ds, train_loader, train_classes = get_loader(TRAIN_DIR)
cal_ds,   cal_loader,   cal_classes   = get_loader(CAL_DIR)
test_ds,  test_loader,  test_classes  = get_loader(TEST_DIR)

extract_embeddings(train_ds, train_loader, train_classes, TRAIN_CSV)
extract_embeddings(cal_ds,   cal_loader,   cal_classes,   CAL_CSV)
extract_embeddings(test_ds,  test_loader,  test_classes,  TEST_CSV)

print("\nAll embeddings generated successfully.")
