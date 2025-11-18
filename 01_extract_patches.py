import dask.array as da
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_PATHS = {
    "train": "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/data/full_data_train.csv",
    "calib": "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/data/full_data_calibration.csv",
    "test": "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/data/full_data_test.csv"
}
NORM_IMAGE_DIR = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/images/norm"
MASK_IMAGE_DIR = "/exports/igmm/eddie/khamseh-lab/hwarden/GOFMLPipeline/images/mask_nuc"

def make_dirs():
    os.mkdir("image_patches")
    for sd1 in ["large", "small", "masked"]:
        os.mkdir(os.path.join("image_patches", sd1))
        for sd2 in ["train", "calib", "test"]:
            os.mkdir(os.path.join("image_patches", sd1, sd2))
            for sd3 in ["bcat", "kras"]:
                os.mkdir(os.path.join("image_patches", sd1, sd2, sd3))

def generate_patches(dataset, df_path, out_type):
    df = pd.read_csv(df_path)
    image_paths = df["Meta_ImagePath"]
    image_paths = [path.split("/")[-1] for path in image_paths]
    df["Meta_ImagePath"] = image_paths
    gofs = df["GOF"]
    gofs = [label.lower() for label in gofs]
    df["GOF"] = gofs
    if out_type == "large":
        buffer = 112
    else:
        buffer = 64
    for i in tqdm(range(len(df))):
        entry = df.iloc[i, :]
        entry_out_dir = os.path.join("image_patches", out_type, dataset, entry["GOF"])
        norm_img = da.from_zarr(os.path.join(NORM_IMAGE_DIR, entry["Meta_ImagePath"], "img.zarr"))
        mask_nuc_img = da.from_zarr(os.path.join(MASK_IMAGE_DIR, entry["Meta_ImagePath"], "img.zarr"))
        centroid_y = int(entry["Meta_Nuclei_Mask_CentroidY"])
        centroid_x = int(entry["Meta_Nuclei_Mask_CentroidX"])
        start_y = centroid_y - buffer
        start_x = centroid_x - buffer
        end_y = centroid_y + buffer
        end_x = centroid_x + buffer
        norm_patch = norm_img[start_y:end_y, start_x:end_x, :].compute()
        if out_type != "masked":
            out_img = Image.fromarray(norm_patch)
        else:
            mask_nuc_patch = mask_nuc_img[start_y:end_y, start_x:end_x].compute() == entry["Meta_Global_Mask_Label"]
            out_img = norm_patch * np.stack([mask_nuc_patch, mask_nuc_patch, mask_nuc_patch], axis = -1)
            out_img = Image.fromarray(out_img)
        out_img.save(os.path.join(entry_out_dir, f"patch_{i}.png"))

make_dirs()
for dataset, df_path in DATA_PATHS.items():
    for out_type in ["large", "small", "masked"]:
        generate_patches(dataset, df_path, out_type)