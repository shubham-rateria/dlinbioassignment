import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from PIL import Image
from torchvision import transforms
import pickle
# import multiprocessing as mp
import time

# %%
# lets do a modality comparison first
def load_images_from_folder(folder, regex, resize=False):
    images = []
    for filename in os.listdir(folder):
        if re.match(regex, filename):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                if resize:
                    img = img.resize((128, 128))
                images.append(transforms.ToTensor()(img))
    return torch.from_numpy(np.array([np.array(img) for img in images]))

def l2_norm(image1, image2):
    return torch.norm(image1 - image2, p=2)

folder1 = '/home/sr2369/inpainting/inference/cancer_samples/sample_0_1/t2'
folder2 = '/home/sr2369/inpainting/max_area_3d_voxel/cspca/1_11_picai'
regex1 = r'.*\.png$'
regex2 = r'.*(?<!_seg)_cropped_t2\.png$'

# %%

# %%
base_cspca_sample_folder = "/home/sr2369/inpainting/inference/cancer_samples"
base_no_cspca_sample_folder = "/home/sr2369/inpainting/inference/non_cancer_samples"

base_cspca_train_folder = "/home/sr2369/inpainting/max_area_3d_voxel/cspca"
base_non_cspca_train_folder = "/home/sr2369/inpainting/max_area_3d_voxel/without_cspca"

with_cspca = os.listdir(base_cspca_train_folder)
without_cspca = os.listdir(base_non_cspca_train_folder)

print(f"{len(with_cspca)} folders with cspca")
# %%
def find_lowest_l2_for_sample(sample_dir):
    norms = []
    images1 = load_images_from_folder(sample_dir, regex1)

    for i, folder in enumerate(without_cspca):
        images2 = load_images_from_folder(os.path.join(base_non_cspca_train_folder, folder), regex=regex2, resize=True)
        data = {"train_data_folder": folder}
        for j in range(images1.shape[0]):
            slice1, slice2 = images1[j], images2[j]
            norm = l2_norm(slice1, slice2)
            data[f"slice_norm_{j}"] = norm
        norm = l2_norm(images1, images2)
        data["vol_norm"] = norm
        # print("done", sample_dir, i)
        norms.append(data)
        del images2
    
    data = {}
    
    # check volume norms
    norms.sort(reverse=False, key=lambda x: x["vol_norm"])
    smallest = norms[0]
    second_smallest = norms[1]
    ratio = smallest["vol_norm"] / second_smallest["vol_norm"]
    memorized = ratio < (1/3)
    
    data = {
        "smallest_vol_norms": norms[:3],
        "vol_ratio": ratio,
        "vol_memorized": memorized
    }
    
    # check norm per slice
    for i in range(images1.shape[0]):
        norms.sort(reverse=False, key=lambda x: x[f"slice_norm_{i}"])
        slice_smallest = norms[0]
        slice_second_smallest = norms[1]
        slice_ratio = slice_smallest[f"slice_norm_{i}"] / slice_second_smallest[f"slice_norm_{i}"]
        slice_memorized = ratio < (1/3)
        data[f"smallest_slice_norms_{i}"] = norms[:3]
        data[f"slice_ratio_{i}"] = slice_ratio
        data[f"slice_memorized_{i}"] = slice_memorized
        
    with open(os.path.join(sample_dir, 'smallest_norm_nocspca_t2.pkl'), 'wb') as f:
        pickle.dump(data, f)
# %%

def process_folder(folder):
    print(folder)
    sample_path = os.path.join(base_no_cspca_sample_folder, folder, "t2")
    start = time.time()
    find_lowest_l2_for_sample(sample_path)
    print("time taken", time.time() - start)

if __name__ == '__main__':
    # sample_folders = os.listdir("/home/sr2369/inpainting/inference/cancer_samples")
    sample_folders = os.listdir("/home/sr2369/inpainting/inference/non_cancer_samples")
    print(f"Found {len(sample_folders)}sample folders.")
    # with mp.Pool(1) as pool:
        # pool.map(process_folder, sample_folders)
    for folder in sample_folders:
        process_folder(folder)

# %%



