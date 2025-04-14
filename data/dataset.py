# %%
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv
import imageio
import numpy as np
import pandas as pd

# %%

# %%
class VoxelDataset(Dataset):
    def __init__(self, chunks_csv_path, data_type, base_path, transform=None, return_meta=False, normalize=False):
        self.chunks_csv_path = chunks_csv_path
        self.data_type = data_type
        self.transform = transform
        self.base_path = base_path
        self.return_meta = return_meta
        self.normalize = normalize
        
        print("csv path", chunks_csv_path)
        
        df = pd.read_csv(chunks_csv_path)
        df.replace(-1, np.nan, inplace=True)
        df.dropna(subset=['psa', 'gland_vol', 'cancer'], inplace=True)
        
        # Filter rows where cancer is True and max_area_slice is NaN
        print(df[(df['cancer'] == True) & (df['max_area_slice'].isna())])
        
        self.image_paths = []
        for index, row in df.iterrows():
            if row['cancer'] == True and pd.isna(row['max_area_slice']):
                continue
            base_path_classification = "cspca"
            if not row['cancer']:
                base_path_classification = "without_cspca"
                max_area_slice = int(row['pseg_max_area_slice'])
            else:
                max_area_slice = int(row['max_area_slice'])
            id = row['id']
            chunk_folder = os.path.join(base_path, base_path_classification, f"{id}_{max_area_slice}_{row['dataset']}")
            data = [
                chunk_folder,
                row['cancer'],
                row['psa'],
                row['dataset_gland_vol'],
                row['age'],
            ]    
            self.image_paths.append(data)

    def __len__(self):
        return len(self.image_paths)

    def get_images(self, image_paths):
        output_images = []
        
         # Get all image files in the current subfolder
        for i, img_file in enumerate(image_paths):
            output_image = Image.open(img_file).convert("L")  
            
            if self.transform:
                output_image = self.transform(output_image)
            if self.normalize:
                output_image_array = np.array(output_image).astype(np.float32) / 127.5 - 1
            else:
                output_image_array = np.array(output_image).astype(np.float32) 
                
            output_image = torch.from_numpy(output_image_array).unsqueeze(0)
            
            output_images.append(output_image)

        output_stacked_images = torch.stack(output_images)
        
        return output_stacked_images

    def sort_images(self, image_list):
            return sorted(image_list, key=lambda x: int(os.path.basename(x).split('_')[1]))

    def __getitem__(self, idx):
        # Load images based on the path from CSV
        folder_path, cancer, psa, gland_vol, age = self.image_paths[idx]

        output_images = []
        if isinstance(cancer, float):
            label = cancer
        elif isinstance(cancer, bool):
            label = 1 if cancer else 0  # Set label based on cancer status
        else:
            label = 1 if cancer == "True" else 0  # Set label based on cancer status

        # print(idx, label, cancer, type(cancer))

        t2_images = list(glob.glob(os.path.join(folder_path, '*[!_seg]_cropped_t2.png')))
        adc_images = list(glob.glob(os.path.join(folder_path, '*[!_seg]_cropped_adc.png')))
        dwi_images = list(glob.glob(os.path.join(folder_path, '*[!_seg]_cropped_dwi.png')))
        
        t2_images = self.sort_images(t2_images)
        adc_images = self.sort_images(adc_images)
        dwi_images = self.sort_images(dwi_images)

        t2_stacked_images = self.get_images(t2_images)
        adc_stacked_images = self.get_images(adc_images)
        dwi_stacked_images = self.get_images(dwi_images)

        interleaved_images = []
        for t2_img, adc_img, dwi_img in zip(t2_stacked_images, adc_stacked_images, dwi_stacked_images):
            interleaved_images.extend([t2_img, adc_img, dwi_img])
        
        output_stacked_images = torch.stack(interleaved_images)
 
        if self.return_meta:
            return output_stacked_images, torch.tensor([label, gland_vol, psa, age], dtype=torch.float32)
        return output_stacked_images, torch.tensor([label])

if __name__ == "__main__":
    print("Testing VoxelDataset")
    # test the dataset
    base_path = "../max_area_3d_voxel"
    # dataset = VoxelDataset(data_dir=base_path)
    transform = transforms.Compose([
                transforms.Resize((128, 128)),
            ])
    dataset = VoxelDataset(chunks_csv_path="gen_val.csv", data_type="val", base_path="/home/sr2369/inpainting/max_area_3d_voxel", transform=transform)
    x, y = dataset[0]  # get first sample
    print(f"Vol shape: {x.shape} {y.shape}")
    # print(f"Length: {len(dataset)}")
    # print(dataset[0][0].shape)
