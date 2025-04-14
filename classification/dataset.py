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

synthetic_regex = r'.*\.png$'
real_regex = r'.*(?<!_seg)_cropped_t2\.png$'

class VoxelDataset(Dataset):
    def __init__(
        self, 
        chunks_csv_path, 
        data_type, 
        base_path, 
        transform=None, 
        return_meta=False, 
        normalize=False,
        sample_path=None,
        use_samples=False,
        num_samples_to_use=500
    ):
        self.chunks_csv_path = chunks_csv_path
        self.data_type = data_type
        self.transform = transform
        self.base_path = base_path
        self.return_meta = return_meta
        self.normalize = normalize
        
        self.sample_path = sample_path
        self.num_samples_to_use = num_samples_to_use
        
        df = pd.read_csv(chunks_csv_path)
        df.replace(-1, np.nan, inplace=True)
        df.dropna(subset=['psa', 'gland_vol', 'cancer'], inplace=True)
        
        self.image_paths = []
        
        for index, row in df.iterrows():
            base_path_classification = "cspca"
            if not row['cancer']:
                base_path_classification = "without_cspca"
            id = row['id']
            max_area_slice = int(row['pseg_max_area_slice'])
            chunk_folder = os.path.join(base_path, base_path_classification, f"{id}_{max_area_slice}_{row['dataset']}")
            data = [
                chunk_folder,
                row['cancer'],
                "real"
            ]    
            self.image_paths.append(data)
            
        if use_samples:    
            sample_cspca_dir = os.path.join(sample_path, "cancer_samples")
            sample_without_cspca_dir = os.path.join(sample_path, "non_cancer_samples")
            
            sample_cspca_folders = os.listdir(sample_cspca_dir)[:num_samples_to_use]
            sample_without_cspca_folders = os.listdir(sample_without_cspca_dir)[:num_samples_to_use]
            
            for folder in sample_cspca_folders:
                p = os.path.join(sample_cspca_dir, folder)
                data = [
                    p,
                    1,
                    "synthetic"
                ]
                self.image_paths.append(data)
            
            for folder in sample_without_cspca_folders:
                p = os.path.join(sample_without_cspca_dir, folder)
                data = [
                    p,
                    0,
                    "synthetic"
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
                
            output_image = torch.from_numpy(output_image_array)
            
            output_images.append(output_image)

        output_stacked_images = torch.stack(output_images)
        
        return output_stacked_images

    def sort_images(self, image_list):
            return sorted(image_list, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

    def __getitem__(self, idx):
        # Load images based on the path from CSV
        folder_path, cancer, sample_type = self.image_paths[idx]
        # print("getting item", idx, folder_path, cancer, sample_type)
        
        if sample_type == "real":
            if isinstance(cancer, float):
                label = cancer
            else:
                label = 1 if cancer == True else 0  # Set label based on cancer status

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
            
            # output_stacked_images = torch.stack(interleaved_images).unsqueeze(2)
            # output_stacked_images = torch.cat([t2_stacked_images, adc_stacked_images, dwi_stacked_images], dim=0)
            output_stacked_images = torch.stack([t2_stacked_images, adc_stacked_images, dwi_stacked_images])
            
            # print("output stacked", output_stacked_images.shape)
            
            return output_stacked_images, torch.tensor(label)

        if sample_type == "synthetic":
            label = cancer
            
            t2_path = os.path.join(folder_path, "t2")
            adc_path = os.path.join(folder_path, "adc")
            dwi_path = os.path.join(folder_path, "dwi")
            
            # print("getting synthetic images", t2_path)
            
            t2_images = list(glob.glob(os.path.join(t2_path, "*.png")))
            adc_images = list(glob.glob(os.path.join(adc_path, "*.png")))
            dwi_images = list(glob.glob(os.path.join(dwi_path, "*.png")))
            
            # print("getting synthetic images", os.path.basename(t2_images[0]))
            
            t2_images = self.sort_images(t2_images)
            adc_images = self.sort_images(adc_images)
            dwi_images = self.sort_images(dwi_images)
            
            # print("getting synthetic images", t2_images)

            t2_stacked_images = self.get_images(t2_images)
            adc_stacked_images = self.get_images(adc_images)
            dwi_stacked_images = self.get_images(dwi_images)

            interleaved_images = []
            
            for t2_img, adc_img, dwi_img in zip(t2_stacked_images, adc_stacked_images, dwi_stacked_images):
                interleaved_images.extend([t2_img, adc_img, dwi_img])
            
            # output_stacked_images = torch.stack(interleaved_images)
            # output_stacked_images = torch.cat([t2_stacked_images, adc_stacked_images, dwi_stacked_images], dim=0)
            output_stacked_images = torch.stack([t2_stacked_images, adc_stacked_images, dwi_stacked_images])
            
            # print("synthetic images loaded\n")
            
            return output_stacked_images, torch.tensor(label)
            
            

if __name__ == "__main__":
    print("Testing VoxelDataset")
    # test the dataset
    base_path = "../max_area_3d_voxel"
    # dataset = VoxelDataset(data_dir=base_path)
    transform = transforms.Compose([
                transforms.Resize((128, 128)),
            ])
    train_dataset = VoxelDataset(
            chunks_csv_path="/home/sr2369/inpainting/gen_train.csv",
            data_type="train",
            base_path="/home/sr2369/inpainting/max_area_3d_voxel",
            transform=transform,
            return_meta=False,
            normalize=True,
            sample_path="/home/sr2369/inpainting/inference",
            use_samples=False,
            num_samples_to_use=500
        )    
    x, y = train_dataset[0]  # get first sample
    print(f"Vol shape: {x.shape} {y.shape}")
    # print(f"Length: {len(dataset)}")
    # print(dataset[0][0].shape)
