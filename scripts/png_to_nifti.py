import os
import re
import numpy as np
import nibabel as nib
import imageio.v2 as imageio

def convert_png_to_nifti(input_dir, regex_pattern, output_nifti_path):
    regex = re.compile(regex_pattern)
    
    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png') and regex.match(f)], key=lambda x: int(x.split('_')[1]))

    image_stack = []
    for png_file in png_files:
        print(png_file)
        img = imageio.imread(os.path.join(input_dir, png_file), mode='L') 
        img = np.flip(img, 1)
        img = np.rot90(img, 1)
        img = img / 255.0
        image_stack.append(img)

    image_stack = np.array(image_stack, dtype=np.float32)  
    image_stack = np.transpose(image_stack, (1, 2, 0))

    affine = np.array([[-3.42217445e-01,  2.04184628e-03, -4.49257977e-02,  1.73019684e+02],
                    [ 1.64874582e-04, -3.14033478e-01, -1.19304109e+00,  1.80923203e+02],
                    [-5.51473675e-03, -1.36095613e-01,  2.75220537e+00,  1.21240868e+02],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    nifti_img = nib.Nifti1Image(image_stack, affine=affine) 

    nib.save(nifti_img, output_nifti_path)

def list_subfolders(parent_folder):
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    return subfolders

if __name__ == "__main__":
    output_folder = "/home/sr2369/cspca/nnunet/nnUNet_raw/Dataset300_Prostate/labelsTr"
    parent_folder = "/home/sr2369/cspca/max_area_3d_voxel/without_cspca"
    subfolders = list_subfolders(parent_folder)
    print(f"\nFound {len(subfolders)} folders \n")
    for subfolder in subfolders:
        print(subfolder)
        # get folder name
        folder_name = os.path.basename(subfolder)
        case_identifier = folder_name.split("_")[0]
        output_nifti_path = os.path.join(output_folder, case_identifier + ".nii.gz")
        # convert_png_to_nifti(subfolder, r"^(?!.*_seg_cropped\.png$).*_cropped\.png$", output_nifti_path)
        convert_png_to_nifti(subfolder, r".*_seg_cropped\.png$", output_nifti_path)
    
# Example usage
# input_dir = "/home/sr2369/cspca/max_area_3d_voxel/cspca/3_13_picai"
# regex_pattern = r".*_cropped\.png$"
# output_nifti_path = "output-216.nii.gz"
# convert_png_to_nifti(input_dir, regex_pattern, output_nifti_path)